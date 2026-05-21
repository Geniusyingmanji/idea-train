"""ReAct rollout loop for agentic-OPD.

Single-process, sequential per trajectory. The model generates a "thinking +
action" block; we parse the action, execute the tool, append the observation
to the prompt, and continue. Tracks per-token role tags so the trainer can
mask out observation tokens.

Action format (parsed leniently from the model's output):
```action
{"tool": "search", "query": "...", "year_min": 2018, "year_max": 2024, "k": 5}
```
```action
{"tool": "read", "paper_id": "paper:foo:2021"}
```
```action
{"tool": "propose", "gene_genome": {"mechanism_genome": "...", "niche_genome": "...", ...}}
```

The system prompt explains the protocol.
"""
from __future__ import annotations

import json
import re
import time
from typing import Optional

import torch

from ..tools.propose import parse_propose_action
from ..tools.read import ReadTool
from ..tools.search import SearchTool
from .trajectory import ActionStep, TokenSpan, Trajectory, TokenRole

# v2 imports — new tools are optional (None disables them)
try:
    from ..tools.genome_tool import GenomeExtractTool, format_genome_observation
    from ..tools.diff_tool import GenomeDiffTool, format_diff_observation
    from ..tools.novelty_tool import NoveltyCheckTool, format_novelty_observation
    _V2_TOOLS_AVAILABLE = True
except Exception:
    _V2_TOOLS_AVAILABLE = False


ROLLOUT_SYS_PROMPT = """You are a scientific research agent. Given a research topic, you must:
  1) Search the local paper corpus for relevant prior work.
  2) Read one or more promising papers to understand the lineage.
  3) Propose a novel follow-up research idea grounded in what you read.

You have three tools. Each tool call is ONE JSON object inside a fenced ```action ... ``` block. Only one tool call per turn. After each action, you will see a tool result. Use that result to decide the next action.

Tools:
  search  → ```action
  {"tool": "search", "query": "<short keyword phrase>", "year_min": <int|null>, "year_max": <int|null>, "k": 5}
  ```
  read    → ```action
  {"tool": "read", "paper_id": "<paper_id_from_search>"}
  ```
  propose → ```action
  {"tool": "propose", "gene_genome": {
    "mechanism_genome": "<concrete proposed method>",
    "niche_genome": "<problem domain>",
    "observation_genome": "<expected empirical result>",
    "limitation_genome": "<acknowledged limitation>",
    "delta_genome": "<what this changes vs prior work>",
    "claim_genome": "<the main hypothesis>"
  }}
  ```

Strict rules:
  - Use AT MOST 6 tool calls total.
  - You MUST call `propose` exactly once, and it MUST be your LAST action.
  - You MUST `read` at least one paper before proposing.
  - Write a brief 1-3 sentence rationale before each action.
"""


ROLLOUT_SYS_PROMPT_V2 = """You are a scientific research agent. Given a research topic or question, you must:
  1) Search the literature for relevant prior work.
  2) Read promising papers to understand the lineage.
  3) (Optionally) Extract structured genomes from papers to reason about inheritance.
  4) Propose a novel follow-up research idea grounded in what you read.
  5) (Optionally) Validate your proposal against parent papers / against novelty.

You have SIX tools. Each tool call is ONE JSON object inside a fenced ```action ... ``` block. Only one tool call per turn. After each action, you will see a tool result. Use it to decide the next action.

Tools:

  1) search — query OpenAlex (or local corpus) for related papers
     ```action
     {"tool": "search", "query": "<short keyword phrase>", "year_min": <int|null>, "year_max": <int|null>, "k": 5}
     ```

  2) read — fetch full title + abstract + metadata for a paper
     ```action
     {"tool": "read", "paper_id": "<id_from_search_e.g._oa:W123>"}
     ```

  3) extract_genome — get the structured 6-field gene_genome of a paper
       (NOT just raw abstract — returns mechanism/niche/limitation/etc. atoms)
     ```action
     {"tool": "extract_genome", "paper_id": "<oa:W... or paper:foo:2024>"}
     ```

  4) genome_diff — compare a proposed_genome against a parent paper's genome
       Returns gene fates (INHERITED/MUTATED/LOST/NOVEL/HYBRIDIZED) + dynamics
     ```action
     {"tool": "genome_diff", "parent_id": "<paper_id>", "proposed_genome": {
       "mechanism_genome": "...", "niche_genome": "...", ...
     }}
     ```

  5) novelty_check — find papers most similar to a proposed mechanism
       Returns verdict: "redundant" / "disconnected" / "healthy"
     ```action
     {"tool": "novelty_check", "mechanism": "<one-sentence description>", "year_min": <int|null>, "year_max": <int|null>}
     ```

  6) propose — terminal action: emit final gene_genome
     ```action
     {"tool": "propose", "gene_genome": {
       "mechanism_genome": "<concrete proposed method>",
       "niche_genome": "<problem domain>",
       "observation_genome": "<expected empirical result>",
       "limitation_genome": "<acknowledged limitation>",
       "delta_genome": "<what this changes vs prior work>",
       "claim_genome": "<the main hypothesis>"
     }}
     ```

Strict rules:
  - Use AT MOST 8 tool calls total.
  - You MUST call `propose` exactly once, and it MUST be your LAST action.
  - You SHOULD `read` at least one paper before proposing.
  - You MAY use extract_genome / genome_diff / novelty_check to deepen reasoning.
  - Write a 1-3 sentence rationale before each action.
"""


# Each turn the model generates a block ending with a ```action ... ``` JSON.
# Stop generation when we see the closing ``` of an action block.
_ACTION_BLOCK_RE = re.compile(
    r"```action\s*(\{.*?\})\s*```", re.DOTALL,
)


def _parse_action(text: str) -> tuple[str, dict, str]:
    """Find the FIRST action block in text. Return (tool, args, raw_action_text).

    Returns ("malformed", {}, "") if no parseable block found.
    """
    m = _ACTION_BLOCK_RE.search(text)
    if not m:
        return "malformed", {}, ""
    blob = m.group(1).strip()
    try:
        args = json.loads(blob)
    except Exception:
        return "malformed", {}, m.group(0)
    if not isinstance(args, dict):
        return "malformed", {}, m.group(0)
    tool = str(args.get("tool", "")).strip().lower()
    if tool not in ("search", "read", "propose",
                     "extract_genome", "genome_diff", "novelty_check"):
        return "malformed", args, m.group(0)
    return tool, args, m.group(0)


def _format_search_result(results) -> str:
    if not results:
        return "[result]: 0 papers found. Try a different query."
    lines = ["[result]: " + str(len(results)) + " papers:"]
    for r in results:
        y = r.year if r.year else "?"
        lines.append(f"  - {r.paper_id} ({y}): {r.title[:120]}")
        if r.snippet:
            lines.append(f"      snippet: {r.snippet[:160]}")
    return "\n".join(lines)


def _format_read_result(card_text: str) -> str:
    return f"[result]:\n{card_text[:1800]}"


def _format_error(msg: str) -> str:
    return f"[error]: {msg}. Try again with a corrected action."


def run_rollout(
    *,
    model, tokenizer, device: str,
    prompt: dict,
    search_tool: SearchTool, read_tool: ReadTool,
    max_turns: int = 6,
    max_new_tokens_per_turn: int = 384,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_total_tokens: int = 6144,
    exclude_target_from_search: bool = True,
    # v2 optional tools — pass None to disable each
    extract_tool: "GenomeExtractTool | None" = None,
    diff_tool: "GenomeDiffTool | None" = None,
    novelty_tool: "NoveltyCheckTool | None" = None,
    system_prompt: str | None = None,
) -> Trajectory:
    """Generate one trajectory by interleaving model generation and tool calls.

    Returns a Trajectory with full_ids, token_roles, gen_mask filled in for
    downstream PG loss masking. Observations are appended to the model's
    context as plain text (marked role='observation' so gradient is masked).
    """
    t_start = time.time()
    topic = prompt.get("topic", "(no topic)")
    # v2 prompts have `full_prompt` (the actual benchmark question / arena prompt).
    # v1 prompts only have `topic`; we wrap it minimally.
    if prompt.get("full_prompt"):
        user_prompt = (
            f"{prompt['full_prompt']}\n\n"
            f"Discipline: {prompt.get('discipline', 'any')}\n"
            f"Year window hint: {prompt.get('year_min_hint', 'any')}..{prompt.get('year_max_hint', 'any')}\n"
            "Now begin. Use the tools."
        )
    else:
        user_prompt = (
            f"Research topic: {topic}\n"
            f"Year window hint: {prompt.get('year_min_hint', 'any')}..{prompt.get('year_max_hint', 'any')}\n"
            f"Discipline: {prompt.get('discipline', 'any')}\n"
            "Now begin. Use the tools."
        )
    sys_prompt_text = system_prompt or ROLLOUT_SYS_PROMPT
    messages = [
        {"role": "system", "content": sys_prompt_text},
        {"role": "user", "content": user_prompt},
    ]
    try:
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except (TypeError, ValueError):
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", truncation=True,
                            max_length=2048).input_ids[0].tolist()

    traj = Trajectory(
        prompt_id=prompt["prompt_id"],
        topic=topic,
        discipline=prompt.get("discipline", ""),
    )
    # initialize token streams with the prompt
    traj.full_ids = list(prompt_ids)
    traj.token_roles = ["prompt"] * len(prompt_ids)
    traj.gen_mask = [False] * len(prompt_ids)

    denylist_search: set[str] = set()  # papers already read; don't re-surface
    if exclude_target_from_search and prompt.get("target_paper_id"):
        denylist_search.add(prompt["target_paper_id"])
    raw_chunks: list[str] = []

    # CRITICAL: switch to eval mode for generation so LoRA dropout doesn't
    # corrupt outputs. We restore train mode at the end of the rollout so the
    # caller's PG backward pass works correctly.
    was_training = model.training
    model.eval()

    for turn in range(max_turns):
        if len(traj.full_ids) >= max_total_tokens:
            traj.truncated = True
            break

        # generate next assistant chunk (until ```action block closes or eos)
        inputs = torch.tensor([traj.full_ids], device=device)
        attn_mask = torch.ones_like(inputs)
        with torch.no_grad():
            out_ids = model.generate(
                inputs,
                attention_mask=attn_mask,
                max_new_tokens=max_new_tokens_per_turn,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_tail_full = out_ids[0, inputs.shape[1]:].tolist()
        if not gen_tail_full:
            traj.truncated = True
            break

        gen_chunk_text_full = tokenizer.decode(gen_tail_full, skip_special_tokens=True)

        # truncate at end of FIRST action block (or first sign of repeat action)
        m_first = _ACTION_BLOCK_RE.search(gen_chunk_text_full)
        if m_first:
            cutoff = m_first.end()
            chunk_text = gen_chunk_text_full[:cutoff]
            # re-tokenize the truncated chunk to keep token ids aligned
            gen_tail = tokenizer(chunk_text, add_special_tokens=False).input_ids
        else:
            # no action block found — keep all of it, will be marked as malformed
            chunk_text = gen_chunk_text_full
            gen_tail = gen_tail_full

        raw_chunks.append(chunk_text)
        traj.n_generated_tokens += len(gen_tail)

        # append generated tokens to full_ids with role='action' for everything
        # (we don't separate thought-vs-action token-level — both are generated
        # by π_θ and both should receive gradient)
        traj.full_ids.extend(gen_tail)
        traj.token_roles.extend(["action"] * len(gen_tail))
        traj.gen_mask.extend([True] * len(gen_tail))

        # parse action
        tool, args, raw_action = _parse_action(chunk_text)
        step = ActionStep(turn=turn, tool=tool, action_args=args, raw_text=chunk_text)

        if tool == "malformed":
            step.observation_text = _format_error("could not parse tool call. Use ```action\\n{\"tool\": ...}\\n``` exactly.")
            traj.malformed_count += 1
        elif tool == "search":
            q = str(args.get("query", "")).strip()
            ymin = args.get("year_min")
            ymax = args.get("year_max")
            k = int(args.get("k", 5) or 5)
            try:
                results = search_tool.search(
                    q, k=min(k, 5),
                    year_min=int(ymin) if ymin not in (None, "", "null") else None,
                    year_max=int(ymax) if ymax not in (None, "", "null") else None,
                    denylist_paper_ids=denylist_search,
                )
                step.observation_text = _format_search_result(results)
                traj.search_count += 1
            except Exception as e:
                step.observation_text = _format_error(f"search failed: {type(e).__name__}: {e}")
        elif tool == "read":
            pid = str(args.get("paper_id", "")).strip()
            canon = read_tool._resolve_id(pid)
            if canon is not None:
                step.observation_text = _format_read_result(read_tool.read(canon))
                traj.read_count += 1
                traj.read_paper_ids.append(canon)
                denylist_search.add(canon)
            else:
                step.observation_text = _format_error(
                    f"paper_id {pid!r} not in corpus. Use a paper_id from a recent search result."
                )
        elif tool == "extract_genome":
            if extract_tool is None:
                step.observation_text = _format_error("extract_genome tool is not enabled in this rollout")
            else:
                pid = str(args.get("paper_id") or args.get("text") or "").strip()
                if not pid:
                    step.observation_text = _format_error("extract_genome requires paper_id or text")
                else:
                    try:
                        res = extract_tool.extract(pid)
                        step.observation_text = format_genome_observation(res)
                        traj.extract_count += 1
                    except Exception as e:
                        step.observation_text = _format_error(
                            f"extract_genome failed: {type(e).__name__}: {e}"
                        )
        elif tool == "genome_diff":
            if diff_tool is None:
                step.observation_text = _format_error("genome_diff tool is not enabled in this rollout")
            else:
                parent_id = str(args.get("parent_id", "")).strip()
                proposed = args.get("proposed_genome", {})
                if not parent_id or not isinstance(proposed, dict):
                    step.observation_text = _format_error(
                        "genome_diff requires parent_id and proposed_genome dict"
                    )
                else:
                    try:
                        res = diff_tool.diff(parent_id, proposed)
                        step.observation_text = format_diff_observation(res)
                        traj.diff_count += 1
                    except Exception as e:
                        step.observation_text = _format_error(
                            f"genome_diff failed: {type(e).__name__}: {e}"
                        )
        elif tool == "novelty_check":
            if novelty_tool is None:
                step.observation_text = _format_error("novelty_check tool is not enabled in this rollout")
            else:
                mech = str(args.get("mechanism") or args.get("query") or "").strip()
                ymin = args.get("year_min")
                ymax = args.get("year_max")
                if not mech:
                    step.observation_text = _format_error("novelty_check requires mechanism text")
                else:
                    try:
                        res = novelty_tool.check(
                            mech, k=8,
                            year_min=int(ymin) if ymin not in (None, "", "null") else None,
                            year_max=int(ymax) if ymax not in (None, "", "null") else None,
                        )
                        step.observation_text = format_novelty_observation(res)
                        traj.novelty_count += 1
                    except Exception as e:
                        step.observation_text = _format_error(
                            f"novelty_check failed: {type(e).__name__}: {e}"
                        )
        elif tool == "propose":
            # propose is terminal
            gene_genome = args.get("gene_genome")
            if isinstance(gene_genome, dict):
                traj.final_proposal = gene_genome
                step.parsed_proposal = gene_genome
            else:
                # try to parse from raw text (handles wrong shape)
                traj.final_proposal = parse_propose_action(raw_action) or {}
                step.parsed_proposal = traj.final_proposal
            traj.propose_emitted = True
            step.observation_text = "[result]: proposal accepted. Trajectory complete."
            traj.actions.append(step)
            break

        traj.actions.append(step)
        # append observation as plain text (masked: not generated by π_θ)
        obs_text = f"\n{step.observation_text}\n"
        obs_ids = tokenizer(obs_text, return_tensors="pt",
                             add_special_tokens=False).input_ids[0].tolist()
        traj.full_ids.extend(obs_ids)
        traj.token_roles.extend(["observation"] * len(obs_ids))
        traj.gen_mask.extend([False] * len(obs_ids))

    if not traj.propose_emitted:
        traj.truncated = True

    traj.raw_text = "".join(raw_chunks)
    traj.wall_time_s = time.time() - t_start

    # restore the caller's train/eval state
    if was_training:
        model.train()

    return traj


if __name__ == "__main__":
    # Mini smoke: no model, just verify tool wiring with mocked generations.
    # (Real model smoke happens in evo_opd/trainer/agentic_loop.py)
    search_tool = SearchTool()
    read_tool = ReadTool()
    print(f"tools loaded: search={len(search_tool.docs)} read={len(read_tool.cards)}")

    # Test action parsing
    test_chunks = [
        'Thinking: I should search.\n```action\n{"tool": "search", "query": "diffusion molecule"}\n```',
        '```action\n{"tool": "read", "paper_id": "paper:foo:2021"}\n```',
        'Now I propose:\n```action\n{"tool": "propose", "gene_genome": {"mechanism_genome": "X"}}\n```',
        'malformed reply with no action',
    ]
    for c in test_chunks:
        tool, args, raw = _parse_action(c)
        print(f"  tool={tool}  args={args}")
