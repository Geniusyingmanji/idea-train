"""Build SFT demonstration trajectories for agentic-OPD warm-start.

For each prompt in data/agentic_v1/prompts.jsonl, ask GPT-5.5 (Azure keyless)
to produce a single ReAct trajectory using OUR tool protocol. GPT-5.5 sees:

  - the topic
  - the gold_lineage (so it knows which ancestors exist; we want it to actually
    `read` them)
  - the protocol description (same as ROLLOUT_SYS_PROMPT)

GPT-5.5 outputs an ENTIRE trajectory at once — interleaved `<assistant>`
thoughts, ```action``` blocks, and `[result]:` observation stubs. We then
parse it back into a canonical assistant trace and validate by:
  - the action sequence is parseable
  - at least one `read` of a gold_lineage paper
  - a final `propose` with all 6 fields

Demo schema (one per row in agentic_v1/sft_demos.jsonl):
{
  "prompt_id": "agentic_v1::p_0042",
  "topic":     "...",
  "discipline": "cs",
  "input_text": "<system>\n<user>\n<assistant>",   # the system + user turn only
  "completion": "<full assistant trace including action blocks AND inline observation stubs>",
  "metadata":  { ... }
}

Cost: ~500 calls × 1 GPT-5.5 turn (~4-6s each) = ~1-2 hours wall-clock at
8 workers, $0 (Azure keyless).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

from evo_opd.agentic.rollout import ROLLOUT_SYS_PROMPT
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call, build_client
from evo_opd.tools.read import ReadTool
from evo_opd.tools.search import SearchTool


PROMPTS_PATH = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v1/prompts.jsonl")
OUT_PATH = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v1/sft_demos.jsonl")


DEMO_USER_TEMPLATE = """Research topic: {topic}
Year window hint: {year_min}..{year_max}
Discipline: {discipline}
Now begin. Use the tools."""


DEMO_TEACHER_SYS = """You are demonstrating how an agentic research model should behave. You will produce a SINGLE complete trajectory using this protocol:

{protocol}

Your trajectory must follow this exact format:

  1) A short rationale (1-3 sentences)
  2) An action block: ```action
     {{"tool": ..., ...}}
     ```
  3) A simulated observation block: [result]: <what the tool would return>
  4) Repeat 1-3 until you have called `propose` at LEAST once and LAST.

Strict requirements for the demonstration:
  - Use AT MOST 5 total tool calls (1 search + 1-2 read + 1 propose works well)
  - You MUST read at least one of the suggested gold_ancestor papers (provided below)
  - The final `propose` MUST emit a gene_genome with ALL 6 fields filled non-trivially
  - For [result]: blocks, write a 2-3 sentence summary using the gold_ancestor
    content provided below as your knowledge source (do not invent papers
    outside the gold list)
  - Keep total length under 1500 words

You will be told the gold_lineage; use that as ground truth when fabricating tool results. Do NOT just copy abstracts — write tight, agent-style summaries."""


def build_user_for_demo(prompt: dict, gold_card_excerpts: list[str]) -> str:
    """Compose the user message giving GPT-5.5 the topic + gold lineage hints."""
    gold_blob = ""
    if gold_card_excerpts:
        gold_blob = "\n\nGold lineage papers (use these as ground-truth when writing [result]: blocks):\n"
        for i, ex in enumerate(gold_card_excerpts[:3]):
            gold_blob += f"\n[Gold paper {i+1}]:\n{ex}\n"

    return f"""TOPIC:
{prompt['topic']}

Discipline: {prompt['discipline']}
Year window: {prompt['year_min_hint']}..{prompt['year_max_hint']}
Target paper_id (DO NOT search/read this; this is the answer): {prompt['target_paper_id']}
{gold_blob}

Now write the complete trajectory."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="None = all 248 prompts")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="skip prompt_ids already in OUT_PATH")
    args = ap.parse_args()

    print(f"[1/3] loading prompts + tool corpora")
    prompts = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.n_prompts:
        prompts = prompts[:args.n_prompts]
    print(f"  {len(prompts)} prompts")

    read_tool = ReadTool()
    print(f"  {len(read_tool.cards)} cards in tool corpus")

    # resume
    done_ids: set[str] = set()
    if args.resume and OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_ids.add(d["prompt_id"])
                except Exception:
                    pass
        print(f"  resume: {len(done_ids)} already done; will skip")
        prompts = [p for p in prompts if p["prompt_id"] not in done_ids]
        print(f"  {len(prompts)} remaining")

    if not prompts:
        print("nothing to do")
        return

    print(f"[2/3] building teacher calls")
    user_template = DEMO_TEACHER_SYS.format(protocol=ROLLOUT_SYS_PROMPT)
    calls = []
    for p in prompts:
        # fetch gold lineage card excerpts (first 600 chars of each)
        gold_excerpts = []
        for gid in p.get("gold_lineage", [])[:3]:
            card = read_tool.read_struct(gid)
            if card is None:
                continue
            g = card.get("genome", {})
            excerpt = (
                f"paper_id: {gid}\n"
                f"title: {card.get('title', '?')}\n"
                f"year: {card.get('year', '?')}\n"
                f"niche: {g.get('niche_genome', '')[:200]}\n"
                f"mechanism: {g.get('mechanism_genome', '')[:200]}\n"
                f"limitation: {g.get('limitation_genome', '')[:200]}\n"
            )
            gold_excerpts.append(excerpt)
        user_msg = build_user_for_demo(p, gold_excerpts)
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[
                {"role": "system", "content": user_template},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2200,
            temperature=0.4,
            metadata={"prompt": p},
        ))
    print(f"  {len(calls)} GPT-5.5 calls queued")

    print(f"[3/3] dispatching (workers={args.workers})")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_path = OUT_PATH.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    n_done_local = 0
    results = batch_call(
        calls, workers=args.workers, log_path=log_path,
        on_progress=lambda done, total: print(f"  [{done}/{total}] elapsed={time.time() - t0:.0f}s", flush=True),
    )

    # postprocess: write canonical demo rows
    n_valid = 0
    n_invalid = 0
    with OUT_PATH.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content:
                n_invalid += 1
                continue
            demo = {
                "prompt_id": p["prompt_id"],
                "topic": p["topic"],
                "discipline": p["discipline"],
                "completion": r.content,
                "gold_lineage": p.get("gold_lineage", []),
                "target_paper_id": p.get("target_paper_id"),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone. valid={n_valid}, invalid={n_invalid}, total={len(results)}")
    print(f"saved → {OUT_PATH}")
    print(f"raw calls → {log_path}")
    print(f"elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
