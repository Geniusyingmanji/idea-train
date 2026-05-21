"""Build agentic_v2 SFT demonstrations with web-native tool use.

For each prompt in data/agentic_v2/prompts.jsonl, we:
  1) Pre-fetch 5 real OpenAlex candidates for the topic (so demo has real IDs).
  2) Ask GPT-5.5 to produce a single complete trajectory that uses our 6 tools
     in a sensible sequence (varies: simple search→read→propose vs deep
     search→read→extract→diff→propose etc.).
  3) Validate + save.

Cost: ~$0 (Azure keyless GPT-5.5 + free OpenAlex). Wall-clock: ~15min for
500 demos at 16 workers.

Output: data/agentic_v2/sft_demos.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

from evo_opd.agentic.rollout import ROLLOUT_SYS_PROMPT_V2
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call, build_client
from evo_opd.tools.web_search import WebSearchTool


PROMPTS_PATH = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v2/prompts.jsonl")
OUT_PATH = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v2/sft_demos.jsonl")


DEMO_TEACHER_SYS = """You are demonstrating how an agentic research model should behave. You will produce a SINGLE complete trajectory using the agent's tool protocol below:

{protocol}

Your trajectory must follow this format, repeated until you call `propose`:

  <brief 1-3 sentence rationale>
  ```action
  {{"tool": ..., ...}}
  ```
  [result]: <simulated tool result, 2-5 sentences>

After several action+result steps, end with a `propose` action that contains a complete `gene_genome` with all 6 fields.

Strict requirements:
  - Use AT MOST 6 total tool calls (mix of search/read/extract_genome/genome_diff/novelty_check/propose).
  - Vary your tool sequence across demos — some should be simple (search→read→propose), others should be richer (search→read→extract_genome→genome_diff→propose, or search→novelty_check→read→propose).
  - When simulating tool results, use the REAL OpenAlex paper candidates given below (use their oa:W... IDs and plausible abstracts).
  - The final `propose` MUST have all 6 fields non-trivially filled.
  - Keep total length under 1800 words.
  - For `[result]:` blocks, write a tight 2-5 sentence summary of what the tool would return.

You will be given a sample of REAL papers retrieved by OpenAlex for the topic — USE those as your search/read targets to keep IDs realistic."""


def build_user_for_demo(prompt: dict, candidates: list[dict]) -> str:
    cand_blob = ""
    if candidates:
        cand_blob = "\n\nReal OpenAlex candidates already retrieved for this topic (use these in your simulated search/read):\n"
        for i, c in enumerate(candidates[:5]):
            cand_blob += (
                f"\n  [{i+1}] paper_id={c['paper_id']}, year={c.get('year', '?')}\n"
                f"      title: {c.get('title', '')[:120]}\n"
                f"      snippet: {c.get('snippet', '')[:200]}\n"
            )

    return f"""TOPIC / FULL PROMPT TO THE AGENT:

{prompt['full_prompt'][:3000]}

Discipline: {prompt.get('discipline', 'general')}
Year window: {prompt.get('year_min_hint', '2018')}..{prompt.get('year_max_hint', '2025')}
Source: {prompt.get('source')}
{cand_blob}

Now write the complete agent trajectory."""


def prefetch_candidates(prompt: dict, search_tool: WebSearchTool) -> list[dict]:
    """Try to fetch 5 real OpenAlex candidates for the topic."""
    # build a search query from the topic / discipline
    topic = prompt.get("topic", "") or prompt.get("full_prompt", "")[:200]
    discipline = prompt.get("discipline", "")
    # strip discipline tag from topic
    q = topic
    if q.startswith("[") and "]" in q:
        q = q.split("]", 1)[1].strip()
    q = q[:200]
    try:
        results = search_tool.search(
            q, k=5,
            year_min=prompt.get("year_min_hint"),
            year_max=prompt.get("year_max_hint"),
        )
        return [r.to_dict() for r in results]
    except Exception as e:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=None,
                    help="None = all 478 prompts")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--resume", action="store_true",
                    help="skip prompt_ids already in OUT_PATH")
    args = ap.parse_args()

    print(f"[1/4] loading {PROMPTS_PATH}")
    prompts = []
    with PROMPTS_PATH.open() as f:
        for line in f:
            prompts.append(json.loads(line))
    if args.n_prompts:
        prompts = prompts[:args.n_prompts]
    print(f"  {len(prompts)} prompts")

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

    print(f"[2/4] pre-fetching OpenAlex candidates for {len(prompts)} prompts")
    search_tool = WebSearchTool()
    t0 = time.time()
    prefetched: dict[str, list[dict]] = {}
    for i, p in enumerate(prompts):
        cands = prefetch_candidates(p, search_tool)
        prefetched[p["prompt_id"]] = cands
        if (i + 1) % 50 == 0:
            print(f"  pre-fetch {i+1}/{len(prompts)}  ({time.time() - t0:.0f}s elapsed)")
    print(f"  done in {time.time() - t0:.1f}s")

    print(f"[3/4] building {len(prompts)} GPT-5.5 calls")
    sys_msg = DEMO_TEACHER_SYS.format(protocol=ROLLOUT_SYS_PROMPT_V2)
    calls = []
    for p in prompts:
        user_msg = build_user_for_demo(p, prefetched.get(p["prompt_id"], []))
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2200,
            temperature=0.45,
            metadata={"prompt": p, "candidates": prefetched.get(p["prompt_id"], [])},
        ))

    print(f"[4/4] dispatching (workers={args.workers}, est ~15-25 min)")
    raw_log = OUT_PATH.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda done, total: print(
            f"  [{done}/{total}] elapsed={time.time() - t0:.0f}s", flush=True
        ),
    )

    n_valid = 0
    n_invalid = 0
    with OUT_PATH.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content or len(r.content) < 200:
                n_invalid += 1
                continue
            # quality gates
            if r.content.count("```action") < 2:
                n_invalid += 1
                continue
            if '"propose"' not in r.content:
                n_invalid += 1
                continue
            demo = {
                "prompt_id": p["prompt_id"],
                "source": p.get("source"),
                "discipline": p.get("discipline"),
                "topic": p.get("topic", "")[:200],
                "full_prompt": p.get("full_prompt", ""),
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone. valid={n_valid}, invalid={n_invalid}, total={len(results)}")
    print(f"saved → {OUT_PATH}")
    print(f"raw → {raw_log}")
    print(f"elapsed: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
