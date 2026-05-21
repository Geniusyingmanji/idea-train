"""v5: short-focused + schema-aware demo expansion.

Combined v3+v4 has 77% long demos (7+ tools). For the SFT model to learn QUICK
proposing (key to SGI/ArenaRL where over-tooling kills propose rate), we need
to flip the balance: this round adds 500 demos with 60% short / 30% medium /
10% long, plus explicit schema-aware prompts that match the eval-time
distribution (gene_genome vs idea_plan vs raw answer).

Output: data/agentic_v5/sft_demos.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")

from evo_opd.teachers.gpt55_client import TeacherCall, batch_call, build_client
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v5")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


# ─── Length tiers (more aggressive short focus) ─────────────────────────────

LENGTH_TIERS = {
    "very_short": {
        "n_actions": "1-2 (direct propose OR single search + propose)",
        "guidance": "MINIMAL — propose immediately if context allows, or just one search then propose. NO read, NO extract, NO novelty. The point is FAST, decisive idea generation.",
        "weight": 0.3,
        "max_tokens": 1800,
    },
    "short": {
        "n_actions": "2-3 (search + read + propose)",
        "guidance": "Be DECISIVE — one search, optionally one read, then propose. Don't over-deliberate.",
        "weight": 0.3,
        "max_tokens": 2200,
    },
    "medium": {
        "n_actions": "4-5 (search + read + maybe extract or novelty + propose)",
        "guidance": "Moderate depth: search, read 1-2 papers, optionally extract or novelty_check, then propose.",
        "weight": 0.3,
        "max_tokens": 3000,
    },
    "long": {
        "n_actions": "6-8 (rich multi-turn)",
        "guidance": "Full agentic chain — search multiple times, read several papers, possibly extract/diff/novelty, then propose.",
        "weight": 0.1,
        "max_tokens": 4000,
    },
}


# ─── Output schemas (the agent should learn to output as instructed) ────────

OUTPUT_SCHEMAS = {
    "gene_genome": {
        "name": "gene_genome (GENE-Arena style)",
        "fields": ["mechanism_genome", "niche_genome", "observation_genome",
                   "limitation_genome", "delta_genome", "claim_genome"],
        "format_hint": '{"tool": "propose", "gene_genome": {"mechanism_genome": "...", "niche_genome": "...", ...6 fields}}',
        "weight": 0.6,
    },
    "idea_plan": {
        "name": "idea_plan (SGI-Bench style)",
        "fields": ["Idea", "ImplementationSteps", "ImplementationOrder",
                   "Dataset", "EvaluationMetrics", "ExpectedOutcome"],
        "format_hint": '{"tool": "propose", "idea_plan": {"Idea": "...", "ImplementationSteps": {"1": "...", "2": "..."}, "ImplementationOrder": ["1-2", "2-3"], "Dataset": "...", "EvaluationMetrics": {"metric": "desc"}, "ExpectedOutcome": "..."}}',
        "weight": 0.3,
    },
    "free_text_answer": {
        "name": "free-text answer (ArenaRL style)",
        "fields": ["answer (3-5 paragraphs)"],
        "format_hint": '{"tool": "propose", "answer": "<long-form 3-5 paragraph research proposal in natural language>"}',
        "weight": 0.1,
    },
}


# ─── Disciplines (full 20+ from v3+v4 combined) ─────────────────────────────

DISCIPLINES = [
    "computer_science", "physics", "chemistry", "biology", "materials",
    "mathematics", "neuroscience", "astronomy", "earth_science", "energy",
    "clinical_medicine", "pharmacology", "robotics_control", "economics_finance",
    "sociology", "agriculture_food", "urban_planning", "cognitive_science",
    "philosophy_ethics", "interdisciplinary",
]


# ─── Prompt synthesis ────────────────────────────────────────────────────────

PROMPT_GEN_TEMPLATE = """\
Generate {n} diverse research prompts in the area of {discipline}. Each should be 1-3 sentences. Output JSON array of strings inside ```json ... ``` fences."""


def build_synthetic_prompts(n_per_disc: int = 15, client=None,
                             workers: int = 12,
                             ) -> list[dict]:
    if client is None:
        client = build_client()
    calls = []
    for disc in DISCIPLINES:
        calls.append(TeacherCall(
            prompt_id=f"v5::{disc}",
            messages=[{
                "role": "user",
                "content": PROMPT_GEN_TEMPLATE.format(
                    n=n_per_disc, discipline=disc,
                ),
            }],
            max_tokens=1500,
            temperature=0.8,
            metadata={"discipline": disc},
        ))
    print(f"  dispatching {len(calls)} prompt-gen calls ({n_per_disc} × {len(DISCIPLINES)} disc = {n_per_disc * len(DISCIPLINES)} prompts)")
    t0 = time.time()
    results = batch_call(calls, workers=workers)
    print(f"  done in {time.time() - t0:.1f}s")

    out = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2026)
    tiers = list(LENGTH_TIERS.keys())
    tier_weights = [LENGTH_TIERS[t]["weight"] for t in tiers]
    schemas = list(OUTPUT_SCHEMAS.keys())
    schema_weights = [OUTPUT_SCHEMAS[s]["weight"] for s in schemas]

    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try:
            arr = json.loads(m.group(1) if m else r.content)
        except Exception:
            continue
        if not isinstance(arr, list): continue
        disc = r.metadata["discipline"]
        for i, q in enumerate(arr[:n_per_disc]):
            if not isinstance(q, str) or len(q) < 25: continue
            tier = rng.choices(tiers, weights=tier_weights)[0]
            schema = rng.choices(schemas, weights=schema_weights)[0]
            out.append({
                "prompt_id": f"v5::{disc}::{i:02d}::{tier}::{schema}",
                "source": "synthetic_v5",
                "topic": f"[{disc}/{tier}/{schema}] {q[:160]}",
                "discipline": disc,
                "length_tier": tier,
                "output_schema": schema,
                "year_min_hint": 2018,
                "year_max_hint": 2025,
                "full_prompt": q.strip(),
            })
    return out


# ─── Demo generation ────────────────────────────────────────────────────────

DEMO_SYS_TEMPLATE = """You are demonstrating an agentic research trajectory.

LENGTH TIER: {tier_name}
ACTIONS: {n_actions}
GUIDANCE: {tier_guidance}

OUTPUT SCHEMA: {schema_name}
The final `propose` action MUST emit in this exact schema:
{schema_format}
Fields: {schema_fields}

Format: for each step, write a 1-2 sentence rationale, then a ```action ... ``` block with one tool call, then a [result]: 1-3 sentence simulated tool result. Repeat until you call `propose`. The very last action MUST be `propose` with the schema above.

Available tools (use as needed for the tier):
  search: {{"tool": "search", "query": "...", "year_min": ..., "year_max": ..., "k": 5}}
  read: {{"tool": "read", "paper_id": "..."}}
  extract_genome: {{"tool": "extract_genome", "paper_id": "..."}}
  genome_diff: {{"tool": "genome_diff", "parent_id": "...", "proposed_genome": {{...}}}}
  novelty_check: {{"tool": "novelty_check", "mechanism": "...", "year_min": ..., "year_max": ...}}
  propose: see schema above

Use the real OpenAlex paper IDs from the candidates. Keep total length appropriate to the tier."""


def build_demo_user(prompt: dict, candidates: list[dict]) -> str:
    schema_info = OUTPUT_SCHEMAS[prompt["output_schema"]]
    cand_blob = ""
    if candidates:
        cand_blob = "\n\nReal OpenAlex candidates:\n"
        for i, c in enumerate(candidates[:5]):
            cand_blob += (
                f"  [{i+1}] {c['paper_id']} ({c.get('year', '?')}): "
                f"{c.get('title', '')[:120]}\n"
            )
    return f"""TOPIC:
{prompt['full_prompt'][:3000]}

Discipline: {prompt['discipline']}
Length tier: {prompt['length_tier']}
Output schema: {prompt['output_schema']}

The agent should produce a {prompt['length_tier']}-length trajectory ending with a `propose` action containing the {prompt['output_schema']} schema.
{cand_blob}"""


def prefetch_candidates(prompt: dict, search_tool: WebSearchTool, k: int = 5) -> list[dict]:
    try:
        results = search_tool.search(
            prompt["full_prompt"][:200], k=k,
            year_min=prompt.get("year_min_hint"),
            year_max=prompt.get("year_max_hint"),
        )
        return [r.to_dict() for r in results]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-disc", type=int, default=25,
                    help="prompts per discipline; 20 disc × 25 = 500 prompts")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v5/A1] synthesizing prompts")
    prompts = build_synthetic_prompts(args.n_per_disc, workers=args.workers)
    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  {len(prompts)} prompts written")

    # tier + schema distribution
    from collections import Counter
    print(f"  tiers: {dict(Counter(p['length_tier'] for p in prompts))}")
    print(f"  schemas: {dict(Counter(p['output_schema'] for p in prompts))}")

    done_ids = set()
    if args.resume and DEMOS_OUT.exists():
        with DEMOS_OUT.open() as f:
            for line in f:
                try: done_ids.add(json.loads(line)["prompt_id"])
                except: pass
        prompts = [p for p in prompts if p["prompt_id"] not in done_ids]
        print(f"  resume: {len(prompts)} remaining")

    print(f"[v5/A2] prefetching OpenAlex candidates ({len(prompts)})")
    search_tool = WebSearchTool()
    t0 = time.time()
    prefetched = {}
    for i, p in enumerate(prompts):
        prefetched[p["prompt_id"]] = prefetch_candidates(p, search_tool)
        if (i + 1) % 100 == 0:
            print(f"  prefetch {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v5/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        tier = LENGTH_TIERS[p["length_tier"]]
        schema = OUTPUT_SCHEMAS[p["output_schema"]]
        sys_msg = DEMO_SYS_TEMPLATE.format(
            tier_name=p["length_tier"],
            n_actions=tier["n_actions"],
            tier_guidance=tier["guidance"],
            schema_name=schema["name"],
            schema_format=schema["format_hint"],
            schema_fields=", ".join(schema["fields"]),
        )
        user_msg = build_demo_user(p, prefetched.get(p["prompt_id"], []))
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=tier["max_tokens"],
            temperature=0.5,
            metadata={"prompt": p, "candidates": prefetched.get(p["prompt_id"], [])},
        ))

    print(f"  dispatching {len(calls)} GPT-5.5 calls")
    raw_log = DEMOS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    n_valid = n_invalid = 0
    with DEMOS_OUT.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content or len(r.content) < 300:
                n_invalid += 1; continue
            if r.content.count("```action") < 1 or '"propose"' not in r.content:
                n_invalid += 1; continue
            demo = {
                "prompt_id": p["prompt_id"],
                "source": p["source"],
                "discipline": p["discipline"],
                "length_tier": p["length_tier"],
                "output_schema": p["output_schema"],
                "archetype": "v5_short_focused",
                "topic": p["topic"][:200],
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "latency_ms": r.latency_ms,
            }
            f.write(json.dumps(demo, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone. valid={n_valid} / total={len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
