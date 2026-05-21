"""v7: preference-pair data for DPO/RLHF training.

For each prompt, generate K=2 demos with EXPLICIT quality differential:
  - "chosen": well-structured, decisive, correct schema, evidence-grounded
  - "rejected": one of {rambly_overlong, premature_propose, wrong_schema,
                       no_evidence, schema_collapse, made_up_paper_ids}

Saved as {prompt, chosen, rejected, rejection_type} triplets that can later
feed DPO or pairwise RM training.

Output: data/agentic_v7/preferences.jsonl  (~300 prompts × 1 pair each)
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

from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v7")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
PAIRS_OUT = OUT_DIR / "preferences.jsonl"


DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials", "physics",
    "neuroscience", "clinical_medicine", "economics_finance", "energy",
    "robotics_control", "interdisciplinary",
]


REJECTION_MODES = {
    "premature_propose": {
        "instruction": "Skip searching/reading entirely. Jump directly to `propose` with a vague, generic gene_genome that has no specific evidence basis. The 6 fields should be one-line each, generic, not grounded in any paper.",
        "weight": 0.25,
    },
    "rambly_overlong": {
        "instruction": "Make 10+ tool calls (search 5x, read 3x, extract 2x, novelty 2x) where each step adds little new info. Repeat searches with slight variations. The final propose has decent fields but the path was wasteful.",
        "weight": 0.15,
    },
    "wrong_schema": {
        "instruction": "Use gene_genome schema for a prompt that explicitly asks for idea_plan, OR use idea_plan when asked for gene_genome. Other elements (rationale, evidence) are fine — just the final schema is wrong.",
        "weight": 0.20,
    },
    "no_evidence": {
        "instruction": "Make tool calls (search/read) but the final propose's gene_genome/idea_plan fields are unrelated to what the search/read returned. The agent essentially ignores its own retrieved evidence.",
        "weight": 0.15,
    },
    "schema_collapse": {
        "instruction": "The final `propose` action emits a JSON-like blob that is MALFORMED — missing fields, extra fields, fields shoved into a single string, or the gene_genome object replaced by a free-text paragraph.",
        "weight": 0.10,
    },
    "made_up_papers": {
        "instruction": "Use FAKE paper_ids like oa:W9999999 or paper:fake-2024 in search/read steps. The candidate list is ignored. Final propose is OK but built on hallucinated evidence.",
        "weight": 0.15,
    },
}


PROMPT_GEN_TEMPLATE = """Generate {n} diverse research prompts in the area of {discipline}. Some should EXPLICITLY ask for gene_genome output, others for idea_plan, others left implicit. 1-3 sentences each. Output JSON array inside ```json ... ``` fences."""


SCHEMA_GUIDE = """\
gene_genome schema: 6 fields [mechanism_genome, niche_genome, observation_genome, limitation_genome, delta_genome, claim_genome]
idea_plan schema: [Idea, ImplementationSteps (dict), ImplementationOrder (list), Dataset, EvaluationMetrics (dict), ExpectedOutcome]
"""


CHOSEN_SYS = f"""You are demonstrating a HIGH-QUALITY agentic research trajectory.

Length: 3-5 actions (search, optionally read 1-2, optionally extract or novelty, then propose).
Be decisive, evidence-grounded, schema-correct.

{SCHEMA_GUIDE}

Format each step as: rationale (1-2 sentences) + ```action ... ``` JSON tool call + [result]: simulated tool result (1-3 sentences). End with `propose` in the schema the prompt requests (default gene_genome).

Tools: search, read, extract_genome, genome_diff, novelty_check, propose.

Use real paper_ids from candidates. Ground the final propose's fields in what you actually retrieved."""


REJECTED_SYS_TMPL = f"""You are demonstrating a LOW-QUALITY agentic trajectory that exhibits a specific failure mode.

FAILURE MODE: {{rejection_type}}
INSTRUCTION: {{rejection_instr}}

The trajectory should still LOOK like an agentic research attempt — JSON action blocks, simulated results, ending in `propose`. But it must exhibit the specific failure above.

{SCHEMA_GUIDE}

Format: rationale + ```action ... ``` + [result], ending with `propose`. Tools: search, read, extract_genome, genome_diff, novelty_check, propose."""


def gen_prompts(n_per_disc: int, workers: int) -> list[dict]:
    calls = [TeacherCall(
        prompt_id=f"v7::{disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_TEMPLATE.format(n=n_per_disc, discipline=disc)}],
        max_tokens=1200, temperature=0.85,
        metadata={"discipline": disc},
    ) for disc in DISCIPLINES]
    print(f"  dispatching {len(calls)} prompt-gen calls")
    t0 = time.time()
    results = batch_call(calls, workers=workers)
    print(f"  done in {time.time()-t0:.1f}s")
    out = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2026)
    modes = list(REJECTION_MODES.keys())
    mode_weights = [REJECTION_MODES[m]["weight"] for m in modes]
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        for i, q in enumerate(arr[:n_per_disc]):
            if not isinstance(q, str) or len(q) < 25: continue
            mode = rng.choices(modes, weights=mode_weights)[0]
            out.append({
                "prompt_id": f"v7::{r.metadata['discipline']}::{i:02d}::{mode}",
                "discipline": r.metadata["discipline"],
                "rejection_mode": mode,
                "year_min_hint": 2018, "year_max_hint": 2025,
                "full_prompt": q.strip(),
            })
    return out


def prefetch_candidates(prompt, st):
    try:
        rs = st.search(prompt["full_prompt"][:200], k=5,
                       year_min=prompt.get("year_min_hint"),
                       year_max=prompt.get("year_max_hint"))
        return [r.to_dict() for r in rs]
    except: return []


def build_user(prompt, candidates):
    cb = ""
    if candidates:
        cb = "\n\nCandidates:\n"
        for i, c in enumerate(candidates[:5]):
            cb += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
    return f"PROMPT: {prompt['full_prompt'][:2500]}\nDiscipline: {prompt['discipline']}{cb}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-disc", type=int, default=30,
                    help="~11 disc × 30 = 330 prompts → 330 pairs")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v7/A1] synthesizing prompts")
    prompts = gen_prompts(args.n_per_disc, args.workers)
    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; rejection_modes={dict(Counter(p['rejection_mode'] for p in prompts))}")

    done = set()
    if args.resume and PAIRS_OUT.exists():
        with PAIRS_OUT.open() as f:
            for line in f:
                try: done.add(json.loads(line)["prompt_id"])
                except: pass
        prompts = [p for p in prompts if p["prompt_id"] not in done]

    print(f"[v7/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {p["prompt_id"]: prefetch_candidates(p, st) for p in prompts}
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v7/A3] generating chosen + rejected pairs (workers={args.workers})")
    chosen_calls, rejected_calls = [], []
    for p in prompts:
        user_msg = build_user(p, prefetched.get(p["prompt_id"], []))
        chosen_calls.append(TeacherCall(
            prompt_id=f"{p['prompt_id']}::chosen",
            messages=[
                {"role": "system", "content": CHOSEN_SYS},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=2800, temperature=0.4,
            metadata={"prompt": p, "candidates": prefetched.get(p["prompt_id"], []), "kind": "chosen"},
        ))
        rej_info = REJECTION_MODES[p["rejection_mode"]]
        rej_sys = REJECTED_SYS_TMPL.format(
            rejection_type=p["rejection_mode"],
            rejection_instr=rej_info["instruction"],
        )
        rejected_calls.append(TeacherCall(
            prompt_id=f"{p['prompt_id']}::rejected",
            messages=[
                {"role": "system", "content": rej_sys},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=3000, temperature=0.7,
            metadata={"prompt": p, "candidates": prefetched.get(p["prompt_id"], []), "kind": "rejected"},
        ))

    all_calls = chosen_calls + rejected_calls
    print(f"  dispatching {len(all_calls)} calls ({len(chosen_calls)} chosen + {len(rejected_calls)} rejected)")
    raw_log = PAIRS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        all_calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    by_kind = {"chosen": {}, "rejected": {}}
    for r in results:
        if r.error or not r.content or len(r.content) < 200: continue
        kind = r.metadata["kind"]
        p = r.metadata["prompt"]
        by_kind[kind][p["prompt_id"]] = (r, p)

    pairs_out = []
    for pid, (cr, p) in by_kind["chosen"].items():
        if pid not in by_kind["rejected"]: continue
        rr, _ = by_kind["rejected"][pid]
        # extra sanity: chosen should at least have action+propose
        if cr.content.count("```action") < 1 or '"propose"' not in cr.content: continue
        pairs_out.append({
            "prompt_id": pid,
            "discipline": p["discipline"],
            "rejection_mode": p["rejection_mode"],
            "full_prompt": p["full_prompt"],
            "candidates": cr.metadata.get("candidates", []),
            "chosen": cr.content,
            "rejected": rr.content,
            "chosen_output_tokens": cr.output_tokens,
            "rejected_output_tokens": rr.output_tokens,
        })

    with PAIRS_OUT.open("a") as f:
        for pair in pairs_out: f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\nDone v7. pairs={len(pairs_out)}/{len(prompts)}  saved → {PAIRS_OUT}")


if __name__ == "__main__":
    main()
