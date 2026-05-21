"""v18: self-correction demos.

Different from v14 (user feedback driving revision) and v8-recovery (tool failures).
v18 = agent self-detects an error mid-trajectory: a search returns evidence that
CONTRADICTS its initial hypothesis, and it must REVISE its planned proposal
before reaching propose.

Output: data/agentic_v18/sft_demos.jsonl  (~200 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v18")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


CORRECTION_PATTERNS = [
    ("evidence_contradicts_hypothesis", "Initial rationale stated hypothesis H. First search returned evidence supporting ¬H (or a more specific case where H fails). Agent updates: 'Actually, the evidence suggests X instead, so I'll pivot to...' then proposes the revised idea."),
    ("recent_work_supersedes", "Initial rationale proposed method M. First search returned a recent paper that already does M (or strictly better). Agent updates: 'M is already done by [paper]. I'll instead propose [delta from M]...' then proposes."),
    ("scope_too_broad", "Initial rationale proposed a sweeping change. First search returned a focused recent result showing one piece works well. Agent narrows: 'Most of this is solved; the open piece is Y. I'll focus on Y.' then proposes Y."),
    ("wrong_dataset", "Initial rationale named dataset D. First search showed D is not standard for the task (or has known issues). Agent updates: 'D is the wrong benchmark; the standard is D2. I'll use D2.' then proposes."),
    ("dependency_unmet", "Initial rationale assumed condition C. First search/read showed C is not yet realized. Agent updates: 'C is not available; under realistic constraints, the right approach is C\\'. ' then proposes."),
]


DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials",
    "neuroscience", "robotics_control", "clinical_medicine",
    "economics_finance", "energy", "interdisciplinary",
]


PROMPT_GEN_TMPL = """Generate {n} research prompts in {discipline} where the agent might initially propose one direction but should pivot based on evidence. 1-3 sentences each. Output JSON array in ```json ... ``` fences."""


DEMO_SYS = """You are demonstrating a self-correcting agentic trajectory.

PATTERN: {pattern_name}
DESCRIPTION: {pattern_desc}

STRUCTURE (3-4 actions):
  1. Initial rationale states a planned direction with explicit hypothesis or method choice
  2. ```action search ... ``` to validate the plan
  3. [result] returns evidence that CONTRADICTS or SUPERSEDES the initial plan
  4. New rationale explicitly acknowledges the contradiction: "Actually..." / "Hmm, this changes things..."
  5. Optionally 1 more search/read to confirm pivot
  6. Final propose with the REVISED direction

The KEY is the explicit moment of self-correction. The agent must NAME what the contradicting evidence showed and how it's changing course.

Tools: search, read, propose. Final propose: {{"tool":"propose","gene_genome":{{6 fields}}}}."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=5,
                    help="5 patterns × 10 disc × 5 = 250 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v18/A1] generating prompts")
    calls = [TeacherCall(
        prompt_id=f"v18::{pat}::{disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_TMPL.format(n=args.n_per_combo, discipline=disc)}],
        max_tokens=1200, temperature=0.85,
        metadata={"pat": pat, "pat_desc": desc, "disc": disc},
    ) for pat, desc in CORRECTION_PATTERNS for disc in DISCIPLINES]
    t0 = time.time()
    results = batch_call(calls, workers=args.workers)
    print(f"  done in {time.time()-t0:.1f}s")

    prompts = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        md = r.metadata
        for i, q in enumerate(arr[:args.n_per_combo]):
            if not isinstance(q, str) or len(q) < 25: continue
            prompts.append({
                "prompt_id": f"v18::{md['pat']}::{md['disc']}::{i:02d}",
                "source": "synthetic_v18", "pattern": md["pat"],
                "pattern_desc": md["pat_desc"],
                "discipline": md["disc"], "lang": "en",
                "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; patterns={dict(Counter(p['pattern'] for p in prompts))}")

    print(f"[v18/A2] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        sys_msg = DEMO_SYS.format(pattern_name=p["pattern"], pattern_desc=p["pattern_desc"])
        user_msg = f"PROMPT: {p['full_prompt'][:2000]}\nDiscipline: {p['discipline']}\nSelf-correction pattern: {p['pattern']}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=2800, temperature=0.55,
            metadata={"prompt": p},
        ))

    raw_log = DEMOS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    n_valid = 0
    with DEMOS_OUT.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content or len(r.content) < 250: continue
            if r.content.count("```action") < 2 or '"propose"' not in r.content: continue
            # must show explicit self-correction keyword
            if not any(k in r.content.lower() for k in ["actually", "hmm", "wait,", "rethink", "pivot", "changes things", "contradict", "supersed"]):
                continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "kind": "self_correction", "pattern": p["pattern"],
                "discipline": p["discipline"], "lang": p["lang"],
                "archetype": "v18_self_correction",
                "topic": f"[v18/self-correct/{p['pattern']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v18. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
