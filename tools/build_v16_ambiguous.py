"""v16: ambiguous-prompt / clarification demos.

Most prior demos assume clear, well-formed prompts. Real prompts are often
ambiguous — under-specified domain, conflicting constraints, vague target.

v16 demos teach the agent to:
  (a) make assumptions EXPLICIT in the rationale,
  (b) propose with the most reasonable interpretation,
  (c) note the assumption in the proposal's niche or limitation.

Output: data/agentic_v16/sft_demos.jsonl (~250 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v16")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


AMBIGUITY_TYPES = [
    ("vague_domain", "Domain is vague: prompt mentions a broad area (e.g., 'AI for healthcare') without specifying disease/method/data."),
    ("conflicting_goals", "Two goals conflict: e.g., 'maximize accuracy AND minimize compute' without saying which dominates."),
    ("undefined_baseline", "Asks for an 'improvement' without specifying what's being improved over."),
    ("ambiguous_target", "Target metric/dataset/timeframe is left implicit."),
    ("multiple_interpretations", "The prompt could be interpreted as proposing method X or method Y; both are valid."),
    ("missing_constraints", "Compute/data/expertise constraints aren't stated."),
]


DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials",
    "neuroscience", "robotics_control", "clinical_medicine",
    "economics_finance", "energy", "interdisciplinary",
]


PROMPT_GEN_TMPL = """Generate {n} INTENTIONALLY AMBIGUOUS research prompts in {discipline} that exhibit the ambiguity type "{amb_type}". {amb_desc}

The prompts should be REALISTIC — the kind a real researcher or learner might actually ask. Each 1-3 sentences.

Output JSON array inside ```json ... ``` fences."""


DEMO_SYS = """You are demonstrating a clarification-first agentic trajectory.

The prompt is AMBIGUOUS in this specific way: {amb_type}

Your trajectory should:
  1. First rationale (1-2 sentences): explicitly NAME the ambiguity. e.g., "The prompt doesn't specify X; I'll assume Y because Z."
  2. Optionally 1 search to confirm assumption.
  3. Final propose: in the propose's gene_genome, make the assumption visible — usually in niche_genome (the scope) or limitation_genome (what's NOT covered under this assumption).

Length: 2-3 actions total. Format: rationale + ```action ... ``` + [result] + final propose. Tools: search, propose.

Final propose schema: {{"tool":"propose","gene_genome":{{6 fields, with assumption explicit}}}}.

The KEY learning: the rationale should NEVER pretend the prompt is unambiguous. Always surface what you're assuming."""


def gen_prompts(n_per_combo, workers):
    return [TeacherCall(
        prompt_id=f"v16::{amb_type}::{disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_TMPL.format(
            n=n_per_combo, discipline=disc, amb_type=amb_type, amb_desc=amb_desc,
        )}],
        max_tokens=1500, temperature=0.9,
        metadata={"amb_type": amb_type, "amb_desc": amb_desc, "disc": disc},
    ) for amb_type, amb_desc in AMBIGUITY_TYPES for disc in DISCIPLINES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=5,
                    help="6 amb × 10 disc × 5 = 300 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v16/A1] generating ambiguous prompts")
    calls = gen_prompts(args.n_per_combo, args.workers)
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
                "prompt_id": f"v16::{md['amb_type']}::{md['disc']}::{i:02d}",
                "source": "synthetic_v16", "amb_type": md["amb_type"],
                "amb_desc": md["amb_desc"],
                "discipline": md["disc"], "lang": "en",
                "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; amb_types={dict(Counter(p['amb_type'] for p in prompts))}")

    print(f"[v16/A2] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        sys_msg = DEMO_SYS.format(amb_type=p["amb_type"])
        user_msg = f"AMBIGUOUS PROMPT: {p['full_prompt'][:2000]}\nDiscipline: {p['discipline']}\nAmbiguity type: {p['amb_type']} — {p['amb_desc']}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=2200, temperature=0.55,
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
            if r.error or not r.content or len(r.content) < 200: continue
            if '"propose"' not in r.content: continue
            # require explicit "assum" or "ambig" keyword in the rationale (case-insensitive)
            kw_check = any(k in r.content.lower() for k in ["assum", "ambig", "interpret", "vague", "clarif"])
            if not kw_check: continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "kind": "ambiguous", "amb_type": p["amb_type"],
                "discipline": p["discipline"], "lang": p["lang"],
                "archetype": "v16_ambiguous",
                "topic": f"[v16/ambig/{p['amb_type']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v16. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
