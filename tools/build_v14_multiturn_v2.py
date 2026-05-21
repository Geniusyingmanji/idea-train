"""v14: improved multi-turn refinement.

v9's multiturn got only 44/200 valid (22%) because the format requirements
were strict. v14 uses a SIMPLER 2-action structure:

  [User]: <prompt>
  [Agent]: <rationale> + <propose v1 with gene_genome>
  [User feedback]: <specific critique using template>
  [Agent]: <acknowledgment + propose v2 with revised gene_genome>

Just 2 propose calls, no search tools in between. Easier to parse + validate.
Target: 250 valid pairs.

Output: data/agentic_v14/sft_demos.jsonl
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v14")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


FEEDBACK_TYPES = [
    ("vague_delta", "The delta_genome is too vague. Make it more specific with a quantified target."),
    ("weak_mechanism", "The mechanism_genome could be sharpened — what's the SPECIFIC mechanism being proposed?"),
    ("missing_metrics", "No measurable validation in claim_genome. Add a concrete experiment with target metric."),
    ("overlap_with_prior", "The mechanism overlaps too much with existing work. Pivot to a sharper distinction."),
    ("broad_niche", "The niche_genome is too broad. Narrow to a specific dataset/population/regime."),
    ("inconsistent", "The delta_genome and mechanism_genome are not consistent with each other. Make them aligned."),
    ("no_concrete_paper", "The proposal cites no concrete prior work. Mention specific methods/datasets being improved upon."),
    ("limitation_no_path", "The limitation_genome doesn't suggest how to address the limitation. Add a path forward."),
]


DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials", "physics",
    "neuroscience", "robotics_control", "energy", "clinical_medicine",
    "pharmacology", "economics_finance", "interdisciplinary",
]


PROMPT_GEN_TMPL = """Generate {n} research prompts in {discipline}. Each 1-3 sentences, suitable for the agent to propose a gene_genome idea. Output JSON array inside ```json ... ``` fences."""


DEMO_SYS = """You are demonstrating a two-turn refinement dialogue.

STRUCTURE:
  [Turn 1 - Agent's first attempt]
  Rationale (1-2 sentences) explaining the initial direction.
  ```action
  {{"tool":"propose","gene_genome":{{ 6 fields filled }}}}
  ```
  [Turn 2 - User feedback]
  USER: {feedback_msg}
  [Turn 3 - Agent's revised proposal]
  Rationale (1 sentence) acknowledging the critique.
  ```action
  {{"tool":"propose","gene_genome":{{ 6 fields, VISIBLY fixing the criticized aspect }}}}
  ```

Both `propose` actions must be valid gene_genome JSON. The second propose must visibly address the feedback in the {feedback_field} field (and stay coherent elsewhere).

Do NOT use any other tools (no search, no read). Just two proposes separated by user feedback."""


FEEDBACK_FIELD_MAP = {
    "vague_delta": "delta_genome",
    "weak_mechanism": "mechanism_genome",
    "missing_metrics": "claim_genome",
    "overlap_with_prior": "mechanism_genome/delta_genome",
    "broad_niche": "niche_genome",
    "inconsistent": "mechanism_genome+delta_genome consistency",
    "no_concrete_paper": "observation_genome",
    "limitation_no_path": "limitation_genome",
}


def gen_prompts(n_per_disc, workers):
    return [TeacherCall(
        prompt_id=f"v14::{disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_TMPL.format(n=n_per_disc, discipline=disc)}],
        max_tokens=2500, temperature=0.85,
        metadata={"disc": disc},
    ) for disc in DISCIPLINES]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-disc", type=int, default=25,
                    help="12 disc × 25 = 300 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v14/A1] generating prompts")
    calls = gen_prompts(args.n_per_disc, args.workers)
    t0 = time.time()
    results = batch_call(calls, workers=args.workers)
    print(f"  done in {time.time()-t0:.1f}s")

    prompts = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2032)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        for i, q in enumerate(arr[:args.n_per_disc]):
            if not isinstance(q, str) or len(q) < 30: continue
            fb_key, fb_msg = rng.choice(FEEDBACK_TYPES)
            prompts.append({
                "prompt_id": f"v14::{r.metadata['disc']}::{i:02d}::{fb_key}",
                "source": "synthetic_v14", "discipline": r.metadata["disc"],
                "lang": "en", "feedback_type": fb_key,
                "feedback_msg": fb_msg,
                "feedback_field": FEEDBACK_FIELD_MAP[fb_key],
                "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; feedback={dict(Counter(p['feedback_type'] for p in prompts))}")

    print(f"[v14/A2] generating two-turn demos (workers={args.workers})")
    calls = []
    for p in prompts:
        sys_msg = DEMO_SYS.format(
            feedback_msg=p["feedback_msg"],
            feedback_field=p["feedback_field"],
        )
        user_msg = f"PROMPT: {p['full_prompt'][:2000]}\nDiscipline: {p['discipline']}\nFeedback critique to use in Turn 2: \"{p['feedback_msg']}\""
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=2500, temperature=0.5,
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
            # MUST have 2 propose calls
            if r.content.count('"propose"') < 2: continue
            # MUST mention USER feedback marker
            if "USER:" not in r.content and "[Turn 2" not in r.content: continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "kind": "multiturn_v2",
                "discipline": p["discipline"], "lang": p["lang"],
                "feedback_type": p["feedback_type"],
                "feedback_msg": p["feedback_msg"],
                "archetype": "v14_multiturn_v2",
                "topic": f"[v14/multiturn/{p['feedback_type']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v14. valid={n_valid}/{len(results)} ({n_valid/max(len(results),1)*100:.1f}%)")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
