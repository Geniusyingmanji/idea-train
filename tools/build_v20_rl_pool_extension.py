"""v20: RL prompt pool extension.

v10 produced 1190 RL prompts. v20 adds ~800 more with focus on:
  - More disciplines (extend beyond v10's 20)
  - More prompt styles (peer-review request, conference CFP, learner Q)
  - Held-out from any SFT discipline×style combo already used
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call

OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v20")
OUT.mkdir(parents=True, exist_ok=True)
RL_OUT = OUT / "rl_prompts.jsonl"


DISC_STYLES = [
    # discipline, prompt_style
    ("computer_science", "peer_review_request"),
    ("computer_science", "conference_cfp"),
    ("biology", "industry_practitioner"),
    ("biology", "learner_question"),
    ("chemistry", "policy_question"),
    ("materials", "industry_practitioner"),
    ("physics", "academic_question"),
    ("neuroscience", "peer_review_request"),
    ("clinical_medicine", "policy_question"),
    ("pharmacology", "industry_practitioner"),
    ("robotics_control", "conference_cfp"),
    ("economics_finance", "policy_question"),
    ("sociology", "learner_question"),
    ("agriculture_food", "policy_question"),
    ("urban_planning", "policy_question"),
    ("cognitive_science", "academic_question"),
    ("energy", "industry_practitioner"),
    ("interdisciplinary", "academic_question"),
    ("interdisciplinary", "learner_question"),
    ("interdisciplinary", "peer_review_request"),
]

STYLE_HINTS = {
    "academic_question": "framed as an open research question from an academic",
    "peer_review_request": "framed as a peer-review critique request for a 2-3 sentence proposal",
    "conference_cfp": "framed as a CFP excerpt soliciting submissions",
    "learner_question": "framed as a curious newcomer's question",
    "industry_practitioner": "framed as an industry practitioner solving a real problem",
    "policy_question": "framed as a policy-maker asking for evidence-based research",
}


PROMPT_GEN = """Generate {n} diverse research prompts in the area of {discipline}, {style_hint}. Each 1-3 sentences. Output JSON array inside ```json ... ``` fences."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-combo", type=int, default=40,
                    help="20 combos × 40 = 800 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v20/A1] generating prompts")
    calls = [TeacherCall(
        prompt_id=f"v20::{disc}::{style}",
        messages=[{"role": "user", "content": PROMPT_GEN.format(
            n=args.n_per_combo, discipline=disc, style_hint=STYLE_HINTS[style],
        )}],
        max_tokens=2500, temperature=0.9,
        metadata={"disc": disc, "style": style},
    ) for disc, style in DISC_STYLES]
    print(f"  dispatching {len(calls)}")
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
                "prompt_id": f"v20::{md['disc']}::{md['style']}::{i:02d}",
                "source": "v20_rl_pool", "lang": "en",
                "discipline": md["disc"], "style": md["style"],
                "full_prompt": q.strip(),
            })

    with RL_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; styles={dict(Counter(p['style'] for p in prompts))}")
    print(f"saved → {RL_OUT}")


if __name__ == "__main__":
    main()
