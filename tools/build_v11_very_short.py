"""v11: very_short / direct-propose demos.

Many research prompts are CLEAR enough that the agent should just propose
without any search. Current v3-v10 still over-tools on these. v11 = 500 demos
where the prompt explicitly provides enough context (citation, constraints,
prior method) so that the model should propose IMMEDIATELY with 0 or 1 tool calls.

Output: data/agentic_v11/sft_demos.jsonl  (~500 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v11")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT_DIR / "prompts.jsonl"
DEMOS_OUT = OUT_DIR / "sft_demos.jsonl"


# ─── Prompt template asks for context-rich prompts ─────────────────────────

PROMPT_GEN_TMPL = """Generate {n} research prompts in {discipline} that already CONTAIN enough context for an agent to propose without further search. Each prompt should:
- Cite 1-2 prior methods or approaches by name
- State a concrete limitation or gap
- Specify a target (better metric / new domain / cheaper compute)
- Be 2-4 sentences

These are prompts where the agent should just propose. Output JSON array inside ```json ... ``` fences."""


DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials", "physics",
    "neuroscience", "energy", "robotics_control", "economics_finance",
    "clinical_medicine", "pharmacology", "agriculture_food", "cognitive_science",
    "interdisciplinary",
]


# ─── Demo: encourage 0-1 tool call patterns ─────────────────────────────────

DIRECT_SYS = """You are demonstrating a VERY-SHORT agentic trajectory.

LENGTH: 1-2 actions total. The prompt provides enough context — DO NOT over-search.

Pattern options (pick one):
  (a) DIRECT PROPOSE: just 1 propose action. The prompt already cites prior work and gap.
  (b) SINGLE-SEARCH: 1 search to confirm/refine a detail + propose. 2 actions.

NEVER use read, extract_genome, genome_diff, novelty_check for this trajectory. Stay decisive.

Format: 1-2 sentence rationale ("the prompt already cites X and asks for Y; I can propose directly") + ```action ... ``` JSON tool call + [result] if search. End with propose using gene_genome (6 fields)."""


CONFIRM_SYS = """You are demonstrating a VERY-SHORT agentic trajectory with SINGLE confirming search.

LENGTH: 2 actions total — exactly 1 search + 1 propose.

The search is a single targeted check (e.g., "is method X still SOTA in 2025?") not exploratory. The result confirms or slightly adjusts the agent's preformed idea, and the agent proposes.

Format: rationale + ```action search ... ``` + [result] + rationale + ```action propose ... ```.
Final propose: {"tool":"propose","gene_genome":{6 fields}}.

NEVER more than 2 actions."""


def gen_prompts(n_per_disc, workers):
    # Use 4000 max_tokens so 40 multi-sentence prompts fit
    # (GPT-5.5 reasoning eats much of the budget)
    return [TeacherCall(
        prompt_id=f"v11::{disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_TMPL.format(n=n_per_disc, discipline=disc)}],
        max_tokens=4000, temperature=0.9,
        metadata={"discipline": disc},
    ) for disc in DISCIPLINES]


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=3, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-disc", type=int, default=40,
                    help="14 disc × 40 = 560 prompts")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v11/A1] generating context-rich prompts")
    calls = gen_prompts(args.n_per_disc, args.workers)
    print(f"  dispatching {len(calls)}")
    t0 = time.time()
    results = batch_call(calls, workers=args.workers)
    print(f"  done in {time.time()-t0:.1f}s")

    prompts = []
    fence_re = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
    rng = random.Random(2029)
    for r in results:
        if r.error or not r.content: continue
        m = fence_re.search(r.content)
        try: arr = json.loads(m.group(1) if m else r.content)
        except: continue
        if not isinstance(arr, list): continue
        for i, q in enumerate(arr[:args.n_per_disc]):
            if not isinstance(q, str) or len(q) < 40: continue
            # 60% direct propose, 40% single-search confirm
            mode = "direct" if rng.random() < 0.6 else "confirm"
            prompts.append({
                "prompt_id": f"v11::{r.metadata['discipline']}::{i:02d}::{mode}",
                "source": "synthetic_v11", "discipline": r.metadata["discipline"],
                "mode": mode, "lang": "en", "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"  {len(prompts)} prompts; modes={dict(Counter(p['mode'] for p in prompts))}")

    done = set()
    if args.resume and DEMOS_OUT.exists():
        with DEMOS_OUT.open() as f:
            for line in f:
                try: done.add(json.loads(line)["prompt_id"])
                except: pass
        prompts = [p for p in prompts if p["prompt_id"] not in done]

    # Only prefetch for confirm mode (direct mode doesn't search)
    print("[v11/A2] light prefetch for confirm-mode prompts")
    st = WebSearchTool()
    prefetched = {}
    t0 = time.time()
    for i, p in enumerate(prompts):
        if p["mode"] == "confirm":
            prefetched[p["prompt_id"]] = prefetch(p, st)
        if (i+1) % 100 == 0: print(f"  {i+1}/{len(prompts)} ({time.time()-t0:.0f}s)")
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v11/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        sys_msg = DIRECT_SYS if p["mode"] == "direct" else CONFIRM_SYS
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\nCandidates (use one if needed):\n"
            for i, c in enumerate(cands[:3]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"PROMPT: {p['full_prompt'][:2500]}\nDiscipline: {p['discipline']}\nMode: {p['mode']}{cand_blob}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=1800, temperature=0.45,
            metadata={"prompt": p, "candidates": cands},
        ))

    raw_log = DEMOS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    n_valid = 0
    n_overshoot = 0
    with DEMOS_OUT.open("a") as f:
        for r in results:
            p = r.metadata["prompt"]
            if r.error or not r.content or len(r.content) < 150: continue
            if '"propose"' not in r.content: continue
            n_actions = r.content.count("```action")
            if n_actions > 3:  # enforce the short discipline
                n_overshoot += 1; continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "mode": p["mode"], "discipline": p["discipline"],
                "lang": p["lang"], "archetype": f"v11_{p['mode']}",
                "length_tier": "very_short",
                "topic": f"[v11/{p['mode']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v11. valid={n_valid}/{len(results)} (overshoot dropped={n_overshoot})")
    print(f"saved → {DEMOS_OUT}")


if __name__ == "__main__":
    main()
