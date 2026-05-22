"""v23: mathematics and formal-sciences focus.

Combined corpus is sparse on mathematics, theoretical CS, formal logic.
Adds ~200 demos in these underrepresented domains.

Output: data/agentic_v23/sft_demos.jsonl  (~200 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v23")
OUT.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT / "prompts.jsonl"
DEMOS_OUT = OUT / "sft_demos.jsonl"


DOMAINS = [
    ("pure_mathematics", "pure mathematics — algebra, topology, analysis, number theory"),
    ("applied_mathematics", "applied math — PDEs, optimization, numerical methods"),
    ("theoretical_cs", "theoretical computer science — algorithms, complexity, formal methods"),
    ("statistics", "statistics — inference, causal methods, experimental design"),
    ("formal_logic", "logic and verification — type theory, theorem proving, model checking"),
    ("information_theory", "information theory and coding"),
    ("optimization", "convex/nonconvex optimization, OR"),
    ("control_theory", "control theory and dynamical systems"),
]


PROMPT_GEN_TMPL = """Generate {n} research prompts in {description}. Each 1-3 sentences, technical and substantive. Output JSON array inside ```json ... ``` fences."""


DEMO_SYS = """You are demonstrating a short agentic research trajectory in a mathematical/formal-sciences domain (2-4 actions).

The trajectory should show: rationale + search/read + propose. The final propose should use precise technical language appropriate for the formal domain.

Format: rationale + ```action ... ``` JSON + [result] simulated. End with propose:
{"tool":"propose","gene_genome":{"mechanism_genome":"...","niche_genome":"...","observation_genome":"...","limitation_genome":"...","delta_genome":"...","claim_genome":"..."}}

Tools: search, read, propose. Use real OpenAlex paper_ids if candidates provided."""


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=3, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-disc", type=int, default=25,
                    help="8 domains × 25 = 200 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v23/A1] generating math/formal prompts")
    calls = [TeacherCall(
        prompt_id=f"v23::{disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_TMPL.format(n=args.n_per_disc, description=desc)}],
        max_tokens=1800, temperature=0.85,
        metadata={"disc": disc, "desc": desc},
    ) for disc, desc in DOMAINS]
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
        for i, q in enumerate(arr[:args.n_per_disc]):
            if not isinstance(q, str) or len(q) < 25: continue
            prompts.append({
                "prompt_id": f"v23::{md['disc']}::{i:02d}",
                "source": "synthetic_v23", "lang": "en",
                "discipline": md["disc"], "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  {len(prompts)} prompts")

    print("[v23/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {p["prompt_id"]: prefetch(p, st) for p in prompts}
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v23/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\nCandidates:\n"
            for i, c in enumerate(cands[:3]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"PROMPT: {p['full_prompt'][:2000]}\nDomain: {p['discipline']}{cand_blob}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": DEMO_SYS},
                      {"role": "user", "content": user_msg}],
            max_tokens=2400, temperature=0.5,
            metadata={"prompt": p, "candidates": cands},
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
            if r.content.count("```action") < 1 or '"propose"' not in r.content: continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "lang": "en", "discipline": p["discipline"],
                "archetype": "v23_math_formal",
                "topic": f"[v23/math/{p['discipline']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v23. valid={n_valid}/{len(results)}")


if __name__ == "__main__":
    main()
