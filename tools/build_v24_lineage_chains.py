"""v24: full lineage-tool chain demos.

Most demos use search + read + propose. v24 exercises the full pipeline:
  search → read parent paper → extract_genome → propose child → genome_diff
  vs parent → optionally novelty_check → final propose.

Targets prompts that explicitly mention "extend / build on / improve" a
prior method, where the lineage tools are most natural.

Output: data/agentic_v24/sft_demos.jsonl (~150 demos)
"""
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.tools.web_search import WebSearchTool

OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v24")
OUT.mkdir(parents=True, exist_ok=True)
PROMPTS_OUT = OUT / "prompts.jsonl"
DEMOS_OUT = OUT / "sft_demos.jsonl"


DISCIPLINES = [
    "computer_science", "biology", "chemistry", "materials",
    "neuroscience", "robotics_control", "energy", "interdisciplinary",
]


PROMPT_GEN_TMPL = """Generate {n} research prompts in {discipline} that explicitly invite "extending" or "building on" a prior recent method. Phrase each prompt to make the lineage relationship clear (e.g., "How would you extend the X approach to address Y?"). 1-3 sentences each. Output JSON array inside ```json ... ``` fences."""


DEMO_SYS = """You are demonstrating a FULL lineage-tool chain agentic trajectory (5-7 actions):

  1. search — find a recent parent paper relevant to the prompt
  2. read — fetch the parent paper's details
  3. extract_genome — pull the parent's 6-field genome (mechanism/niche/...)
  4. propose — articulate the child proposal extending the parent
  5. genome_diff — compute the delta between parent_genome and proposed_genome
  6. novelty_check (optional) — verify the delta is new
  7. final propose — refined gene_genome (optional re-propose)

Format each step: rationale (1-2 sentences) + ```action ... ``` JSON tool call + [result] (1-3 sentences). End with propose.

Tools schemas:
  search: {{"tool":"search","query":"...","year_min":...,"year_max":...,"k":5}}
  read: {{"tool":"read","paper_id":"oa:Wxxxxx"}}
  extract_genome: {{"tool":"extract_genome","paper_id":"oa:Wxxxxx"}}
  genome_diff: {{"tool":"genome_diff","parent_id":"oa:Wxxxxx","proposed_genome":{{6 fields}}}}
  novelty_check: {{"tool":"novelty_check","mechanism":"...","year_min":...,"year_max":...}}
  propose: {{"tool":"propose","gene_genome":{{6 fields filled}}}}

The trajectory should naturally cite the parent paper's paper_id throughout."""


def prefetch(p, st):
    try:
        rs = st.search(p["full_prompt"][:200], k=5, year_min=2018, year_max=2025)
        return [r.to_dict() for r in rs]
    except: return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-disc", type=int, default=20,
                    help="8 disc × 20 = 160 prompts")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()

    print("[v24/A1] generating lineage-oriented prompts")
    calls = [TeacherCall(
        prompt_id=f"v24::{disc}",
        messages=[{"role": "user", "content": PROMPT_GEN_TMPL.format(n=args.n_per_disc, discipline=disc)}],
        max_tokens=1800, temperature=0.85,
        metadata={"disc": disc},
    ) for disc in DISCIPLINES]
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
        for i, q in enumerate(arr[:args.n_per_disc]):
            if not isinstance(q, str) or len(q) < 30: continue
            prompts.append({
                "prompt_id": f"v24::{r.metadata['disc']}::{i:02d}",
                "source": "synthetic_v24", "lang": "en",
                "discipline": r.metadata["disc"],
                "full_prompt": q.strip(),
            })

    with PROMPTS_OUT.open("w") as f:
        for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  {len(prompts)} prompts")

    print("[v24/A2] prefetching candidates")
    st = WebSearchTool()
    t0 = time.time()
    prefetched = {p["prompt_id"]: prefetch(p, st) for p in prompts}
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[v24/A3] generating demos (workers={args.workers})")
    calls = []
    for p in prompts:
        cands = prefetched.get(p["prompt_id"], [])
        cand_blob = ""
        if cands:
            cand_blob = "\n\nReal OpenAlex candidates (use ONE as parent):\n"
            for i, c in enumerate(cands[:5]):
                cand_blob += f"  [{i+1}] {c['paper_id']} ({c.get('year','?')}): {c.get('title','')[:120]}\n"
        user_msg = f"PROMPT: {p['full_prompt'][:2000]}\nDiscipline: {p['discipline']}{cand_blob}"
        calls.append(TeacherCall(
            prompt_id=p["prompt_id"],
            messages=[{"role": "system", "content": DEMO_SYS},
                      {"role": "user", "content": user_msg}],
            max_tokens=3500, temperature=0.5,
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
            if r.error or not r.content or len(r.content) < 300: continue
            if r.content.count("```action") < 4 or '"propose"' not in r.content: continue
            # require extract_genome AND genome_diff in the chain
            if 'extract_genome' not in r.content or 'genome_diff' not in r.content: continue
            f.write(json.dumps({
                "prompt_id": p["prompt_id"], "source": p["source"],
                "kind": "lineage_chain", "discipline": p["discipline"], "lang": "en",
                "archetype": "v24_lineage_chains",
                "topic": f"[v24/lineage/{p['discipline']}] {p['full_prompt'][:160]}",
                "full_prompt": p["full_prompt"],
                "completion": r.content,
                "candidates": r.metadata.get("candidates", []),
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v24. valid={n_valid}/{len(results)}")


if __name__ == "__main__":
    main()
