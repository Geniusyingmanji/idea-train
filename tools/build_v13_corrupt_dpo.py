"""v13: guaranteed-pair DPO via existing-demo corruption.

Past DPO rounds (v7, v10) had ~35-40% pair yield because chosen+rejected calls
both had to succeed independently. This round inverts the strategy:

  1. Sample 400 existing high-quality SFT demos as "chosen".
  2. For each chosen, ask GPT-5.5 to CORRUPT it in a specific way (premature
     propose, wrong schema, no evidence, schema collapse).
  3. Use original = chosen, corrupted = rejected. 100% pair yield.

Output: data/agentic_v13/preferences.jsonl  (~400 pairs)
"""
from __future__ import annotations
import argparse, json, random, sys, time
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call

OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v13")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_OUT = OUT_DIR / "preferences.jsonl"

CORPUS = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_combined_v3to18/sft_demos.jsonl")


CORRUPTIONS = {
    "premature_propose": """Rewrite this trajectory by REMOVING all tool calls except the final `propose`. The propose's gene_genome/idea_plan must still be present but should now look generic and ungrounded since no search was done. Output ONLY the rewritten trajectory.""",
    "wrong_schema": """If this trajectory's final `propose` uses gene_genome, rewrite it to use idea_plan schema instead (with the 6 idea_plan fields). If it uses idea_plan, rewrite to use gene_genome. Keep the rest of the trajectory the same. Output ONLY the rewritten trajectory.""",
    "no_evidence": """Keep the search/read tool calls and their [results], but rewrite the final `propose` so that the gene_genome/idea_plan fields are COMPLETELY UNRELATED to what was retrieved. Make up generic content that ignores the actual evidence. Output ONLY the rewritten trajectory.""",
    "schema_collapse": """Take the final `propose` action and CORRUPT it: merge all 6 fields into one big text blob, OR drop 2-3 fields, OR replace the JSON with prose. Keep the rest of the trajectory intact. Output ONLY the rewritten trajectory.""",
    "truncated": """Cut the trajectory off MIDWAY through, before the `propose` action. End at a tool call result. The trajectory should look like the agent gave up. Output ONLY the rewritten trajectory.""",
    "verbose_padding": """Pad the trajectory with rambly, repetitive thinking — duplicate or near-duplicate search queries, redundant rationale, extra paragraphs that say nothing new. The final propose is unchanged. Output ONLY the rewritten trajectory.""",
}


def load_corpus_sample(n_per_corruption: int) -> list[dict]:
    """Sample n_per_corruption demos per corruption type, stratified by source round."""
    rng = random.Random(2031)
    demos = []
    with CORPUS.open() as f:
        for line in f:
            try: demos.append(json.loads(line))
            except: pass
    # Only sample demos with valid propose + at least 1 action
    eligible = [d for d in demos if d["completion"].count("```action") >= 1 and '"propose"' in d["completion"]]
    print(f"  corpus: {len(eligible)} eligible demos out of {len(demos)} total")
    # Stratified sample: 70% from short (1-3 tools), 30% from longer
    short_pool = [d for d in eligible if d["completion"].count("```action") <= 3]
    long_pool = [d for d in eligible if d["completion"].count("```action") >= 4]
    n_each = n_per_corruption
    n_total = n_each * len(CORRUPTIONS)
    sample = []
    rng.shuffle(short_pool); rng.shuffle(long_pool)
    n_short = int(n_total * 0.7)
    n_long = n_total - n_short
    sample = short_pool[:n_short] + long_pool[:n_long]
    rng.shuffle(sample)
    print(f"  sample: {len(sample)} = {n_short} short + {n_long} long")
    # assign corruptions in order
    corruption_keys = list(CORRUPTIONS.keys())
    out = []
    for i, d in enumerate(sample):
        corruption = corruption_keys[i % len(corruption_keys)]
        out.append({
            "prompt_id": f"v13::{d['prompt_id']}::{corruption}",
            "source_prompt_id": d["prompt_id"],
            "discipline": d.get("discipline", "unknown"),
            "lang": d.get("lang", "en"),
            "corruption": corruption,
            "full_prompt": d["full_prompt"],
            "candidates": d.get("candidates", []),
            "original_completion": d["completion"],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-corruption", type=int, default=70,
                    help="6 corruptions × 70 = 420 pairs")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    print("[v13/A1] loading corpus + sampling chosen demos")
    samples = load_corpus_sample(args.n_per_corruption)
    print(f"  {len(samples)} samples; corruptions={dict((c, sum(1 for s in samples if s['corruption']==c)) for c in CORRUPTIONS)}")

    done = set()
    if args.resume and PAIRS_OUT.exists():
        with PAIRS_OUT.open() as f:
            for line in f:
                try: done.add(json.loads(line)["prompt_id"])
                except: pass
        samples = [s for s in samples if s["prompt_id"] not in done]
        print(f"  resume: {len(samples)} remaining")

    print(f"[v13/A2] generating corrupted versions (workers={args.workers})")
    calls = []
    for s in samples:
        instr = CORRUPTIONS[s["corruption"]]
        sys_msg = f"You are corrupting an agentic research trajectory for a preference-pair training dataset. The CORRUPTION TYPE is: {s['corruption']}.\n\nINSTRUCTION: {instr}"
        user_msg = f"ORIGINAL TRAJECTORY:\n\n{s['original_completion']}\n\n----\n\nApply the {s['corruption']} corruption and output the rewritten trajectory only."
        calls.append(TeacherCall(
            prompt_id=s["prompt_id"],
            messages=[{"role": "system", "content": sys_msg},
                      {"role": "user", "content": user_msg}],
            max_tokens=3500, temperature=0.6,
            metadata={"sample": s},
        ))

    raw_log = PAIRS_OUT.with_suffix(".raw_calls.jsonl")
    t0 = time.time()
    results = batch_call(
        calls, workers=args.workers, log_path=raw_log,
        on_progress=lambda d, t: print(f"  [{d}/{t}] {time.time()-t0:.0f}s", flush=True),
    )

    n_valid = 0
    with PAIRS_OUT.open("a") as f:
        for r in results:
            s = r.metadata["sample"]
            if r.error or not r.content or len(r.content) < 100: continue
            # No structural validation — the whole point is corruption can produce malformed output
            f.write(json.dumps({
                "prompt_id": s["prompt_id"],
                "source_prompt_id": s["source_prompt_id"],
                "discipline": s["discipline"], "lang": s["lang"],
                "corruption": s["corruption"],
                "full_prompt": s["full_prompt"],
                "candidates": s["candidates"],
                "chosen": s["original_completion"],
                "rejected": r.content,
                "chosen_actions": s["original_completion"].count("```action"),
                "rejected_actions": r.content.count("```action"),
            }, ensure_ascii=False) + "\n")
            n_valid += 1
    print(f"\nDone v13. pairs={n_valid}/{len(samples)} ({n_valid/max(len(samples),1)*100:.1f}%)")
    print(f"saved → {PAIRS_OUT}")


if __name__ == "__main__":
    main()
