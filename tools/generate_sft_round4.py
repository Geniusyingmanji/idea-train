"""Round 4: target T2/T4 weakest tiers with COMPACT prompts (avoid round 3's timeouts).

Key change: cap card text at 120 chars (was 200) to keep prompts short.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.teachers.gpt55_client import TeacherCall, batch_call
from evo_opd.teachers.prompts import SYSTEM_PROMPT
from evo_opd.verifier import compute_verifier

ROUND1 = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/train.jsonl")
OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft/round4_train.jsonl")
MAX_FIELD = 120  # shorten to avoid prompt-length-induced timeouts


def parse_card(comp):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", comp, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(1))
    except: return None


def load_cards():
    out = []
    with ROUND1.open() as f:
        for line in f:
            r = json.loads(line)
            if r["task_type"] != "gene_card_extract": continue
            card = parse_card(r["completion"])
            if not card or not card.get("mechanism_genome") or not card.get("niche_genome"): continue
            out.append({"paper_id": r["metadata"].get("source_paper_id"), "card": card})
    return out


# T2-04 grouping_8 (8 papers into 2 lineages, compact)
def synth_t2_04(cards, rng):
    chosen = rng.sample(cards, 8)
    blocks = []
    for i, c in enumerate(chosen):
        cd = c["card"]
        blocks.append(f"[G{i+1}] Mech: {cd['mechanism_genome'][:MAX_FIELD]} | Niche: {cd['niche_genome'][:MAX_FIELD]}")
    prompt = (
        "8 genomes from TWO research lineages mixed. Identify which 4 go to lineage A vs B, in chronological order each.\n\n"
        + "\n".join(blocks)
        + '\n\nJSON: {"ordered_group_a": [4 positions], "ordered_group_b": [4 positions]}'
    )
    return chosen, prompt


# T4-02 intruder_domain — given 5 papers from same domain + 1 intruder, find intruder
def synth_t4_02(cards, rng):
    chosen = rng.sample(cards, 6)
    blocks = []
    for i, c in enumerate(chosen):
        cd = c["card"]
        blocks.append(f"[{chr(65+i)}] {cd['niche_genome'][:MAX_FIELD]}")
    prompt = (
        "Below are 6 paper niches. 5 belong to the same domain, 1 is an intruder from a different domain. Identify the intruder and list the lineage members in order.\n\n"
        + "\n".join(blocks)
        + '\n\nJSON: {"intruder": "A|B|C|D|E|F", "lineage_members": ["<labels of the 5 lineage members>"]}'
    )
    return chosen, prompt


# T2-12 gene_alignment — match 4 genes G1..G4 to 4 types A..D
def synth_t2_12(cards, rng):
    chosen = rng.sample(cards, 4)
    blocks = []
    for i, c in enumerate(chosen):
        cd = c["card"]
        blocks.append(f"[G{i+1}] {cd['mechanism_genome'][:MAX_FIELD]}")
    prompt = (
        "Below are 4 gene descriptions. Match each gene to one of types A (mechanism), B (niche), C (observation), D (limitation).\n\n"
        + "\n".join(blocks)
        + '\n\nJSON: {"assignments": {"G1": "A|B|C|D", "G2": "A|B|C|D", "G3": "A|B|C|D", "G4": "A|B|C|D"}}'
    )
    return chosen, prompt


# T3-13 hidden_gene_fate — two genes, classify each as INHERITED/MUTATED/LOST/NOVEL/HYBRIDIZED
def synth_t3_13(cards, rng):
    a, b = rng.sample(cards, 2)
    blocks = []
    for label, c in [("Parent", a), ("Child", b)]:
        cd = c["card"]
        blocks.append(f"[{label}] G1: {cd['mechanism_genome'][:MAX_FIELD]} | G2: {cd['niche_genome'][:MAX_FIELD]}")
    prompt = (
        "Given parent's two genes (G1=mechanism, G2=niche) and child's, classify each gene's fate AND the overall dynamics.\n\n"
        + "\n".join(blocks)
        + "\n\nFate options: INHERITED | MUTATED | LOST | NOVEL | HYBRIDIZED\n"
        + "Dynamics: Mutation | Adaptive Radiation | Hybridization | Speciation | Niche Competition\n"
        + '\nJSON: {"G1_status": "<fate>", "G2_status": "<fate>", "dynamics": "<dynamics>"}'
    )
    return [a, b], prompt


SYNTHS = {
    "T2-04_grouping_8": synth_t2_04,
    "T4-02_intruder_domain": synth_t4_02,
    "T2-12_gene_alignment": synth_t2_12,
    "T3-13_hidden_gene_fate": synth_t3_13,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-task", type=int, default=200)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--output", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cards = load_cards()
    print(f"  {len(cards)} cards")

    calls = []
    for tt, fn in SYNTHS.items():
        n = 0
        for _ in range(args.n_per_task):
            try:
                src, prompt = fn(cards, rng)
            except (ValueError, IndexError):
                continue
            calls.append(TeacherCall(
                prompt_id=f"r4::{tt}::{n:04d}",
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}],
                max_tokens=512,
                metadata={"task_type": tt, "prompt_for_train": prompt,
                          "source_paper_ids": [c["paper_id"] for c in src]},
            ))
            n += 1
        print(f"  {tt}: {n}")
    print(f"  total: {len(calls)}")

    print(f"running teacher (workers={args.workers})")
    log = OUT.parent.parent / "teacher_logs" / "gpt55_sft_round4.jsonl"
    def cb(d, t): print(f"  teacher progress: {d}/{t}", flush=True)
    t0 = time.time()
    results = batch_call(calls, workers=args.workers, log_path=log, on_progress=cb)
    print(f"  done in {(time.time()-t0)/60:.1f}min")

    out_path = Path(args.output); out_path.write_text("")
    stats = Counter()
    with out_path.open("a") as fp:
        for r in results:
            if r.error or not r.content:
                stats["api_error"] += 1; continue
            md = r.metadata or {}
            tt = md["task_type"]
            score, _ = compute_verifier(r.content, tt)
            if score.schema_valid < 1.0:
                stats[f"reject_schema_{tt}"] += 1; continue
            stats[f"accept_{tt}"] += 1
            fp.write(json.dumps({
                "instance_id": r.prompt_id, "task_type": tt,
                "prompt": md["prompt_for_train"], "completion": r.content,
                "metadata": {**md, "teacher_model": "gpt-5.5", "round": 4,
                              "teacher_input_tokens": r.input_tokens,
                              "teacher_output_tokens": r.output_tokens,
                              "verifier_score": score.to_dict()},
            }, ensure_ascii=False) + "\n")
    n_accept = sum(v for k, v in stats.items() if k.startswith("accept_"))
    print(f"acceptance: {n_accept}/{sum(stats.values())} = {100*n_accept/max(sum(stats.values()),1):.1f}%")
    print(f"detail: {stats}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
