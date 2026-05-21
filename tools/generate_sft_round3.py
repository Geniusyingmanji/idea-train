"""Round 3 SFT generation: T2 + a few T3/T4 with STRICT schema field names."""
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
OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft")


def parse_card(comp):
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", comp, re.DOTALL)
    if not m: return None
    try: return json.loads(m.group(1))
    except: return None


def load_round1_cards():
    out = []
    with ROUND1.open() as f:
        for line in f:
            r = json.loads(line)
            if r["task_type"] != "gene_card_extract":
                continue
            card = parse_card(r["completion"])
            if not card or not card.get("mechanism_genome") or not card.get("niche_genome"):
                continue
            out.append({"paper_id": r["metadata"].get("source_paper_id"),
                         "year": r["metadata"].get("year"),
                         "card": card})
    return out


# T2-01 ordering 5
def synth_t2_01(cards, rng):
    chosen = rng.sample(cards, 5)
    # use index-of-paper as a fake "real" ordering, then shuffle indices
    real_order = list(range(5))  # papers in chosen are in "chronological" order arbitrarily
    rng.shuffle(real_order)
    # build prompt with chosen labelled by shuffled positions
    blocks = []
    for i, c in enumerate(chosen):
        # the displayed Genome i shows card with shuffled order
        cd = c["card"]
        blocks.append(
            f"[Genome {i+1}]\n"
            f"  Mechanism: {cd['mechanism_genome'][:200]}\n"
            f"  Niche: {cd['niche_genome'][:200]}\n"
            f"  Observation: {cd.get('observation_genome','')[:200]}\n"
            f"  Limitation: {cd.get('limitation_genome','')[:200]}\n"
        )
    prompt = (
        "Below are 5 genome descriptions from a single research lineage, in RANDOM order. "
        "Reconstruct the chronological order (oldest predecessor first).\n\n"
        + "\n".join(blocks)
        + '\n\nOutput JSON with EXACTLY this key:\n{"correct_order": [<five 1-indexed positions in chronological order>]}'
    )
    return chosen, prompt


# T2-04 grouping 8 into 2 lineages
def synth_t2_04(cards, rng):
    chosen = rng.sample(cards, 8)
    blocks = []
    for i, c in enumerate(chosen):
        cd = c["card"]
        blocks.append(
            f"[Genome {i+1}]\n"
            f"  Mechanism: {cd['mechanism_genome'][:200]}\n"
            f"  Niche: {cd['niche_genome'][:200]}\n"
        )
    prompt = (
        "Below are 8 genome descriptions from TWO distinct research lineages mixed together. "
        "Identify which 4 belong to lineage A and which 4 to lineage B, and present each group in chronological order.\n\n"
        + "\n".join(blocks)
        + '\n\nOutput JSON with EXACTLY these keys:\n'
        + '{"ordered_group_a": [<four 1-indexed positions>], "ordered_group_b": [<four 1-indexed positions>]}'
    )
    return chosen, prompt


# T2-07 lim/delta match — for each Limitation paper, find which Delta paper repairs it
def synth_t2_07(cards, rng):
    n = 3
    chosen = rng.sample(cards, n * 2)
    # pretend half are L (with limitation), half are D (with delta)
    limitations = chosen[:n]
    deltas = chosen[n:]
    blocks = []
    for i, c in enumerate(limitations):
        blocks.append(f"[L{i+1}] Limitation: {c['card'].get('limitation_genome', c['card']['mechanism_genome'])[:200]}")
    for i, c in enumerate(deltas):
        blocks.append(f"[D{i+1}] Delta: {c['card'].get('delta_genome', c['card']['mechanism_genome'])[:200]}")
    prompt = (
        "Below are 3 papers' Limitations (L1..L3) and 3 papers' Deltas (D1..D3). "
        "For each limitation, identify which delta most plausibly repairs it.\n\n"
        + "\n".join(blocks)
        + '\n\nOutput JSON with EXACTLY this key:\n'
        + '{"mapping": {"L1": "D?", "L2": "D?", "L3": "D?"}}'
    )
    return chosen, prompt


# T3-09 relation classify
def synth_t3_09(cards, rng):
    a, b = rng.sample(cards, 2)
    blocks = []
    for label, c in [("A", a), ("B", b)]:
        cd = c["card"]
        blocks.append(
            f"[Genome {label}]\n"
            f"  Mechanism: {cd['mechanism_genome'][:200]}\n"
            f"  Niche: {cd['niche_genome'][:200]}\n"
        )
    prompt = (
        "Classify the relation between paper A and paper B.\n\n"
        + "\n".join(blocks)
        + "\n\nRelation options: lineage_inheritance | foundation | competitor | isolation\n"
        + "Dynamics options: Mutation | Adaptive Radiation | Hybridization | Speciation | Niche Competition\n"
        + '\nOutput JSON with EXACTLY these keys:\n'
        + '{"label": "lineage_inheritance|foundation|competitor|isolation", '
          '"dynamics": "Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition"}'
    )
    return [a, b], prompt


# T4-03 wrong-step: given a 5-paper lineage, one adjacent pair is impossible; spot it
def synth_t4_03(cards, rng):
    chosen = rng.sample(cards, 5)
    blocks = []
    for i, c in enumerate(chosen):
        cd = c["card"]
        blocks.append(
            f"[Step {chr(65+i)}]\n"
            f"  Mechanism: {cd['mechanism_genome'][:200]}\n"
            f"  Niche: {cd['niche_genome'][:200]}\n"
        )
    prompt = (
        "Below is a proposed 5-step research lineage. ONE adjacent step (or no step) is wrong. "
        "Identify the wrong step (if any) and propose the corrected ordering + dynamics for the fixed step.\n\n"
        + "\n".join(blocks)
        + "\n\nDynamics options: Mutation | Adaptive Radiation | Hybridization | Speciation | Niche Competition\n"
        + '\nOutput JSON with EXACTLY these keys:\n'
        + '{"label": "A|B|C|D|E|none", '
          '"correct_order": ["A","B","C","D","E"], '
          '"correct_dynamics": "Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition"}'
    )
    return chosen, prompt


SYNTHS = {
    "T2-01_ordering_5": synth_t2_01,
    "T2-04_grouping_8": synth_t2_04,
    "T2-07_lim_delta_match": synth_t2_07,
    "T3-09_relation_classify": synth_t3_09,
    "T4-03_wrong_step": synth_t4_03,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-task", type=int, default=200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--output", default=str(OUT_DIR / "round3_train.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"[1/3] Loading round 1 cards")
    cards = load_round1_cards()
    print(f"  {len(cards)} cards")

    print(f"[2/3] Building calls")
    calls = []
    for tt, fn in SYNTHS.items():
        n = 0
        attempts = 0
        while n < args.n_per_task and attempts < args.n_per_task * 2:
            attempts += 1
            try:
                src, prompt = fn(cards, rng)
            except (ValueError, IndexError):
                continue
            calls.append(TeacherCall(
                prompt_id=f"r3::{tt}::{n:04d}",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                metadata={"task_type": tt, "prompt_for_train": prompt,
                          "source_paper_ids": [c["paper_id"] for c in src]},
            ))
            n += 1
        print(f"  {tt}: {n}")
    print(f"  total: {len(calls)}")

    print(f"[3/3] Running teacher (workers={args.workers})")
    log = OUT_DIR.parent / "teacher_logs" / "gpt55_sft_round3.jsonl"
    def cb(d, t): print(f"  teacher progress: {d}/{t}", flush=True)
    t0 = time.time()
    results = batch_call(calls, workers=args.workers, log_path=log, on_progress=cb)
    el = time.time() - t0
    n_ok = sum(1 for r in results if r.content)
    print(f"  done in {el/60:.1f}min: {n_ok}/{len(results)} OK")

    print(f"[4/4] Verifying + writing")
    out_path = Path(args.output); out_path.write_text("")
    stats = Counter()
    out_fp = out_path.open("a")
    for r in results:
        if r.error or not r.content:
            stats["api_error"] += 1
            continue
        md = r.metadata or {}
        tt = md["task_type"]
        score, _ = compute_verifier(r.content, tt)
        if score.schema_valid < 1.0:
            stats[f"reject_schema_{tt}"] += 1
            continue
        stats[f"accept_{tt}"] += 1
        out_fp.write(json.dumps({
            "instance_id": r.prompt_id,
            "task_type": tt,
            "prompt": md["prompt_for_train"],
            "completion": r.content,
            "metadata": {
                "source_paper_ids": md["source_paper_ids"],
                "teacher_model": "gpt-5.5", "round": 3,
                "teacher_input_tokens": r.input_tokens,
                "teacher_output_tokens": r.output_tokens,
                "verifier_score": score.to_dict(),
            },
        }, ensure_ascii=False) + "\n")
    out_fp.close()
    n_accept = sum(v for k, v in stats.items() if k.startswith("accept_"))
    print(f"  acceptance: {n_accept}/{sum(stats.values())} = {100*n_accept/max(sum(stats.values()),1):.1f}%")
    print(f"  detail: {stats}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
