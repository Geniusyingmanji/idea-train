"""Round 2 SFT data generation: T1/T3/T4 exam-style examples.

Re-uses the 865 GPT-5.5-generated gene cards from round 1 as substrate, then
synthesizes T1-01, T1-03, T3-01, T4-01 style training instances.
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
OUT_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/stage1_sft")


def parse_gene_card_from_completion(comp: str) -> dict | None:
    """Extract gene-card JSON from a round-1 completion."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", comp, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def load_round1_cards() -> list[dict]:
    """Returns list of {paper_id, title, card_json (with 6 fields)}."""
    out = []
    with ROUND1.open() as f:
        for line in f:
            r = json.loads(line)
            if r["task_type"] != "gene_card_extract":
                continue
            card = parse_gene_card_from_completion(r["completion"])
            if not card:
                continue
            # need at least mechanism + niche populated to be useful
            if not card.get("mechanism_genome") or not card.get("niche_genome"):
                continue
            out.append({
                "paper_id": r["metadata"].get("source_paper_id"),
                "year": r["metadata"].get("year"),
                "domain": r["metadata"].get("domain"),
                "card": card,
            })
    return out


# --------------------------------------------------------------------------
# Synthesizers for each task type
# --------------------------------------------------------------------------

# T1-01: classify contribution_type for each of 4 papers
def synth_t1_01(cards: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    """Build a T1-01 prompt over 4 cards. Gold is GPT-5.5's verdict."""
    chosen = rng.sample(cards, 4)
    genome_blocks = []
    for i, c in enumerate(chosen, 1):
        cd = c["card"]
        genome_blocks.append(
            f"[Genome G{i}]\n"
            f"  Mechanism: {cd.get('mechanism_genome','')}\n"
            f"  Niche: {cd.get('niche_genome','')}\n"
            f"  Observation: {cd.get('observation_genome','')}\n"
            f"  Claim: {cd.get('claim_genome','')}\n"
        )
    prompt = (
        "Classify the contribution type of ALL 4 genomes. All four genomes are from "
        "the same broad domain and have intentionally similar evidence. "
        "Options: method, dataset, analysis, system, theory\n\n"
        + "\n".join(genome_blocks)
        + "\nReturn the structured JSON answer with key `multi_contrib_types`."
    )
    user_inst = (
        prompt
        + "\n\nRequired JSON schema:\n"
        + '{\n  "multi_contrib_types": {"G1": "method|dataset|analysis|system|theory", '
          '"G2": "...", "G3": "...", "G4": "..."}\n}'
    )
    return chosen, {"prompt": prompt, "user": user_inst}


# T1-03: which gene is driver vs passenger (we need cards that list multi-gene)
def synth_t1_03(cards: list[dict], rng: random.Random) -> tuple[list[dict], dict] | None:
    """Synthesize a driver-vs-passenger question. Picks a card and constructs 2 mock genes."""
    c = rng.choice(cards)
    cd = c["card"]
    # build 2 mock genes from the card's fields
    g_driver = {"name": "mechanism", "text": cd.get("mechanism_genome", "")[:200]}
    g_passenger = {"name": "limitation", "text": cd.get("limitation_genome", "")[:200] or
                                                 cd.get("observation_genome","")[:200]}
    if not g_driver["text"] or not g_passenger["text"]:
        return None
    genes_order = [(0, g_driver), (1, g_passenger)]
    rng.shuffle(genes_order)
    labels = ["G1", "G2"]
    blocks = []
    for idx, (orig_idx, g) in enumerate(genes_order):
        blocks.append(f"[{labels[idx]}] {g['text']}")
    prompt = (
        "Given a single paper's two genes below, identify which is the DRIVER "
        "(the core mechanism that uniquely defines the paper's contribution) "
        "and which is the PASSENGER (a derived or downstream property).\n\n"
        + "\n".join(blocks)
        + '\n\nReturn JSON: {"driver_gene": "G1|G2", "passenger_gene": "G1|G2"}'
    )
    return [c], {"prompt": prompt, "user": prompt}


# T3-01: given two genomes (predecessor, successor), classify driver + dynamics
def synth_t3_01(cards: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    a, b = rng.sample(cards, 2)
    blocks = []
    for label, c in [("A (predecessor)", a), ("B (successor)", b)]:
        cd = c["card"]
        blocks.append(
            f"[Genome {label}]\n"
            f" Mechanism: {cd.get('mechanism_genome','')}\n"
            f" Niche: {cd.get('niche_genome','')}\n"
            f" Observation: {cd.get('observation_genome','')}\n"
            f" Limitation: {cd.get('limitation_genome','')}\n"
        )
    prompt = (
        "Given a predecessor genome A and successor genome B, classify the PRIMARY "
        "DRIVER of the change and the evolutionary DYNAMICS.\n\n"
        + "\n".join(blocks)
        + "\n\nDynamics options: Mutation | Adaptive Radiation | Hybridization | Speciation | Niche Competition\n"
        + "Driver options: mechanism | niche | observation | limitation\n\n"
        + 'Return JSON: {"driver": "mechanism|niche|observation|limitation", '
          '"dynamics": "Mutation|Adaptive Radiation|Hybridization|Speciation|Niche Competition"}'
    )
    return [a, b], {"prompt": prompt, "user": prompt}


# T4-01: given 6 genome fields supposedly from same paper, one is swapped in; identify
def synth_t4_01(cards: list[dict], rng: random.Random) -> tuple[list[dict], dict]:
    main = rng.choice(cards)
    other = rng.choice([c for c in cards if c is not main])
    fields = ["mechanism_genome", "niche_genome", "observation_genome",
              "limitation_genome", "delta_genome", "claim_genome"]
    # pick which field to swap (positions A-F)
    swap_idx = rng.randint(0, 3)  # only first 4 are usually populated
    labels = ["A", "B", "C", "D", "E", "F"]
    swap_label = labels[swap_idx]
    parts = []
    contrib_type = "method"  # arbitrary; ground truth comes from GPT-5.5
    for i, fld in enumerate(fields):
        if i == swap_idx:
            txt = other["card"].get(fld, "") or other["card"].get("mechanism_genome", "")
        else:
            txt = main["card"].get(fld, "") or "(none)"
        parts.append(f'{labels[i]}. "{txt[:200]}"')
    prompt = (
        "The following six unlabeled genome fields are supposedly from the SAME paper, "
        "but ONE has been swapped in from a closely related paper. "
        "Identify which field is the intruder, then classify the contribution_type of the original paper, "
        "and verify 4 statements as True/False (you'll make up these statements based on the genuine fields).\n\n"
        + "\n".join(parts)
        + "\n\nT/F statements:\n"
        + "1. The paper's primary mechanism is described in the field marked A.\n"
        + "2. The paper targets the niche described in the field marked B.\n"
        + "3. The observation described in field C is consistent with the paper's mechanism.\n"
        + "4. The limitation described in field D follows from the paper's approach.\n"
        + '\nReturn JSON: {"label": "A|B|C|D|E|F", '
          '"contribution_type": "method|dataset|analysis|system|theory", '
          '"verify": ["T|F", "T|F", "T|F", "T|F"]}'
    )
    return [main, other], {"prompt": prompt, "user": prompt}


SYNTHESIZERS = {
    "T1-01_contribution_type": synth_t1_01,
    "T1-03_driver_vs_passenger": synth_t1_03,
    "T3-01_single_dynamics": synth_t3_01,
    "T4-01_consistency_check": synth_t4_01,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-task", type=int, default=200)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--output", default=str(OUT_DIR / "round2_train.jsonl"))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"[1/4] Loading round 1 gene cards")
    cards = load_round1_cards()
    print(f"  {len(cards)} cards available")

    print(f"[2/4] Building call list")
    calls: list[TeacherCall] = []
    task_meta: list[dict] = []
    for tt, fn in SYNTHESIZERS.items():
        n_built = 0
        attempts = 0
        while n_built < args.n_per_task and attempts < args.n_per_task * 3:
            attempts += 1
            try:
                res = fn(cards, rng)
            except ValueError:
                continue
            if res is None:
                continue
            source_papers, payload = res
            calls.append(TeacherCall(
                prompt_id=f"r2::{tt}::{n_built:04d}",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload["user"]},
                ],
                max_tokens=512,
                metadata={
                    "task_type": tt,
                    "prompt_for_train": payload["prompt"],
                    "source_paper_ids": [c["paper_id"] for c in source_papers],
                },
            ))
            n_built += 1
        print(f"  {tt}: built {n_built} calls")
    print(f"  total: {len(calls)} calls")

    print(f"[3/4] Running teacher (workers={args.workers})")
    log_path = OUT_DIR.parent / "teacher_logs" / "gpt55_sft_round2.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    def cb(d, t): print(f"  teacher progress: {d}/{t}", flush=True)
    t0 = time.time()
    results = batch_call(calls, workers=args.workers, log_path=log_path, on_progress=cb)
    el = time.time() - t0
    n_ok = sum(1 for r in results if r.content and not r.error)
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
        tt = md.get("task_type", "?")
        # We don't have a gold answer here — verifier just checks schema_valid + dynamics_consistency.
        score, _ = compute_verifier(r.content, tt)
        if score.schema_valid < 1.0:
            stats[f"reject_schema_{tt}"] += 1
            continue
        stats[f"accept_{tt}"] += 1
        # write in same SFT envelope shape as round 1
        out_fp.write(json.dumps({
            "instance_id": r.prompt_id,
            "task_type": tt,
            "prompt": md["prompt_for_train"],
            "completion": r.content,
            "metadata": {
                "source_paper_ids": md["source_paper_ids"],
                "teacher_model": "gpt-5.5",
                "teacher_input_tokens": r.input_tokens,
                "teacher_output_tokens": r.output_tokens,
                "round": 2,
                "verifier_score": score.to_dict(),
            },
        }, ensure_ascii=False) + "\n")
    out_fp.close()

    n_accept = sum(v for k, v in stats.items() if k.startswith("accept_"))
    print(f"  acceptance rate: {n_accept}/{sum(stats.values())} = {100*n_accept/max(sum(stats.values()),1):.1f}%")
    print(f"  detail: {stats}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
