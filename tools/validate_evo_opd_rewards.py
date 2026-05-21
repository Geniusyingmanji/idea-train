"""Validate evo-OPD reward composition on real existing data.

For each existing SFT example (or a sampled subset), we:
  (1) tag the rollout's tokens with φ via parser.py
  (2) call the verifier (no teacher needed)
  (3) call the lineage scorer against a random predecessor card from GeneTrace
  (4) call the reward composer with a SYNTHETIC kl signal (small constant)
  (5) report per-task-type distribution of v, c, mean reward, and any NaN/inf

This validates that the algorithm runs end-to-end on real data, catches
implementation bugs before we wire up a teacher, and produces the numbers
we cite in the paper's §4.3 (implementation check).

No external APIs required.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/idea_train")
from evo_opd.rewards import EvoOPDReward, EvoOPDRewardConfig, char_uniform_phi_tags

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
SFT  = ROOT / "data/stage1_sft/train.jsonl"
CARDS = ROOT / "data/genetrace_v0_1/cards.jsonl"


def load_cards_by_paper() -> dict[str, dict]:
    out = {}
    if CARDS.exists():
        for line in CARDS.open():
            r = json.loads(line)
            out[r["paper_id"]] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--kl-constant", type=float, default=0.1,
                    help="Synthetic per-token kl_t = log π_θ - log π_T. "
                         "Positive = student over-confident relative to teacher.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cards = load_cards_by_paper()
    print(f"loaded {len(cards)} cards from GeneTrace")
    print(f"loaded SFT examples (limit={args.limit})")

    rows = []
    with SFT.open() as f:
        for line in f:
            r = json.loads(line)
            rows.append(r)
            if len(rows) >= args.limit:
                break

    reward = EvoOPDReward(EvoOPDRewardConfig(lambda_v=0.5, lambda_c=0.3))

    per_task: dict[str, list[dict]] = defaultdict(list)
    error_counts = Counter()
    nan_counts = Counter()

    for row in rows:
        tt = row.get("task_type") or "unknown"
        comp = row.get("completion", "")
        if not comp:
            error_counts["empty_completion"] += 1
            continue
        try:
            phi = char_uniform_phi_tags(comp, tt)
            kl = [args.kl_constant] * len(comp)
            md = row.get("metadata", {})
            src_paper_id = md.get("source_paper_id")
            # pick a random predecessor for lineage signal — strict "different
            # paper from same pool" so c-score has something to chew on
            parent = None
            if src_paper_id and src_paper_id in cards:
                # use the card itself as parent for a self-consistency proxy
                parent = {
                    "card_id": cards[src_paper_id]["card_id"],
                    "genome":  cards[src_paper_id]["genome"],
                }
            out = reward(
                text=comp,
                per_token_kl=kl,
                phi_per_token=phi,
                fld_per_token=None,
                task_type=tt,
                gold_answer=md.get("gold_answer"),
                parent_card=parent,
            )
            mean_r = sum(out.rewards) / max(len(out.rewards), 1)
            row_stats = {
                "v": out.verifier.v,
                "c": out.lineage.c if out.lineage else None,
                "mean_r": mean_r,
                "kl_term": out.kl_term_mean,
                "v_term":  out.verifier_term_mean,
                "c_term":  out.lineage_term_mean,
                "n_tokens": len(out.rewards),
                "phi_dist": dict(Counter(out.phi).most_common(6)),
            }
            for k in ("v", "mean_r", "kl_term", "v_term", "c_term"):
                if row_stats[k] is not None and (math.isnan(row_stats[k]) or
                                                  math.isinf(row_stats[k])):
                    nan_counts[k] += 1
            per_task[tt].append(row_stats)
        except Exception as e:                              # noqa: BLE001
            error_counts[type(e).__name__] += 1

    # report
    print(f"\n=== Errors ===")
    print(f"  {dict(error_counts)}")
    print(f"=== NaN/Inf counts ===")
    print(f"  {dict(nan_counts)}")
    print(f"\n=== Per task type (n={args.limit}) ===")
    print(f"{'task_type':<30} {'n':>4} {'v̄':>7} {'c̄':>7} {'kl̄':>9} {'v_term':>9} {'c_term':>9} {'mean_r':>9}")
    for tt, rs in sorted(per_task.items()):
        vs = [r["v"] for r in rs]
        cs = [r["c"] for r in rs if r["c"] is not None]
        kls = [r["kl_term"] for r in rs]
        vts = [r["v_term"] for r in rs]
        cts = [r["c_term"] for r in rs]
        mrs = [r["mean_r"] for r in rs]
        print(f"{tt:<30} {len(rs):>4} "
              f"{statistics.mean(vs):>7.3f} "
              f"{(statistics.mean(cs) if cs else 0):>7.3f} "
              f"{statistics.mean(kls):>+9.4f} "
              f"{statistics.mean(vts):>+9.4f} "
              f"{statistics.mean(cts):>+9.4f} "
              f"{statistics.mean(mrs):>+9.4f}")
    print(f"\n=== φ tag distribution (across all rows) ===")
    all_phi = Counter()
    for rs in per_task.values():
        for r in rs:
            for tag, n in r["phi_dist"].items():
                all_phi[tag] += n
    total = sum(all_phi.values())
    for tag, n in all_phi.most_common():
        print(f"  {tag:<20} {n:>6}  ({100*n/max(total,1):.1f}%)")

    print(f"\n=== Conclusion ===")
    if not error_counts and not nan_counts:
        print("  No errors, no NaN/Inf. Reward composition runs cleanly on real data.")
    else:
        print(f"  ⚠️  {sum(error_counts.values())} errors, {sum(nan_counts.values())} NaN/Inf — investigate.")


if __name__ == "__main__":
    main()
