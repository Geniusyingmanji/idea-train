"""Find traces where v3 beats evo-OPD-v5 by largest margin, and vice versa.

Produces a small qualitative table for paper inclusion.
"""
from __future__ import annotations

import json
from pathlib import Path

PE_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/arena_merged/population_eval")
IDEAS_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/arena_merged/ideas")

PIDS_TO_COMPARE = ["qwen3-8b-sft-v3", "qwen3-8b-evo-opd-v5"]
SETTINGS = ["Library", "Lineage", "Question"]

scores = {}  # scores[trace_id][setting][pid] = pes
for tdir in sorted(PE_DIR.iterdir()):
    if not tdir.is_dir():
        continue
    tid = tdir.name
    scores[tid] = {}
    for f in tdir.glob("*.json"):
        try:
            parts = f.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            pid, setting = parts
            if pid not in PIDS_TO_COMPARE or setting not in SETTINGS:
                continue
            data = json.loads(f.read_text())
            pes = data.get("scores", {}).get("gene_arena_score")
            if pes is None:
                continue
            scores[tid].setdefault(setting, {})[pid] = pes
        except Exception:
            pass

# Compute per-setting margins (v3 - evo-OPD-v5)
margins = []
for tid, s_data in scores.items():
    for setting, pid_scores in s_data.items():
        if len(pid_scores) != 2:
            continue
        margin = pid_scores["qwen3-8b-sft-v3"] - pid_scores["qwen3-8b-evo-opd-v5"]
        margins.append({
            "trace_id": tid, "setting": setting, "margin": margin,
            "v3": pid_scores["qwen3-8b-sft-v3"],
            "evo_v5": pid_scores["qwen3-8b-evo-opd-v5"],
        })

margins.sort(key=lambda r: r["margin"])

print(f"=== Top-5 cases where EVO-OPD-v5 beats v3 by most (negative margins) ===\n")
for r in margins[:5]:
    print(f"  trace={r['trace_id']:<40} setting={r['setting']:<10} "
          f"v3={r['v3']:.1f}  evo={r['evo_v5']:.1f}  Δ={r['margin']:+.1f}")

print(f"\n=== Top-5 cases where v3 beats evo-OPD-v5 by most (positive margins) ===\n")
for r in margins[-5:][::-1]:
    print(f"  trace={r['trace_id']:<40} setting={r['setting']:<10} "
          f"v3={r['v3']:.1f}  evo={r['evo_v5']:.1f}  Δ={r['margin']:+.1f}")

# Print median + IQR + std for sanity
n = len(margins)
print(f"\nN={n} matched (v3, evo-OPD-v5) pairs")
margs = sorted(r["margin"] for r in margins)
print(f"  median margin: {margs[n//2]:+.2f}")
print(f"  Q1 / Q3: {margs[n//4]:+.2f} / {margs[3*n//4]:+.2f}")
print(f"  fraction v3>evo: {sum(1 for m in margs if m > 0)/n:.2%}")
print(f"  fraction tie:    {sum(1 for m in margs if m == 0)/n:.2%}")
print(f"  fraction evo>v3: {sum(1 for m in margs if m < 0)/n:.2%}")
