"""Aggregate PES per-idea scores into a (participant × setting) table.

Reads results/arena_merged/population_eval/<trace_id>/<pid>_<setting>.json,
prints mean / count per (participant, setting), plus an overall row and a
per-dimension breakdown.

Output:
  - stdout pretty table
  - results/arena_merged/pes_summary.json   (machine-readable)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

PE_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/arena_merged/population_eval")
OUT_PATH = PE_DIR.parent / "pes_summary.json"

POP_DIMS = ["parent_grounding", "gene_inheritance", "limitation_repair",
            "evolutionary_plausibility", "balanced_novelty", "graph_insertion"]
SQ_DIMS = ["problem_importance", "mechanism_concreteness", "originality",
           "feasibility", "validation_rigor", "expected_impact"]

# bucket: rows[(pid, setting)] = list of records
rows = defaultdict(list)
all_files = sorted(PE_DIR.rglob("*.json"))
for f in all_files:
    parts = f.stem.rsplit("_", 1)  # "<pid>_<setting>"
    if len(parts) != 2:
        continue
    pid, setting = parts
    try:
        data = json.loads(f.read_text())
    except Exception as e:
        print(f"  skip {f}: {e}")
        continue
    scores = data.get("scores", {})
    rec = {
        "trace_id": data.get("trace_id"),
        "pes": scores.get("gene_arena_score"),
        "pop": scores.get("population_score"),
        "sq": scores.get("scientific_quality_score"),
        "pop_sub": scores.get("population_subscores", {}),
        "sq_sub": scores.get("scientific_quality_subscores", {}),
        "n_valid_judges": data.get("layer3", {}).get("n_valid_judges", 0),
    }
    if rec["pes"] is not None:
        rows[(pid, setting)].append(rec)

print(f"Loaded {sum(len(v) for v in rows.values())} scored ideas across "
      f"{len(rows)} (participant × setting) buckets\n")

# print main table
print("=" * 90)
print(f"{'participant':<28} {'setting':<10} {'n':>4} {'PES mean':>10} {'±sd':>8} "
      f"{'pop':>7} {'sq':>7}")
print("-" * 90)
participants = sorted({p for p, _ in rows.keys()})
settings = sorted({s for _, s in rows.keys()})

per_pid_totals = defaultdict(list)
for pid in participants:
    for setting in settings:
        recs = rows.get((pid, setting), [])
        if not recs:
            continue
        ps = [r["pes"] for r in recs]
        pop = [r["pop"] for r in recs]
        sq = [r["sq"] for r in recs]
        per_pid_totals[pid].extend(ps)
        sd = stdev(ps) if len(ps) > 1 else 0.0
        print(f"{pid:<28} {setting:<10} {len(recs):>4} {mean(ps):>10.2f} {sd:>8.2f} "
              f"{mean(pop):>7.2f} {mean(sq):>7.2f}")
    if per_pid_totals[pid]:
        all_ps = per_pid_totals[pid]
        sd = stdev(all_ps) if len(all_ps) > 1 else 0.0
        print(f"{pid:<28} {'ALL':<10} {len(all_ps):>4} {mean(all_ps):>10.2f} {sd:>8.2f}")
        print("-" * 90)

# per-dim mean for each participant (collapsed across settings)
print()
print("Population sub-dimensions (mean across all settings):")
print("=" * 90)
hdr = f"{'pid':<28} " + " ".join(f"{d[:8]:>9}" for d in POP_DIMS)
print(hdr)
print("-" * 90)
for pid in participants:
    all_recs = [r for (p, _), v in rows.items() if p == pid for r in v]
    if not all_recs:
        continue
    row_vals = []
    for d in POP_DIMS:
        vals = [r["pop_sub"].get(d) for r in all_recs if r["pop_sub"].get(d) is not None]
        row_vals.append(mean(vals) if vals else 0.0)
    print(f"{pid:<28} " + " ".join(f"{v:>9.2f}" for v in row_vals))

print()
print("Scientific-quality sub-dimensions (mean across all settings):")
print("=" * 90)
hdr = f"{'pid':<28} " + " ".join(f"{d[:8]:>9}" for d in SQ_DIMS)
print(hdr)
print("-" * 90)
for pid in participants:
    all_recs = [r for (p, _), v in rows.items() if p == pid for r in v]
    if not all_recs:
        continue
    row_vals = []
    for d in SQ_DIMS:
        vals = [r["sq_sub"].get(d) for r in all_recs if r["sq_sub"].get(d) is not None]
        row_vals.append(mean(vals) if vals else 0.0)
    print(f"{pid:<28} " + " ".join(f"{v:>9.2f}" for v in row_vals))

# machine-readable dump
summary = {
    "n_files": len(all_files),
    "participants": participants,
    "settings": settings,
    "per_pid_setting": {},
    "per_pid_overall": {},
    "per_pid_pop_dims": {},
    "per_pid_sq_dims": {},
}
for pid in participants:
    pid_data = {}
    for setting in settings:
        recs = rows.get((pid, setting), [])
        if not recs:
            continue
        pid_data[setting] = {
            "n": len(recs),
            "pes_mean": mean(r["pes"] for r in recs),
            "pes_sd": stdev([r["pes"] for r in recs]) if len(recs) > 1 else 0.0,
            "pop_mean": mean(r["pop"] for r in recs),
            "sq_mean": mean(r["sq"] for r in recs),
        }
    summary["per_pid_setting"][pid] = pid_data
    all_ps = per_pid_totals[pid]
    if all_ps:
        summary["per_pid_overall"][pid] = {
            "n": len(all_ps),
            "pes_mean": mean(all_ps),
            "pes_sd": stdev(all_ps) if len(all_ps) > 1 else 0.0,
        }
    all_recs = [r for (p, _), v in rows.items() if p == pid for r in v]
    summary["per_pid_pop_dims"][pid] = {
        d: mean([r["pop_sub"].get(d) for r in all_recs if r["pop_sub"].get(d) is not None] or [0.0])
        for d in POP_DIMS
    }
    summary["per_pid_sq_dims"][pid] = {
        d: mean([r["sq_sub"].get(d) for r in all_recs if r["sq_sub"].get(d) is not None] or [0.0])
        for d in SQ_DIMS
    }

OUT_PATH.write_text(json.dumps(summary, indent=2))
print(f"\nWrote {OUT_PATH}")
