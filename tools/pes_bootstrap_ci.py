"""Bootstrap 95% CI + pairwise p-values for PES means.

Bootstrap-by-trace, not by idea: each trace appears once per (pid, setting), and
ideas from the same trace share a topic, so naive bootstrap underestimates noise.
We resample TRACE-IDS with replacement, then collect every (pid, setting) idea
from those trace-ids. This preserves the paired structure.

Outputs:
  - stdout: table of (pid, setting, mean, 95% CI), pairwise diff matrix
  - pes_ci.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

PE_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/arena_merged/population_eval")
OUT = PE_DIR.parent / "pes_ci.json"
N_BOOT = 2000
SEED = 42

random.seed(SEED)

# bucket[trace_id][(pid, setting)] = pes_score
by_trace = defaultdict(dict)
for f in sorted(PE_DIR.rglob("*.json")):
    parts = f.stem.rsplit("_", 1)
    if len(parts) != 2:
        continue
    pid, setting = parts
    data = json.loads(f.read_text())
    tid = data.get("trace_id")
    pes = data.get("scores", {}).get("gene_arena_score")
    if pes is None or tid is None:
        continue
    by_trace[tid][(pid, setting)] = pes

trace_ids = sorted(by_trace.keys())
print(f"Loaded {len(trace_ids)} traces, {sum(len(v) for v in by_trace.values())} idea-scores")
keys = sorted({k for v in by_trace.values() for k in v.keys()})


def boot_mean_ci(samples_per_trace, n_boot=N_BOOT, alpha=0.05):
    """samples_per_trace: list-of-floats (one per trace, may be empty for missing)."""
    means = []
    n = len(samples_per_trace)
    for _ in range(n_boot):
        idx = [random.randrange(n) for _ in range(n)]
        vals = [v for i in idx for v in samples_per_trace[i]]
        if vals:
            means.append(mean(vals))
    means.sort()
    lo = means[int(alpha / 2 * len(means))]
    hi = means[int((1 - alpha / 2) * len(means))]
    return lo, hi


def boot_diff(samples_a, samples_b, n_boot=N_BOOT):
    """Paired bootstrap: same resampled trace indices for both samples."""
    n = len(samples_a)
    diffs = []
    n_pos = 0
    for _ in range(n_boot):
        idx = [random.randrange(n) for _ in range(n)]
        va = [v for i in idx for v in samples_a[i]]
        vb = [v for i in idx for v in samples_b[i]]
        if va and vb:
            d = mean(va) - mean(vb)
            diffs.append(d)
            if d > 0:
                n_pos += 1
    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    p_a_gt_b = n_pos / len(diffs) if diffs else 0.0
    return lo, hi, p_a_gt_b


# Build per-key arrays
arrs = {}
for k in keys:
    arr = []
    for tid in trace_ids:
        v = by_trace[tid].get(k)
        arr.append([v] if v is not None else [])
    arrs[k] = arr

# Also overall (all settings) per pid
pids = sorted({p for p, _ in keys})
settings_set = sorted({s for _, s in keys})
overall = {}
for pid in pids:
    arr = []
    for tid in trace_ids:
        vs = [by_trace[tid][(pid, s)] for s in settings_set if (pid, s) in by_trace[tid]]
        arr.append(vs)
    overall[pid] = arr

# Print per (pid, setting) means + CIs
print()
print("Per (participant, setting) — bootstrap 95% CI over trace_ids")
print("=" * 80)
print(f"{'participant':<28} {'setting':<10} {'n':>4} {'mean':>8} {'CI95':>20}")
for pid in pids:
    for s in settings_set:
        if (pid, s) not in arrs:
            continue
        arr = arrs[(pid, s)]
        vals = [v for sub in arr for v in sub]
        if not vals:
            continue
        m = mean(vals)
        lo, hi = boot_mean_ci(arr)
        print(f"{pid:<28} {s:<10} {len(vals):>4} {m:>8.2f}  [{lo:>5.2f}, {hi:>5.2f}]")
    arr = overall[pid]
    vals = [v for sub in arr for v in sub]
    if vals:
        m = mean(vals)
        lo, hi = boot_mean_ci(arr)
        print(f"{pid:<28} {'ALL':<10} {len(vals):>4} {m:>8.2f}  [{lo:>5.2f}, {hi:>5.2f}]")
    print("-" * 80)

# Pairwise comparisons (overall PES)
print()
print("Pairwise (overall PES) — paired bootstrap diff (A−B), 95% CI, P(A>B)")
print("=" * 80)
for i, pid_a in enumerate(pids):
    for pid_b in pids[i+1:]:
        d_lo, d_hi, p_a_gt_b = boot_diff(overall[pid_a], overall[pid_b])
        d_mean = mean([v for sub in overall[pid_a] for v in sub]) - \
                 mean([v for sub in overall[pid_b] for v in sub])
        sig = "*" if (d_lo > 0 or d_hi < 0) else " "
        print(f"  {pid_a:>22} vs {pid_b:<22}  Δ={d_mean:+5.2f}  "
              f"[{d_lo:+5.2f}, {d_hi:+5.2f}] {sig}  P(A>B)={p_a_gt_b:.3f}")

summary = {
    "n_traces": len(trace_ids),
    "n_boot": N_BOOT,
    "seed": SEED,
    "per_pid_overall": {},
    "per_pid_setting": {},
    "pairwise_overall": {},
}
for pid in pids:
    arr = overall[pid]
    vals = [v for sub in arr for v in sub]
    if vals:
        lo, hi = boot_mean_ci(arr)
        summary["per_pid_overall"][pid] = {"mean": mean(vals), "lo": lo, "hi": hi, "n": len(vals)}
    summary["per_pid_setting"][pid] = {}
    for s in settings_set:
        if (pid, s) not in arrs:
            continue
        arr = arrs[(pid, s)]
        vals = [v for sub in arr for v in sub]
        if vals:
            lo, hi = boot_mean_ci(arr)
            summary["per_pid_setting"][pid][s] = {
                "mean": mean(vals), "lo": lo, "hi": hi, "n": len(vals)
            }
for i, pid_a in enumerate(pids):
    for pid_b in pids[i+1:]:
        d_lo, d_hi, p_a_gt_b = boot_diff(overall[pid_a], overall[pid_b])
        d_mean = mean([v for sub in overall[pid_a] for v in sub]) - \
                 mean([v for sub in overall[pid_b] for v in sub])
        summary["pairwise_overall"][f"{pid_a}_vs_{pid_b}"] = {
            "diff_mean": d_mean, "diff_lo": d_lo, "diff_hi": d_hi,
            "p_a_gt_b": p_a_gt_b,
        }
OUT.write_text(json.dumps(summary, indent=2))
print(f"\nWrote {OUT}")
