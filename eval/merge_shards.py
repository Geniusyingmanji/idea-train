"""Merge per-shard summaries into one overall report."""
import argparse, glob, json
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True, help="parent dir containing shard*/")
ap.add_argument("--output", default=None)
args = ap.parse_args()

base = Path(args.dir)
per_task_correct = defaultdict(int)
per_task_total = defaultdict(int)
total_inst = 0
total_correct = 0
shards_seen = []

for shard_summary in sorted(base.glob("shard*/summary_shard*.json")):
    s = json.loads(shard_summary.read_text())
    shards_seen.append(s.get("shard", "?"))
    for t, acc in s.get("per_task_accuracy", {}).items():
        # we need raw counts, but summary stores acc; back out using per_instance.jsonl
        pass
    total_inst += s["n_instances"]
    total_correct += s["n_correct"]

# walk per_instance for exact counts
for inst_file in sorted(base.glob("shard*/per_instance_shard*.jsonl")):
    with inst_file.open() as f:
        for line in f:
            r = json.loads(line)
            per_task_total[r["task_type"]] += 1
            if r["is_correct"]:
                per_task_correct[r["task_type"]] += 1

per_task_acc = {t: per_task_correct[t] / per_task_total[t] for t in sorted(per_task_total) if per_task_total[t]}
tiers = defaultdict(list)
for t, a in per_task_acc.items():
    for k in ("T1", "T2", "T3", "T4"):
        if t.startswith(f"{k}-"):
            tiers[k].append(a)

merged = {
    "shards": shards_seen,
    "n_instances": sum(per_task_total.values()),
    "n_correct": sum(per_task_correct.values()),
    "macro_accuracy": sum(per_task_correct.values()) / max(sum(per_task_total.values()), 1),
    "per_tier_macro": {k: sum(v) / len(v) for k, v in tiers.items()},
    "per_task_accuracy": per_task_acc,
    "per_task_n": dict(per_task_total),
}
out = Path(args.output) if args.output else base / "merged_summary.json"
out.write_text(json.dumps(merged, indent=2))
print(f"=== {base.name} ===")
print(f"  n: {merged['n_instances']}  correct: {merged['n_correct']}")
print(f"  macro_accuracy: {merged['macro_accuracy']*100:.2f}%")
print(f"  per_tier_macro: {merged['per_tier_macro']}")
print(f"Wrote: {out}")
