"""Produce a side-by-side baseline vs trained comparison report."""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--baseline", required=True, help="path to merged_summary.json OR summary_shard*.json")
ap.add_argument("--trained", required=True)
ap.add_argument("--output", default="/home/azureuser/workspace-gzy/zyf/idea_train/eval/results/comparison.md")
args = ap.parse_args()


def load(p):
    p = Path(p)
    if p.is_dir():
        merged = p / "merged_summary.json"
        if merged.exists():
            return json.loads(merged.read_text())
        # fall back: single-shard
        cand = list(p.glob("summary_shard*.json"))
        if cand:
            return json.loads(cand[0].read_text())
        cand = list(p.glob("summary.json"))
        if cand:
            return json.loads(cand[0].read_text())
        raise FileNotFoundError(p)
    return json.loads(p.read_text())


b = load(args.baseline)
t = load(args.trained)

lines = [
    "# GENE-Exam comparison: baseline vs trained",
    "",
    f"- Baseline: `{args.baseline}`",
    f"- Trained : `{args.trained}`",
    "",
    "## Headline",
    "",
    f"| metric          | baseline | trained | delta |",
    f"|-----------------|---------:|--------:|------:|",
    f"| n_instances     | {b['n_instances']:>7} | {t['n_instances']:>7} | {t['n_instances']-b['n_instances']:+d} |",
    f"| macro_accuracy  | {b['macro_accuracy']*100:>6.2f}% | {t['macro_accuracy']*100:>6.2f}% | {(t['macro_accuracy']-b['macro_accuracy'])*100:+.2f} pts |",
    "",
    "## Per tier",
    "",
    "| tier | baseline | trained | delta |",
    "|------|---------:|--------:|------:|",
]
b_tier = b.get("per_tier_macro", {})
t_tier = t.get("per_tier_macro", {})
for k in ("T1", "T2", "T3", "T4"):
    bv = b_tier.get(k, 0); tv = t_tier.get(k, 0)
    lines.append(f"| {k}   | {bv*100:>6.2f}% | {tv*100:>6.2f}% | {(tv-bv)*100:+.2f} pts |")
lines += ["", "## Per task", "", "| task | baseline | trained | delta |", "|------|---------:|--------:|------:|"]
b_task = b.get("per_task_accuracy", {})
t_task = t.get("per_task_accuracy", {})
all_tasks = sorted(set(b_task) | set(t_task))
for task in all_tasks:
    bv = b_task.get(task, 0); tv = t_task.get(task, 0)
    arrow = " ↑" if tv > bv else (" ↓" if tv < bv else "  ")
    lines.append(f"| {task:<32} | {bv*100:>6.2f}% | {tv*100:>6.2f}% | {(tv-bv)*100:+6.2f}{arrow} |")

out = Path(args.output)
out.write_text("\n".join(lines))
print("\n".join(lines))
print(f"\nWrote: {out}")
