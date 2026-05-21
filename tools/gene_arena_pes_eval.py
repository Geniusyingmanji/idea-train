"""PES (Population-Evolving Score) eval wrapper.

Merges arena_<participant>/ idea dirs into one results_dir, then calls
gene_arena.population_eval.run_population_eval against the Azure GPT-5.5
(keyless) endpoint.

Reuses arena's own scoring logic — no re-implementation. PES output dropped
to results_dir/population_eval/<trace_id>/<participant>_<setting>.json.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# IdeaEvolving paths
IDEA_EVOLVING = Path("/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
sys.path.insert(0, str(IDEA_EVOLVING))
sys.path.insert(0, str(IDEA_EVOLVING / "gene_arena"))

os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from gene_arena.adapters import make_client
from gene_arena.arena_config import PROVIDERS, TASK_DIR
from gene_arena.population_eval import run_population_eval

OUR_ARENA_ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/eval/results")
DEFAULT_MERGED_DIR = OUR_ARENA_ROOT / "arena_merged"

# (subdir-name-under-eval-results, participant-id)
DEFAULT_RUNS = [
    ("arena_baseline",    "qwen3-8b-baseline"),
    ("arena_v3",          "qwen3-8b-sft-v3"),
    ("arena_v8",          "qwen3-8b-sft-v8"),
    ("arena_v10",         "qwen3-8b-sft-v10"),
    ("arena_evo_opd_v4",  "qwen3-8b-evo-opd-v4"),
    ("arena_evo_opd_v5",  "qwen3-8b-evo-opd-v5"),
]


def merge_idea_dirs(src_pairs: list[tuple[str, str]], dst: Path) -> tuple[set[str], set[str]]:
    """Symlink each src_pair's idea files into dst/ideas/<trace>/<pid>_<setting>.json.
    Returns (trace_ids, participants) sets."""
    dst_ideas = dst / "ideas"
    dst_ideas.mkdir(parents=True, exist_ok=True)
    trace_ids = set()
    participants = set()
    for sub, pid in src_pairs:
        src = OUR_ARENA_ROOT / sub / "ideas"
        if not src.exists():
            print(f"  SKIP {sub}: no ideas/ dir")
            continue
        participants.add(pid)
        for trace_dir in sorted(src.iterdir()):
            if not trace_dir.is_dir():
                continue
            tid = trace_dir.name
            trace_ids.add(tid)
            d = dst_ideas / tid
            d.mkdir(parents=True, exist_ok=True)
            for f in trace_dir.glob(f"{pid}_*.json"):
                target = d / f.name
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(f.resolve())
    return trace_ids, participants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged-dir", default=str(DEFAULT_MERGED_DIR))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--settings", nargs="+", default=["Library", "Lineage", "Question"])
    args = ap.parse_args()

    merged = Path(args.merged_dir)
    print(f"[1/3] merging idea dirs into {merged}")
    trace_ids, participants = merge_idea_dirs(DEFAULT_RUNS, merged)
    print(f"  {len(trace_ids)} trace_ids, {len(participants)} participants: {sorted(participants)}")

    print(f"[2/3] building Azure GPT-5.5 client")
    azure_cfg = PROVIDERS["azure"]
    client = make_client(azure_cfg)

    print(f"[3/3] running PES eval (workers={args.workers})")
    run_population_eval(
        trace_ids=sorted(trace_ids),
        results_dir=merged,
        task_dir=Path(TASK_DIR),
        participants=sorted(participants),
        settings=args.settings,
        client=client,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
