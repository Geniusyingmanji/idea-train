"""Build v10 = v7 (JSON-format) + v9 (plain-text-format) union.

Hypothesis: a model that learns BOTH output formats picks the right one
based on prompt cues. Closes v9's T2/T3/T4 regression while keeping v9's T1
gain (31.20%).

Result: ~4000 examples (~2154 v7 + ~1859 v9 - overlap dedup).
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
SRC1 = ROOT / "data/stage1_sft/train_v7.jsonl"
SRC2 = ROOT / "data/stage1_sft/train_v9.jsonl"
OUT  = ROOT / "data/stage1_sft/train_v10.jsonl"


def main():
    rows = []
    seen_keys = set()                                       # dedup on (task_type, prompt_hash)

    def add_from(path: Path, tag: str):
        added = 0
        for line in path.open():
            r = json.loads(line)
            # use (task_type, src_paper_id) as dedup key when possible
            md = r.get("metadata", {})
            tt = r.get("task_type", "?")
            spid = md.get("source_paper_id") or md.get("source_paper_ids")
            v9_regen = md.get("v9_regen", False)
            key = (tt, str(spid), bool(v9_regen))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            md["v10_source"] = tag
            r["metadata"] = md
            rows.append(r)
            added += 1
        return added

    n7 = add_from(SRC1, "v7_json")
    n9 = add_from(SRC2, "v9_plain")
    print(f"v7 added: {n7}")
    print(f"v9 added: {n9}")
    print(f"total:    {len(rows)}")

    OUT.write_text("")
    with OUT.open("a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote: {OUT}")

    # task-type breakdown
    from collections import Counter
    c = Counter((r["task_type"], r.get("metadata", {}).get("v10_source"))
                for r in rows)
    for (tt, src), n in sorted(c.items()):
        print(f"  {tt:<30}  [{src}]  {n}")


if __name__ == "__main__":
    main()
