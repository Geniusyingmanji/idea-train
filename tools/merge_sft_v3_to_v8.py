"""Merge all SFT demo rounds (v3 + v4 + v5 + v6 + v8) into one combined file
with deduplication by full_prompt and basic quality filters.

Output: data/agentic_combined_v3to8/sft_demos.jsonl
"""
from __future__ import annotations
import json, hashlib, collections, sys
from pathlib import Path

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data")
OUT_DIR = ROOT / "agentic_combined_v3to23"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "sft_demos.jsonl"

INPUTS = [
    ROOT / "agentic_v3" / "sft_demos.jsonl",
    ROOT / "agentic_v4" / "sft_demos.jsonl",
    ROOT / "agentic_v5" / "sft_demos.jsonl",
    ROOT / "agentic_v6" / "sft_demos.jsonl",
    ROOT / "agentic_v8" / "sft_demos.jsonl",
    ROOT / "agentic_v9" / "sft_demos.jsonl",
    ROOT / "agentic_v11" / "sft_demos.jsonl",
    ROOT / "agentic_v12" / "sft_demos.jsonl",
    ROOT / "agentic_v14" / "sft_demos.jsonl",
    ROOT / "agentic_v15" / "sft_demos.jsonl",
    ROOT / "agentic_v16" / "sft_demos.jsonl",
    ROOT / "agentic_v18" / "sft_demos.jsonl",
    ROOT / "agentic_v21" / "sft_demos.jsonl",
    ROOT / "agentic_v23" / "sft_demos.jsonl",
]


def normalize_prompt(p: str) -> str:
    return " ".join(p.lower().split())[:300]


def main():
    seen = set()
    out_rows = []
    src_counts = collections.Counter()
    drop_dup = drop_short = drop_no_propose = 0

    for path in INPUTS:
        if not path.exists():
            print(f"  skipping (not found): {path}")
            continue
        with path.open() as f:
            for line in f:
                try: d = json.loads(line)
                except: continue
                key = hashlib.sha1(normalize_prompt(d.get("full_prompt", "")).encode()).hexdigest()
                if key in seen: drop_dup += 1; continue
                seen.add(key)
                comp = d.get("completion", "")
                if len(comp) < 250: drop_short += 1; continue
                if comp.count("```action") < 1 or '"propose"' not in comp:
                    drop_no_propose += 1; continue
                src = path.parent.name
                d["_source_round"] = src
                src_counts[src] += 1
                out_rows.append(d)

    with OUT_FILE.open("w") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # stats
    ns = [r["completion"].count("```action") for r in out_rows]
    ns.sort()
    print(f"\nCombined: {len(out_rows)} demos")
    print(f"  drops: dup={drop_dup}, short={drop_short}, no_propose={drop_no_propose}")
    print(f"  by source: {dict(src_counts)}")
    print(f"  tools/demo: median={ns[len(ns)//2]} mean={sum(ns)/len(ns):.2f} p25={ns[len(ns)//4]} p75={ns[3*len(ns)//4]}")
    print(f"    short (1-3): {sum(1 for n in ns if n<=3)} ({sum(1 for n in ns if n<=3)/len(ns)*100:.0f}%)")
    print(f"    med (4-6):  {sum(1 for n in ns if 4<=n<=6)} ({sum(1 for n in ns if 4<=n<=6)/len(ns)*100:.0f}%)")
    print(f"    long (7+):  {sum(1 for n in ns if n>=7)} ({sum(1 for n in ns if n>=7)/len(ns)*100:.0f}%)")
    discipline_keys = [r.get("discipline") for r in out_rows if r.get("discipline")]
    print(f"  disciplines: {len(set(discipline_keys))} unique")
    langs = collections.Counter(r.get("lang", "en") for r in out_rows)
    print(f"  langs: {dict(langs)}")
    print(f"saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
