"""Merge all DPO preference pairs (v7 + v10 + v13) into a unified file with
consistent schema for downstream DPO training.

Output: data/dpo_combined/preferences.jsonl
"""
import json, collections
from pathlib import Path

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data")
OUT_DIR = ROOT / "dpo_combined"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "preferences.jsonl"

SOURCES = [
    (ROOT / "agentic_v7" / "preferences.jsonl", "v7"),
    (ROOT / "agentic_v10" / "preferences.jsonl", "v10"),
    (ROOT / "agentic_v13" / "preferences.jsonl", "v13"),
    (ROOT / "agentic_v19" / "preferences.jsonl", "v19"),
    (ROOT / "agentic_v22" / "preferences.jsonl", "v22"),
    (ROOT / "agentic_v27" / "preferences.jsonl", "v27"),
]


def main():
    out_rows, src_counts = [], collections.Counter()
    for path, src in SOURCES:
        if not path.exists():
            print(f"  skip: {path}"); continue
        with path.open() as f:
            for line in f:
                try: p = json.loads(line)
                except: continue
                # normalize schema
                row = {
                    "prompt_id": p.get("prompt_id"),
                    "source_round": src,
                    "discipline": p.get("discipline", "unknown"),
                    "lang": p.get("lang", "en"),
                    "rejection_mode": p.get("rejection_mode") or p.get("corruption", "unknown"),
                    "full_prompt": p.get("full_prompt", ""),
                    "candidates": p.get("candidates", []),
                    "chosen": p.get("chosen", ""),
                    "rejected": p.get("rejected", ""),
                }
                # skip empties
                if len(row["chosen"]) < 100 or len(row["rejected"]) < 50: continue
                out_rows.append(row)
                src_counts[src] += 1

    with OUT_FILE.open("w") as f:
        for r in out_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nCombined DPO: {len(out_rows)} pairs")
    print(f"  by source: {dict(src_counts)}")
    print(f"  by lang: {dict(collections.Counter(r['lang'] for r in out_rows))}")
    print(f"  by mode: {dict(collections.Counter(r['rejection_mode'] for r in out_rows))}")
    print(f"saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
