"""Build denylist v0 from local IdeaEvolving assets.

Extracts every paper reference appearing in:
  - data/genome_db/paper_gene_cards.json     (2,076 cards)
  - data/genome_db/trace_graphs.json         (90 lineage traces, ~4,500 nodes)
  - data/genome_db/trace_gene_graphs.json    (176 enriched traces)
  - gene_arena/task/*.json                   (50 frontier task traces)

Cross-resolves missing s2_id via data/paper_db/paper_db.json (29,472 papers).

Output:
  idea_train/denylist/denylist_v0.jsonl   one paper per line
  idea_train/denylist/denylist_v0.csv     spreadsheet-friendly
  idea_train/denylist/denylist_stats.json

Usage:
  python tools/build_denylist.py
"""
from __future__ import annotations

import csv
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path("/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/denylist")
OUT.mkdir(parents=True, exist_ok=True)


def norm_title(t: str | None) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def safe_int_year(y) -> int | None:
    try:
        return int(y) if y else None
    except (TypeError, ValueError):
        return None


def main() -> None:
    # --- 1. paper_db registry (used for cross-resolution) -------------------
    print("[1/5] loading paper_db.json for cross-resolution ...")
    with (REPO / "data/paper_db/paper_db.json").open() as f:
        paper_db = json.load(f)
    # title-year -> s2_id index
    title_year_to_s2: dict[tuple[str, int | None], str] = {}
    paper_id_to_s2: dict[str, str] = {}
    for pid, rec in paper_db.items():
        s2 = rec.get("s2_id")
        if s2:
            paper_id_to_s2[pid] = s2
            title_year_to_s2[(norm_title(rec.get("title")), safe_int_year(rec.get("year")))] = s2
    print(f"  paper_db: {len(paper_db):,} records, {len(paper_id_to_s2):,} have s2_id")

    # --- 2. accumulator -----------------------------------------------------
    # primary key heuristic: prefer s2_id; else normalized (title, year)
    refs: dict[str, dict] = {}  # key -> record

    def add(ref: dict, source: str) -> None:
        s2 = ref.get("s2_id")
        nt = norm_title(ref.get("title"))
        yr = safe_int_year(ref.get("year"))
        # try cross-resolve s2_id via paper_db
        if not s2 and nt:
            s2 = title_year_to_s2.get((nt, yr)) or title_year_to_s2.get((nt, None))
            if s2:
                ref["s2_id"] = s2
                ref["s2_id_via"] = "paper_db_xref"
        key = f"s2:{s2}" if s2 else f"ty:{nt}|{yr}"
        if key in refs:
            existing = refs[key]
            existing["sources"].add(source)
            # fill missing fields
            for k, v in ref.items():
                if v and not existing.get(k):
                    existing[k] = v
        else:
            ref["sources"] = {source}
            refs[key] = ref

    # --- 3. paper_gene_cards.json ------------------------------------------
    print("[2/5] paper_gene_cards.json ...")
    with (REPO / "data/genome_db/paper_gene_cards.json").open() as f:
        cards = json.load(f)
    for c in cards:
        add({
            "title": c.get("title"),
            "year": c.get("year"),
            "internal_paper_id": c.get("paper_id"),  # internal hash like paper_9f6204587f
            "trace_id": c.get("trace_id"),
            "domain": c.get("domain"),
            "subfield": c.get("subfield"),
        }, "gene_card")
    print(f"  +{len(cards):,} cards")

    # --- 4. trace_graphs.json (nodes have s2_id) ---------------------------
    print("[3/5] trace_graphs.json ...")
    with (REPO / "data/genome_db/trace_graphs.json").open() as f:
        traces = json.load(f)
    node_count = 0
    for t in traces:
        for n in t.get("nodes") or []:
            node_count += 1
            add({
                "s2_id": n.get("s2_id"),
                "title": n.get("title"),
                "year": n.get("year"),
                "internal_paper_id": n.get("paper_id"),
                "trace_id": t.get("trace_id"),
                "domain": t.get("domain"),
                "subfield": t.get("subfield"),
                "role": n.get("role"),
            }, "trace_graph")
    print(f"  +{node_count:,} nodes across {len(traces):,} traces")

    # --- 5. trace_gene_graphs.json (uses paper_ids; lookup in paper_db) ---
    print("[4/5] trace_gene_graphs.json ...")
    with (REPO / "data/genome_db/trace_gene_graphs.json").open() as f:
        tg = json.load(f)
    tg_paper_refs = 0
    for t in tg:
        for pid in t.get("paper_ids") or []:
            tg_paper_refs += 1
            rec = paper_db.get(pid, {})
            add({
                "s2_id": rec.get("s2_id"),
                "title": rec.get("title"),
                "year": rec.get("year"),
                "internal_paper_id": pid,
                "trace_id": t.get("trace_id"),
                "domain": t.get("domain"),
                "subfield": t.get("subfield"),
            }, "trace_gene_graph")
    print(f"  +{tg_paper_refs:,} paper_id refs across {len(tg):,} enriched traces")

    # --- 6. gene_arena/task/*.json -----------------------------------------
    print("[5/5] gene_arena/task/*.json ...")
    arena_files = sorted(glob.glob(str(REPO / "gene_arena/task/*.json")))
    arena_papers = 0
    for af in arena_files:
        with open(af) as f:
            t = json.load(f)
        for p in t.get("papers") or []:
            arena_papers += 1
            add({
                "title": p.get("title"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "arxiv_id": p.get("arxiv_id"),
                "trace_id": t.get("trace_id"),
                "domain": t.get("domain"),
                "subfield": t.get("subfield"),
            }, "arena_task")
    print(f"  +{arena_papers:,} arena papers across {len(arena_files)} tasks")

    # --- 7. write outputs ---------------------------------------------------
    print()
    print(f"Total unique paper references: {len(refs):,}")
    print()

    out_jsonl = OUT / "denylist_v0.jsonl"
    out_csv = OUT / "denylist_v0.csv"

    with out_jsonl.open("w") as f:
        for r in refs.values():
            r["sources"] = sorted(r["sources"])
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_fields = [
        "s2_id", "arxiv_id", "internal_paper_id",
        "title", "year", "venue", "domain", "subfield",
        "trace_id", "role", "sources",
    ]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in refs.values():
            row = {k: r.get(k, "") for k in csv_fields}
            row["sources"] = ";".join(r["sources"]) if isinstance(r["sources"], list) else r["sources"]
            w.writerow(row)

    # --- 8. stats -----------------------------------------------------------
    src_counter: Counter[str] = Counter()
    domain_counter: Counter[str] = Counter()
    has_s2 = has_arxiv = has_title = 0
    for r in refs.values():
        for s in r["sources"]:
            src_counter[s] += 1
        if r.get("domain"):
            domain_counter[r["domain"]] += 1
        if r.get("s2_id"):
            has_s2 += 1
        if r.get("arxiv_id"):
            has_arxiv += 1
        if r.get("title"):
            has_title += 1
    cross_resolved = sum(1 for r in refs.values() if r.get("s2_id_via") == "paper_db_xref")

    stats = {
        "total_unique_papers": len(refs),
        "id_coverage": {
            "with_s2_id": has_s2,
            "with_arxiv_id": has_arxiv,
            "with_title": has_title,
            "s2_id_cross_resolved_via_paper_db": cross_resolved,
        },
        "appearances_per_source": dict(src_counter.most_common()),
        "papers_per_domain": dict(domain_counter.most_common()),
        "output_files": [str(out_jsonl), str(out_csv)],
    }

    with (OUT / "denylist_stats.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print("=== Stats ===")
    print(f"  with s2_id        : {has_s2:>6} / {len(refs):,}  ({100*has_s2/len(refs):.1f}%)")
    print(f"  with arxiv_id     : {has_arxiv:>6} / {len(refs):,}  ({100*has_arxiv/len(refs):.1f}%)")
    print(f"  with title        : {has_title:>6} / {len(refs):,}  ({100*has_title/len(refs):.1f}%)")
    print(f"  cross-resolved s2 : {cross_resolved:>6}  (via paper_db title+year lookup)")
    print()
    print("  appearances per source:")
    for s, n in src_counter.most_common():
        print(f"    {s:<22} {n:>6}")
    print()
    print("  top domains:")
    for d, n in domain_counter.most_common(15):
        print(f"    {d:<22} {n:>6}")
    print()
    print(f"Wrote: {out_jsonl}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {OUT / 'denylist_stats.json'}")


if __name__ == "__main__":
    main()
