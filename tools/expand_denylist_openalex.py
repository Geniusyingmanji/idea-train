"""Expand denylist v0 → v1 with OpenAlex 1-hop citation closure (threaded).

For each paper in denylist_v0:
  1. Resolve to an OpenAlex work_id (via DOI, S2, or title search).
  2. Fetch referenced_works (papers IT cites).
  3. Optionally fetch citing works.
  4. Add neighbors to v1 with denylist_tier="1hop_referenced" / "1hop_citedby".

Threaded with ThreadPoolExecutor (default 10 workers). OpenAlex polite-pool
limit is 10 req/s sustained, 100K/day. Pyalex is requests-based and thread-safe.

Usage:
  python tools/expand_denylist_openalex.py [--limit N] [--workers 10] [--with-citedby]

Estimated wall-clock at 10 workers / 8359 seeds:
  - resolve + 1-hop refs only: ~30-45 min
  - +citedby: ~2-3 hours
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyalex

OUT = Path("/home/azureuser/workspace-gzy/zyf/idea_train/denylist")
pyalex.config.email = "hokind@andrew.cmu.edu"  # polite pool
pyalex.config.max_retries = 3
pyalex.config.retry_backoff_factor = 0.5

_PRINT_LOCK = threading.Lock()


def norm_title(t: str | None) -> str:
    if not t:
        return ""
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def resolve_one(rec: dict, with_citedby: bool) -> dict:
    """Worker: resolve a single seed and collect 1-hop neighbors."""
    out = {"seed_key": None, "openalex_id": None, "method": "failed",
           "references": [], "citedby": [], "error": None}
    out["seed_key"] = (f"s2:{rec['s2_id']}" if rec.get("s2_id")
                       else f"ty:{norm_title(rec.get('title'))}|{rec.get('year')}")

    # 1. resolve to OpenAlex ID
    if rec.get("arxiv_id"):
        try:
            w = pyalex.Works().filter(ids={"arxiv": rec["arxiv_id"]}).get(per_page=1)
            if w:
                out["openalex_id"], out["method"] = w[0]["id"], "via_arxiv"
        except Exception:
            pass

    if not out["openalex_id"] and rec.get("title"):
        try:
            results = pyalex.Works().search(rec["title"]).get(per_page=3)
            target = norm_title(rec["title"])
            tgt_year = rec.get("year")
            chosen = None
            for r in results:
                if norm_title(r.get("title")) == target:
                    if not tgt_year or r.get("publication_year") == tgt_year:
                        chosen = (r["id"], "via_title_year")
                        break
            if not chosen:
                for r in results:
                    if norm_title(r.get("title")) == target:
                        chosen = (r["id"], "via_title_only")
                        break
            if chosen:
                out["openalex_id"], out["method"] = chosen
        except Exception as e:
            out["error"] = f"search: {str(e)[:80]}"

    if not out["openalex_id"]:
        return out

    # 2. fetch references (free — included in Work record)
    try:
        w = pyalex.Works()[out["openalex_id"]]
        out["references"] = w.get("referenced_works") or []
    except Exception as e:
        out["error"] = f"refs: {str(e)[:80]}"

    # 3. optional citedby
    if with_citedby and out["openalex_id"]:
        try:
            cb = (
                pyalex.Works()
                .filter(cites=out["openalex_id"])
                .select(["id"])
                .get(per_page=200)
            )
            out["citedby"] = [c["id"] for c in cb]
        except Exception as e:
            out["error"] = (out["error"] or "") + f" citedby: {str(e)[:80]}"

    return out


def hydrate_one(oa_id: str) -> dict | None:
    """Worker: hydrate a neighbor record with title/year/doi/venue."""
    try:
        w = pyalex.Works()[oa_id]
        venue = (w.get("primary_location") or {}).get("source")
        return {
            "openalex_id": oa_id,
            "title": w.get("title"),
            "year": w.get("publication_year"),
            "doi": w.get("doi"),
            "venue": (venue or {}).get("display_name"),
        }
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="cap seeds processed")
    ap.add_argument("--workers", type=int, default=10, help="threads")
    ap.add_argument("--with-citedby", action="store_true")
    ap.add_argument("--skip-hydrate", action="store_true",
                    help="don't fetch title/year for neighbors (faster but less useful)")
    args = ap.parse_args()

    v0_path = OUT / "denylist_v0.jsonl"
    print(f"[1/4] Loading seeds from {v0_path}")
    seeds: list[dict] = []
    with v0_path.open() as f:
        for line in f:
            seeds.append(json.loads(line))
    if args.limit:
        seeds = seeds[: args.limit]
    print(f"  {len(seeds):,} seeds | workers={args.workers} | with_citedby={args.with_citedby}")

    # --- Phase 1: resolve + 1-hop neighbor collection (parallel) ----------
    print("[2/4] Resolving + pulling 1-hop neighbors")
    resolutions: list[dict] = []
    res_method: Counter = Counter()
    n_done = 0
    n_total = len(seeds)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(resolve_one, s, args.with_citedby): s for s in seeds}
        for fut in as_completed(futs):
            r = fut.result()
            resolutions.append(r)
            res_method[r["method"]] += 1
            n_done += 1
            if n_done % 100 == 0:
                el = time.time() - t0
                rate = n_done / el
                eta = (n_total - n_done) / max(rate, 0.01)
                with _PRINT_LOCK:
                    print(f"  resolved {n_done:>5}/{n_total:,} | "
                          f"rate {rate:.1f}/s | ETA {eta/60:.1f}min | "
                          f"methods={dict(res_method)}")

    # write resolution log
    log_path = OUT / "openalex_resolution_log.jsonl"
    with log_path.open("w") as f:
        for r in resolutions:
            f.write(json.dumps(r) + "\n")
    n_resolved = sum(1 for r in resolutions if r["openalex_id"])
    print(f"  resolved {n_resolved}/{n_total} ({100*n_resolved/n_total:.1f}%) in {(time.time()-t0)/60:.1f}min")

    # collect unique neighbors
    neighbors: dict[str, dict] = {}  # oa_id -> {source, from_seed}
    for r in resolutions:
        if not r["openalex_id"]:
            continue
        seed_oa = r["openalex_id"]
        for nid in r["references"]:
            neighbors.setdefault(nid, {"openalex_id": nid, "source": "openalex_referenced",
                                        "from_seed": seed_oa})
        for nid in r["citedby"]:
            neighbors.setdefault(nid, {"openalex_id": nid, "source": "openalex_citedby",
                                        "from_seed": seed_oa})
    print(f"  unique neighbors: {len(neighbors):,}")

    # --- Phase 2: hydrate neighbors (parallel, optional) -------------------
    if not args.skip_hydrate:
        print("[3/4] Hydrating neighbor metadata")
        t1 = time.time()
        n_hyd = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(hydrate_one, nid): nid for nid in neighbors}
            for fut in as_completed(futs):
                meta = fut.result()
                if meta:
                    neighbors[meta["openalex_id"]].update(meta)
                    n_hyd += 1
                if n_hyd % 500 == 0 and n_hyd > 0:
                    el = time.time() - t1
                    rate = n_hyd / el
                    eta = (len(neighbors) - n_hyd) / max(rate, 0.01)
                    with _PRINT_LOCK:
                        print(f"  hydrated {n_hyd:>5}/{len(neighbors):,} | rate {rate:.1f}/s | ETA {eta/60:.1f}min")
        print(f"  hydrated {n_hyd}/{len(neighbors)}")
    else:
        print("[3/4] Skipping neighbor hydration (--skip-hydrate)")

    # --- Phase 3: merge & write v1 ----------------------------------------
    print("[4/4] Writing denylist v1")
    v1: dict[str, dict] = {}
    res_by_key = {r["seed_key"]: r for r in resolutions}
    for s in seeds:
        key = (f"s2:{s['s2_id']}" if s.get("s2_id")
               else f"ty:{norm_title(s.get('title'))}|{s.get('year')}")
        rec = dict(s)
        rec["denylist_tier"] = "seed"
        rec["openalex_id"] = res_by_key.get(key, {}).get("openalex_id")
        v1[key] = rec
    for nid, n in neighbors.items():
        key = f"openalex:{nid}"
        if key in v1:
            continue
        rec = dict(n)
        rec["denylist_tier"] = ("1hop_referenced" if n["source"] == "openalex_referenced"
                                 else "1hop_citedby")
        v1[key] = rec

    v1_jsonl = OUT / "denylist_v1.jsonl"
    v1_csv = OUT / "denylist_v1.csv"
    v1_stats = OUT / "denylist_v1_stats.json"
    with v1_jsonl.open("w") as f:
        for r in v1.values():
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    csv_fields = ["denylist_tier", "openalex_id", "s2_id", "arxiv_id", "doi",
                  "title", "year", "venue", "domain", "sources"]
    with v1_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in v1.values():
            row = {k: r.get(k, "") for k in csv_fields}
            if isinstance(row.get("sources"), list):
                row["sources"] = ";".join(row["sources"])
            w.writerow(row)

    tier = Counter(r["denylist_tier"] for r in v1.values())
    stats = {
        "v1_total": len(v1),
        "v0_seeds": len(seeds),
        "openalex_resolved": n_resolved,
        "openalex_resolution_methods": dict(res_method),
        "neighbors_added": len(neighbors),
        "neighbors_hydrated": n_hyd if not args.skip_hydrate else 0,
        "denylist_tiers": dict(tier),
        "elapsed_seconds_total": round(time.time() - t0, 1),
        "workers": args.workers,
    }
    with v1_stats.open("w") as f:
        json.dump(stats, f, indent=2)

    print()
    print("=== v1 stats ===")
    print(json.dumps(stats, indent=2))
    print(f"Wrote: {v1_jsonl}")


if __name__ == "__main__":
    main()
