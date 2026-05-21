"""Build a real-citation edge graph over the GeneTrace v0.1 safe pool.

For each card in cards.jsonl:
  1. Resolve to an OpenAlex Work ID via title+year (free public API; polite rate).
  2. Fetch its references and citers.
  3. Emit (p, q) edges where both p and q resolve to cards in cards.jsonl AND
     the citation actually goes p->q (q's references contains p).

Output:
  data/genetrace_v0_1/raw_edges_citation.jsonl

Each record is an UNLABELED edge (no dynamics yet — GPT-5.5 labeling is a
separate stage, gated by API availability). These edges become the input
for v0.2 dynamics annotation once we have a working teacher.

Polite rate: OpenAlex asks for a mailto identifier. We pass one via the
`User-Agent` header. Default workers=8 (well below their 10 req/s limit).

Resumable: maintains an on-disk cache of OpenAlex resolutions and per-work
reference lists so re-runs do not re-hit the API.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

ROOT = Path("/home/azureuser/workspace-gzy/zyf/idea_train")
CARDS = ROOT / "data/genetrace_v0_1/cards.jsonl"
OUT = ROOT / "data/genetrace_v0_1/raw_edges_citation.jsonl"

CACHE_DIR = ROOT / "data/openalex_cache"
RESOLVE_CACHE = CACHE_DIR / "resolve.jsonl"        # card_id -> openalex_id
REFS_CACHE = CACHE_DIR / "refs.jsonl"              # openalex_id -> [refs]

OPENALEX = "https://api.openalex.org"
UA = "GeneTrace/0.1 (mailto:hokind@andrew.cmu.edu)"


def http_get_json(url: str, retries: int = 3) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    last_err = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            if e.code == 429:                               # rate limit
                time.sleep(2 ** i)
                continue
            return None
        except Exception as e:                              # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(0.5 * (i + 1))
    return None


def load_cache(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
                key = r.pop("_key")
                out[key] = r
            except (json.JSONDecodeError, KeyError):
                continue
    return out


def append_cache(path: Path, key: str, val: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"_key": key, **val}
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def title_year_from_card(card: dict) -> tuple[str, int | None]:
    title = (card.get("title") or "").strip()
    year = card.get("year")
    if isinstance(year, str) and year.isdigit():
        year = int(year)
    return title, year


def resolve_openalex(title: str, year: int | None) -> str | None:
    """Title+year query against OpenAlex /works."""
    if not title:
        return None
    q = urllib.parse.quote(title)
    filt = []
    if year:
        # OpenAlex filter supports a +/- 1 year range
        filt.append(f"from_publication_date:{year - 1}-01-01,to_publication_date:{year + 1}-12-31")
    filt_str = "&filter=" + ",".join(filt) if filt else ""
    url = f"{OPENALEX}/works?search={q}&per-page=1{filt_str}"
    js = http_get_json(url)
    if not js or not js.get("results"):
        return None
    return js["results"][0]["id"]                            # full URL like https://openalex.org/W...


def fetch_refs(oa_id: str) -> list[str] | None:
    """Returns the referenced_works list for the given OA work."""
    if not oa_id.startswith("http"):
        oa_id = f"https://openalex.org/{oa_id}"
    short = oa_id.rsplit("/", 1)[-1]
    js = http_get_json(f"{OPENALEX}/works/{short}?select=referenced_works")
    if not js:
        return None
    return js.get("referenced_works") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="Only resolve/fetch the first N cards (smoke).")
    args = ap.parse_args()

    cards = [json.loads(l) for l in CARDS.open()]
    if args.limit:
        cards = cards[: args.limit]
    print(f"loaded {len(cards)} cards")

    resolve_cache = load_cache(RESOLVE_CACHE)
    refs_cache = load_cache(REFS_CACHE)
    print(f"  cache: {len(resolve_cache)} resolved, {len(refs_cache)} refs fetched")

    # --- Step 1: resolve each card -> openalex id
    to_resolve = [c for c in cards if c["card_id"] not in resolve_cache]
    print(f"\n[1/3] resolving {len(to_resolve)} cards to OpenAlex Work IDs")
    t0 = time.time()
    done = 0

    def _resolve(card):
        title, year = title_year_from_card(card)
        oa = resolve_openalex(title, year)
        return card["card_id"], {"openalex_id": oa, "title": title, "year": year}

    if to_resolve:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_resolve, c) for c in to_resolve]
            for fut in as_completed(futs):
                cid, info = fut.result()
                resolve_cache[cid] = info
                append_cache(RESOLVE_CACHE, cid, info)
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{len(to_resolve)}  "
                          f"({(time.time() - t0):.1f}s)", flush=True)
    n_resolved = sum(1 for v in resolve_cache.values() if v.get("openalex_id"))
    print(f"  resolved: {n_resolved}/{len(cards)} cards (rate {n_resolved/max(len(cards),1):.0%})")

    # --- Step 2: fetch references for each resolved OA id
    oa_ids_needed = {v["openalex_id"] for v in resolve_cache.values() if v.get("openalex_id")}
    to_fetch = [oa for oa in oa_ids_needed if oa not in refs_cache]
    print(f"\n[2/3] fetching references for {len(to_fetch)} works")
    t0 = time.time()
    done = 0
    if to_fetch:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fetch_refs, oa): oa for oa in to_fetch}
            for fut in as_completed(futs):
                oa = futs[fut]
                refs = fut.result()
                rec = {"refs": refs if refs is not None else []}
                refs_cache[oa] = rec
                append_cache(REFS_CACHE, oa, rec)
                done += 1
                if done % 50 == 0:
                    print(f"    {done}/{len(to_fetch)}  "
                          f"({(time.time() - t0):.1f}s)", flush=True)

    # --- Step 3: build edges where both ends are in our card set
    print(f"\n[3/3] cross-referencing edges (both ends in safe pool)")
    oa_to_card = {v["openalex_id"]: cid
                   for cid, v in resolve_cache.items() if v.get("openalex_id")}
    edges = []
    for q_oa, q_info in refs_cache.items():
        q_cid = oa_to_card.get(q_oa)
        if q_cid is None:
            continue
        for p_oa in (q_info.get("refs") or []):
            p_cid = oa_to_card.get(p_oa)
            if p_cid is None:
                continue
            edges.append({
                "edge_id":      f"edge::{p_cid.replace('card::','')}::{q_cid.replace('card::','')}",
                "p_card_id":    p_cid,
                "q_card_id":    q_cid,
                "p_paper_id":   p_cid.replace("card::", ""),
                "q_paper_id":   q_cid.replace("card::", ""),
                "p_openalex":   p_oa,
                "q_openalex":   q_oa,
                "source":       "openalex_referenced_works",
                "dynamics":     None,             # to-be-labeled when teacher available
                "provenance": {
                    "version":     "genetrace-v0.1",
                    "fetched_ts":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            })
    print(f"  {len(edges)} unlabeled real-citation edges (both ends in safe pool)")

    with OUT.open("w") as f:
        for e in edges:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  wrote {OUT}")

    # quick stats
    if edges:
        out_deg = Counter(e["p_paper_id"] for e in edges)
        in_deg = Counter(e["q_paper_id"] for e in edges)
        print(f"  unique source papers (cited): {len(out_deg)}")
        print(f"  unique target papers (citers): {len(in_deg)}")
        print(f"  top 5 most-cited (in our pool): {out_deg.most_common(5)}")


if __name__ == "__main__":
    main()
