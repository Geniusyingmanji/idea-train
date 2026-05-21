"""OpenAlex web search backend for agentic-OPD.

Drop-in replacement for the local BM25 SearchTool: same `.search(query, ...)`
interface returns `SearchResult` objects with `paper_id, title, year, snippet`.

Key choices:
  - paper_id namespace: "oa:W<openalex_work_id>" so it's distinguishable from
    "paper:..." local IDs.
  - Caches every OpenAlex Work response to disk (~5MB per 1000 papers) so
    repeated queries during RL don't hammer the API.
  - Caches search results by (query, year_min, year_max) for ~1 hour.
  - Falls back gracefully on rate limit (429) by returning empty.

For READS: see `web_read.py` — also OpenAlex-backed; same disk cache.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests

from .search import SearchResult  # reuse the dataclass

OPENALEX_BASE = "https://api.openalex.org"
CACHE_DIR = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/openalex_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# OpenAlex requires a UA / contact email for polite usage (gets faster lane)
USER_AGENT = "evo-opd-agentic/0.1 (mailto:research@local)"


def _query_hash(query: str, year_min, year_max) -> str:
    s = f"{query}|{year_min}|{year_max}"
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _reconstruct_abstract(inv_idx: dict | None, max_chars: int = 800) -> str:
    """OpenAlex returns abstract_inverted_index = {word: [positions]}.
    Rebuild a string."""
    if not inv_idx or not isinstance(inv_idx, dict):
        return ""
    pos_to_word: dict[int, str] = {}
    for w, positions in inv_idx.items():
        if not isinstance(positions, list):
            continue
        for p in positions:
            pos_to_word[p] = w
    if not pos_to_word:
        return ""
    out = " ".join(pos_to_word[i] for i in sorted(pos_to_word.keys()))
    return out[:max_chars]


def _openalex_id_to_local(oa_id: str) -> str:
    """Convert https://openalex.org/W12345 → oa:W12345 ."""
    if oa_id.startswith("https://openalex.org/"):
        oa_id = oa_id.removeprefix("https://openalex.org/")
    return f"oa:{oa_id}"


class WebSearchTool:
    """OpenAlex-backed search with disk caching."""

    def __init__(self, cache_dir: Path = CACHE_DIR, *, timeout: float = 8.0,
                 cache_ttl_s: int = 24 * 3600):
        self.cache_dir = cache_dir
        self.search_cache = cache_dir / "search"
        self.work_cache = cache_dir / "works"
        self.search_cache.mkdir(parents=True, exist_ok=True)
        self.work_cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.cache_ttl_s = cache_ttl_s
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})

    def _load_search_cache(self, key: str) -> list[SearchResult] | None:
        p = self.search_cache / f"{key}.json"
        if not p.exists():
            return None
        mtime = p.stat().st_mtime
        if time.time() - mtime > self.cache_ttl_s:
            return None
        try:
            raw = json.loads(p.read_text())
            return [SearchResult(**r) for r in raw]
        except Exception:
            return None

    def _save_search_cache(self, key: str, results: list[SearchResult]) -> None:
        p = self.search_cache / f"{key}.json"
        p.write_text(json.dumps([r.to_dict() for r in results], ensure_ascii=False))

    def _cache_work(self, oa_id: str, work: dict) -> None:
        p = self.work_cache / f"{oa_id.replace('oa:', '')}.json"
        try:
            p.write_text(json.dumps(work, ensure_ascii=False))
        except Exception:
            pass

    def get_cached_work(self, oa_id: str) -> dict | None:
        p = self.work_cache / f"{oa_id.replace('oa:', '')}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception:
            return None

    def fetch_work(self, oa_id: str) -> dict | None:
        """Fetch + cache one work by OpenAlex ID (handle 'oa:W123' or 'W123')."""
        bare = oa_id.removeprefix("oa:")
        cached = self.get_cached_work(oa_id)
        if cached is not None:
            return cached
        try:
            r = self._session.get(f"{OPENALEX_BASE}/works/{bare}", timeout=self.timeout)
            if r.status_code == 200:
                work = r.json()
                self._cache_work(oa_id, work)
                return work
        except Exception:
            pass
        return None

    def search(self, query: str, *, k: int = 5,
               year_min: int | None = None, year_max: int | None = None,
               discipline: str | None = None,
               denylist_paper_ids: set[str] | None = None,
               ) -> list[SearchResult]:
        if not query or not query.strip():
            return []
        key = _query_hash(query, year_min, year_max)
        cached = self._load_search_cache(key)
        if cached is not None:
            # apply denylist + k cap on cached results
            if denylist_paper_ids:
                cached = [r for r in cached if r.paper_id not in denylist_paper_ids]
            return cached[:k]

        # build filter
        filters = ["has_abstract:true"]
        if year_min is not None:
            filters.append(f"from_publication_date:{int(year_min)}-01-01")
        if year_max is not None:
            filters.append(f"to_publication_date:{int(year_max)}-12-31")
        params = {
            "search": query,
            "per_page": min(max(k * 2, 5), 25),  # over-fetch for denylist
            "filter": ",".join(filters),
        }
        try:
            r = self._session.get(f"{OPENALEX_BASE}/works", params=params, timeout=self.timeout)
            if r.status_code != 200:
                return []
            data = r.json()
        except Exception:
            return []

        results: list[SearchResult] = []
        denylist_paper_ids = denylist_paper_ids or set()
        for w in data.get("results", []):
            oa_id_full = w.get("id", "")
            paper_id = _openalex_id_to_local(oa_id_full)
            if paper_id in denylist_paper_ids:
                continue
            title = (w.get("title") or "")[:200]
            year = w.get("publication_year")
            abstract = _reconstruct_abstract(w.get("abstract_inverted_index"), max_chars=300)
            snippet = (abstract or "")[:240] if abstract else (w.get("doi") or "")[:240]
            relevance = float(w.get("relevance_score") or 0.0)
            # cache the work too (for read)
            self._cache_work(paper_id, w)
            results.append(SearchResult(
                paper_id=paper_id,
                title=title,
                year=year,
                score=relevance,
                snippet=snippet,
            ))
            if len(results) >= k:
                break

        self._save_search_cache(key, results)
        return results[:k]


if __name__ == "__main__":
    tool = WebSearchTool()
    queries = [
        ("perovskite solar cell stability", 2020, 2025),
        ("conditional diffusion molecule generation", 2022, 2025),
        ("large language model reasoning", 2023, 2024),
    ]
    for q, ymin, ymax in queries:
        print(f"\n=== query: {q!r} ({ymin}..{ymax}) ===")
        t0 = time.time()
        results = tool.search(q, k=5, year_min=ymin, year_max=ymax)
        print(f"  {len(results)} results in {time.time() - t0:.2f}s")
        for r in results:
            print(f"  {r.score:5.2f}  {r.year} {r.paper_id[:30]}  — {r.title[:80]}")
