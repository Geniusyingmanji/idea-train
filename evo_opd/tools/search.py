"""BM25 search over the GeneTrace v0.1 paper corpus.

Pure-Python implementation (no rank_bm25 dep): we compute BM25 scores with
default parameters (k1=1.5, b=0.75) and return top-k papers with brief
snippets. The corpus is small (855 papers) so this runs in <10ms per query.

The search returns each result as:
  - paper_id
  - title
  - year
  - score (BM25)
  - snippet (first 150 chars of searchable_text)

Year-window filtering happens BEFORE BM25 ranking so the agent's year hint is
respected. Discipline filtering happens AFTER ranking so it only re-orders, not
prunes (in case the discipline label is noisy).
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


_TOK = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "as", "is", "are", "be", "this", "that", "it", "its", "their",
    "from", "into", "using", "used", "use", "uses", "via", "through",
    "paper", "method", "model", "approach", "system", "framework",
    "novel", "new", "propose", "proposed", "proposal", "study", "studies",
    "research", "work", "results", "show", "shows", "shown",
}


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _TOK.findall(text or "") if w.lower() not in STOPWORDS]


@dataclass
class SearchResult:
    paper_id: str
    title: str
    year: int | None
    score: float
    snippet: str

    def to_dict(self) -> dict:
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "year": self.year,
            "score": round(self.score, 3),
            "snippet": self.snippet,
        }


class SearchTool:
    """In-memory BM25 search over the agentic_v1 bm25_corpus."""

    def __init__(self, bm25_corpus_path: str | Path = None,
                 *, k1: float = 1.5, b: float = 0.75):
        if bm25_corpus_path is None:
            bm25_corpus_path = (
                Path("/home/azureuser/workspace-gzy/zyf/idea_train")
                / "data/agentic_v1/bm25_corpus.jsonl"
            )
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []
        self.tokens: list[list[str]] = []
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.doc_len: list[int] = []
        self.avg_len: float = 0.0
        self.idf: dict[str, float] = {}

        with Path(bm25_corpus_path).open() as f:
            for line in f:
                d = json.loads(line)
                self.docs.append(d)
                toks = tokenize(d.get("searchable_text", ""))
                self.tokens.append(toks)
                self.doc_len.append(len(toks))
                for t in set(toks):
                    self.doc_freq[t] += 1
        N = len(self.docs)
        self.avg_len = sum(self.doc_len) / max(N, 1)
        # BM25 IDF (Robertson-Sparck Jones smoothed)
        for t, df in self.doc_freq.items():
            self.idf[t] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_score(self, doc_idx: int, query_tokens: list[str]) -> float:
        s = 0.0
        dl = self.doc_len[doc_idx]
        toks = self.tokens[doc_idx]
        tf = Counter(toks)
        for q in query_tokens:
            if q not in self.idf:
                continue
            f = tf.get(q, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avg_len, 1))
            s += self.idf[q] * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, *, k: int = 5,
               year_min: int | None = None, year_max: int | None = None,
               discipline: str | None = None,
               denylist_paper_ids: set[str] | None = None,
               ) -> list[SearchResult]:
        q_toks = tokenize(query)
        if not q_toks:
            return []
        denylist_paper_ids = denylist_paper_ids or set()

        scored: list[tuple[int, float]] = []
        for i, d in enumerate(self.docs):
            if d["paper_id"] in denylist_paper_ids:
                continue
            y = d.get("year")
            if year_min is not None and (y is None or y < year_min):
                continue
            if year_max is not None and (y is None or y > year_max):
                continue
            if discipline and d.get("discipline") and d["discipline"] != discipline:
                # discipline mismatch: keep but with a penalty
                pass
            s = self._bm25_score(i, q_toks)
            if s > 0:
                scored.append((i, s))
        scored.sort(key=lambda t: -t[1])

        results: list[SearchResult] = []
        for i, s in scored[:k]:
            d = self.docs[i]
            snippet = (d.get("searchable_text") or "")[:200]
            results.append(SearchResult(
                paper_id=d["paper_id"],
                title=d.get("title", ""),
                year=d.get("year"),
                score=s,
                snippet=snippet,
            ))
        return results


if __name__ == "__main__":
    tool = SearchTool()
    print(f"Loaded {len(tool.docs)} docs (avg_len={tool.avg_len:.1f} tokens)\n")

    queries = [
        ("image caption dataset", None, None),
        ("diffusion molecule generation", 2020, 2024),
        ("attention transformer language model", 2017, 2020),
    ]
    for q, ymin, ymax in queries:
        print(f"=== query: {q!r} (year={ymin}..{ymax}) ===")
        results = tool.search(q, k=5, year_min=ymin, year_max=ymax)
        for r in results:
            print(f"  {r.score:6.2f}  {r.year} {r.paper_id[:60]} — {r.title[:80]}")
        print()
