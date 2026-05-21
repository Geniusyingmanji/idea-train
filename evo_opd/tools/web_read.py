"""Read OpenAlex works fetched by WebSearchTool.

Returns a formatted text block (same shape as ReadTool's output) so the agent
can consume it identically. For unknown IDs it returns an error.
"""
from __future__ import annotations

import json
from pathlib import Path

from .read import ReadTool                    # local card backend (fallback)
from .web_search import WebSearchTool, _reconstruct_abstract


class HybridReadTool:
    """First try the OpenAlex work cache (any oa:W... id, fetch if missing),
    then fall back to the local 855-card ReadTool (paper:... ids)."""

    def __init__(self,
                 web_search_tool: WebSearchTool | None = None,
                 local_read_tool: ReadTool | None = None):
        self.web = web_search_tool or WebSearchTool()
        self.local = local_read_tool or ReadTool()

    def __contains__(self, paper_id: str) -> bool:
        if paper_id.startswith("oa:"):
            # may need to fetch but presume yes
            return True
        return paper_id in self.local

    def _resolve_id(self, paper_id: str) -> str | None:
        if paper_id.startswith("oa:"):
            return paper_id
        return self.local._resolve_id(paper_id)

    def _read_oa(self, oa_id: str) -> str:
        work = self.web.fetch_work(oa_id)
        if work is None:
            return f"[error: openalex id {oa_id!r} could not be fetched]"
        title = work.get("title") or "?"
        year = work.get("publication_year", "?")
        abstract = _reconstruct_abstract(
            work.get("abstract_inverted_index"), max_chars=1200,
        )
        # extract authors and venue
        authors = ", ".join(
            (a.get("author") or {}).get("display_name", "?")
            for a in (work.get("authorships") or [])[:5]
        ) or "?"
        venue_obj = work.get("primary_location") or {}
        venue = (venue_obj.get("source") or {}).get("display_name", "?") if venue_obj else "?"
        cited_by_count = work.get("cited_by_count", 0)
        bits = [
            f"TITLE: {title}",
            f"YEAR: {year}",
            f"AUTHORS: {authors}",
            f"VENUE: {venue}",
            f"CITED BY: {cited_by_count}",
        ]
        if abstract:
            bits.append(f"ABSTRACT: {abstract}")
        return "\n".join(bits)

    def read(self, paper_id: str) -> str:
        canon = self._resolve_id(paper_id)
        if canon is None:
            return f"[error: paper_id {paper_id!r} not in corpus]"
        if canon.startswith("oa:"):
            return self._read_oa(canon)
        return self.local.read(canon)

    def read_struct(self, paper_id: str) -> dict | None:
        canon = self._resolve_id(paper_id)
        if canon is None:
            return None
        if canon.startswith("oa:"):
            return self.web.fetch_work(canon)
        return self.local.read_struct(canon)


if __name__ == "__main__":
    tool = HybridReadTool()
    # known local
    pid_local = list(tool.local.cards.keys())[0]
    print(f"=== local read: {pid_local} ===")
    print(tool.read(pid_local)[:600])
    print()
    # known web (perovskite from earlier search)
    print("=== web read: oa:W4311930535 (perovskite paper) ===")
    print(tool.read("oa:W4311930535")[:800])
