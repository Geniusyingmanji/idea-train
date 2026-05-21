"""novelty_check tool — find papers nearest to a proposed mechanism.

Helps the agent detect:
  - "redundant" (sim > 0.85 against an existing paper): proposal IS that paper
  - "disconnected" (top-sim < 0.15 across all results): proposal is isolated
                                                       from any visible lineage
  - "healthy novelty" (0.30 < sim < 0.70): builds on prior work without copying

Implementation: search OpenAlex with the proposed mechanism as query, then
compute lexical_similarity between proposed text and each result's title +
abstract. Returns top 3 + a verdict.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .web_search import WebSearchTool, _reconstruct_abstract
from .search import SearchTool
from ..structural import lexical_similarity


@dataclass
class NoveltyResult:
    query_text: str
    n_searched: int
    nearest: list[dict] = field(default_factory=list)  # [{paper_id, title, year, sim}]
    max_sim: float = 0.0
    verdict: str = ""              # "redundant" | "disconnected" | "healthy" | "weak_signal"
    diagnostics: str = ""
    error: str | None = None


class NoveltyCheckTool:
    """Combine web_search + lexical similarity to assess novelty of a mechanism."""

    REDUNDANT_THRESHOLD = 0.65     # sim > this → redundant
    DISCONNECTED_THRESHOLD = 0.10  # max sim < this → disconnected
    HEALTHY_LOW = 0.15
    HEALTHY_HIGH = 0.55

    def __init__(self,
                 web_search_tool: WebSearchTool | None = None,
                 local_search_tool: SearchTool | None = None,
                 use_local: bool = False):
        self.web = web_search_tool or WebSearchTool()
        self.local = local_search_tool if use_local else None
        self.use_local = use_local

    def check(self, mechanism_text: str, *, k: int = 8,
              year_min: int | None = None, year_max: int | None = None,
              ) -> NoveltyResult:
        if not mechanism_text or len(mechanism_text.strip()) < 15:
            return NoveltyResult(
                query_text=mechanism_text[:200],
                n_searched=0, nearest=[], max_sim=0.0,
                verdict="weak_signal",
                diagnostics="mechanism text too short for meaningful novelty check",
                error="short_input",
            )

        # Search the web for similar mechanisms
        try:
            results = self.web.search(
                mechanism_text[:200], k=k,
                year_min=year_min, year_max=year_max,
            )
        except Exception as e:
            return NoveltyResult(
                query_text=mechanism_text[:200], n_searched=0,
                error=f"web_search_failed: {type(e).__name__}: {e}",
                verdict="weak_signal", diagnostics="search failed",
            )

        nearest: list[dict] = []
        for r in results[:k]:
            # fetch full text from cache (will be there from search)
            work = self.web.get_cached_work(r.paper_id) or {}
            other = " ".join([
                r.title,
                _reconstruct_abstract(work.get("abstract_inverted_index"), max_chars=600),
            ])
            sim = lexical_similarity(mechanism_text, other)
            nearest.append({
                "paper_id": r.paper_id,
                "title": r.title,
                "year": r.year,
                "sim": round(sim, 4),
            })

        nearest.sort(key=lambda x: -x["sim"])
        max_sim = nearest[0]["sim"] if nearest else 0.0

        # verdict
        if max_sim > self.REDUNDANT_THRESHOLD:
            verdict = "redundant"
            diag = (
                f"⚠ proposed mechanism is very similar (sim={max_sim:.2f}) to "
                f"{nearest[0]['title'][:100]!r} ({nearest[0]['paper_id']}). "
                "Consider differentiating more clearly or proposing something genuinely new."
            )
        elif max_sim < self.DISCONNECTED_THRESHOLD:
            verdict = "disconnected"
            diag = (
                f"⚠ no closely related papers found (top sim={max_sim:.2f}). "
                "Proposal may be isolated from any visible lineage; consider grounding "
                "in a parent paper or revisiting topic relevance."
            )
        elif self.HEALTHY_LOW <= max_sim <= self.HEALTHY_HIGH:
            verdict = "healthy"
            diag = (
                f"✓ moderate novelty (top sim={max_sim:.2f} with "
                f"{nearest[0]['title'][:60]!r}). Proposal builds on related work "
                "without copying it."
            )
        else:
            verdict = "weak_signal"
            diag = (
                f"ℹ top similarity = {max_sim:.2f} — between healthy and redundant. "
                "Proposal is similar to existing work but not a direct match."
            )

        return NoveltyResult(
            query_text=mechanism_text[:200],
            n_searched=len(nearest),
            nearest=nearest[:3],          # only show top 3 to agent
            max_sim=max_sim,
            verdict=verdict,
            diagnostics=diag,
        )


def format_novelty_observation(result: NoveltyResult) -> str:
    if result.error:
        return f"[error]: novelty_check failed: {result.error}"
    lines = [
        f"[result]: novelty verdict = {result.verdict}",
        f"  max_sim = {result.max_sim:.3f}",
        result.diagnostics,
        "  top 3 nearest papers:",
    ]
    for r in result.nearest:
        lines.append(f"    - {r['paper_id']} ({r['year']}): {r['title'][:90]}  sim={r['sim']:.3f}")
    return "\n".join(lines)


if __name__ == "__main__":
    tool = NoveltyCheckTool()
    cases = [
        ("Physics-aware diffusion model for drug discovery with learned energy head from MD trajectories", 2020, 2025),
        ("LLM with chain-of-thought prompting for arithmetic reasoning", 2022, 2025),
        ("Quantum computing for distributed databases using blockchain consensus", None, None),
    ]
    for q, ymin, ymax in cases:
        print(f"\n=== novelty_check: {q[:80]} ===")
        res = tool.check(q, year_min=ymin, year_max=ymax)
        print(format_novelty_observation(res))
