"""extract_genome tool — return structured 6-field gene_genome from a paper.

Ports `IdeaEvolving/agent/genome_extract_agent.py` logic, but uses our existing
Azure GPT-5.5 keyless client (`evo_opd.teachers.gpt55_client`) instead of the
upstream's bespoke config. Accepts:
  - paper_id like "oa:W123" → fetch via OpenAlex (with cache)
  - paper_id like "paper:foo:2024" → look up in local 855-card corpus
  - raw text → use directly as title+abstract input

Output: {mechanism_genome, niche_genome, observation_genome, limitation_genome,
         delta_genome, claim_genome} — same 6 fields as our `propose` action.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ..teachers.gpt55_client import TeacherCall, build_client, call_one
from .read import ReadTool
from .web_search import WebSearchTool

# In-memory cache for the lifetime of a process. Each extract is ~5s of GPT-5.5,
# so caching saves a lot during rollouts that re-extract the same parent paper.
_EXTRACT_CACHE: dict[str, dict] = {}

GENOME_EXTRACT_PROMPT = """\
You are a scientific idea genome extractor. Given a paper's title and abstract, \
extract the paper's heritable "idea genome" — the core innovation atoms that could \
be inherited by future work.

Title: {title}
Abstract: {abstract}

Extract exactly one JSON object with these fields (use "" if truly unknown):
{{
  "niche_genome": "ONE sentence: the specific problem/task/domain this paper targets (NO method details)",
  "mechanism_genome": "ONE sentence: the core technical approach — algorithm, architecture, or procedure. Be precise and specific.",
  "observation_genome": "ONE sentence: the key empirical finding or insight from this paper",
  "limitation_genome": "ONE sentence: the main weakness or unresolved gap that could seed future mutations",
  "delta_genome": "ONE sentence: what specifically changed vs the most closely related prior work",
  "claim_genome": "ONE sentence: the primary falsifiable assertion this paper makes"
}}

Rules:
- mechanism_genome should be at "method-name + one-sentence mechanism" granularity
- niche_genome describes WHAT is solved, not HOW
- limitation_genome must be specific to THIS paper, not general field limitations
- delta_genome should reference what changed vs predecessor, not general novelty claims
- Output ONLY valid JSON inside ```json ... ``` fences. No commentary outside.
"""

# 6 canonical genome fields (matches our propose action schema)
CANON_GENOME_FIELDS = [
    "mechanism_genome", "niche_genome", "observation_genome",
    "limitation_genome", "delta_genome", "claim_genome",
]

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_genome_output(text: str) -> dict | None:
    if not text:
        return None
    text = text.strip()
    m = _FENCE_RE.search(text)
    blob = m.group(1) if m else text
    blob = blob.strip()
    try:
        parsed = json.loads(blob)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = blob.find("{")
    end = blob.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(blob[start:end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def _canonicalize(genome: dict) -> dict:
    out = {f: "" for f in CANON_GENOME_FIELDS}
    for k, v in genome.items():
        kk = k.strip().lower()
        if kk in out:
            out[kk] = str(v).strip()
    return out


@dataclass
class GenomeExtractResult:
    paper_id: str
    title: str
    year: int | None
    genome: dict          # 6 canonical fields, all str
    error: str | None = None


class GenomeExtractTool:
    """Resolve paper_id → text → GPT-5.5 → structured 6-field genome."""

    def __init__(self,
                 read_tool: ReadTool | None = None,
                 web_search_tool: WebSearchTool | None = None,
                 client=None):
        self.local_read = read_tool or ReadTool()
        self.web = web_search_tool or WebSearchTool()
        self.client = client
        # lazy init the client only when we actually need it
        # (avoid loading Azure config when smoke-importing)

    def _ensure_client(self):
        if self.client is None:
            self.client = build_client()

    def _resolve_text(self, paper_id_or_text: str) -> tuple[str, str, int | None]:
        """Return (paper_id, title, abstract)."""
        s = paper_id_or_text.strip()
        # raw text fallback: must be longer than 50 chars and not start with id prefix
        if not s.startswith(("paper:", "oa:")) and len(s) > 50:
            # treat as title+abstract dumped together
            title = s.split("\n", 1)[0][:200]
            return ("raw:" + str(hash(s) % (10 ** 9)), title, None)
        # paper:... → local card
        if s.startswith("paper:"):
            card = self.local_read.cards.get(s)
            if card is None:
                # try fuzzy
                canon = self.local_read._resolve_id(s)
                if canon:
                    card = self.local_read.cards.get(canon)
            if card is None:
                return (s, "", None)
            return (s, card.get("title", ""), card.get("year"))  # abstract added below
        # oa:... → openalex cache
        if s.startswith("oa:"):
            work = self.web.fetch_work(s)
            if work is None:
                return (s, "", None)
            from .web_search import _reconstruct_abstract
            title = work.get("title", "")
            year = work.get("publication_year")
            abstract = _reconstruct_abstract(
                work.get("abstract_inverted_index"), max_chars=2000,
            )
            return (s, title, year, abstract) if False else (s, title or "", year)
        # unknown
        return (s, "", None)

    def _get_abstract(self, paper_id: str) -> str:
        s = paper_id
        if s.startswith("oa:"):
            work = self.web.fetch_work(s)
            if work is None:
                return ""
            from .web_search import _reconstruct_abstract
            return _reconstruct_abstract(
                work.get("abstract_inverted_index"), max_chars=2000,
            )
        if s.startswith("paper:"):
            card = self.local_read.cards.get(s) or self.local_read.cards.get(
                self.local_read._resolve_id(s) or ""
            )
            if not card:
                return ""
            return (card.get("abstract") or "")[:2000]
        return ""

    def extract(self, paper_id_or_text: str, *, retries: int = 2,
                max_tokens: int = 600) -> GenomeExtractResult:
        """Main entry: paper_id (or raw text) → structured genome."""
        # cache (use raw input as key)
        if paper_id_or_text in _EXTRACT_CACHE:
            cached = _EXTRACT_CACHE[paper_id_or_text]
            return GenomeExtractResult(**cached)

        paper_id, title, year = self._resolve_text(paper_id_or_text)
        if paper_id.startswith("raw:"):
            # raw text input — use whole as combined input
            abstract = paper_id_or_text
            title_for_prompt = title or "(no title)"
        else:
            abstract = self._get_abstract(paper_id)
            title_for_prompt = title or paper_id

        if not abstract or len(abstract) < 50:
            res = GenomeExtractResult(
                paper_id=paper_id, title=title_for_prompt, year=year,
                genome={f: "" for f in CANON_GENOME_FIELDS},
                error="abstract unavailable or too short",
            )
            _EXTRACT_CACHE[paper_id_or_text] = {
                "paper_id": res.paper_id, "title": res.title, "year": res.year,
                "genome": res.genome, "error": res.error,
            }
            return res

        self._ensure_client()
        user_text = GENOME_EXTRACT_PROMPT.format(
            title=title_for_prompt[:250], abstract=abstract[:2000],
        )
        call = TeacherCall(
            prompt_id=f"extract::{paper_id}",
            messages=[{"role": "user", "content": user_text}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        r = call_one(self.client, call, retries=retries)

        if r.error or not r.content:
            res = GenomeExtractResult(
                paper_id=paper_id, title=title_for_prompt, year=year,
                genome={f: "" for f in CANON_GENOME_FIELDS},
                error=r.error or "empty LLM response",
            )
        else:
            parsed = _parse_genome_output(r.content)
            if parsed is None:
                res = GenomeExtractResult(
                    paper_id=paper_id, title=title_for_prompt, year=year,
                    genome={f: "" for f in CANON_GENOME_FIELDS},
                    error="failed to parse genome JSON",
                )
            else:
                res = GenomeExtractResult(
                    paper_id=paper_id, title=title_for_prompt, year=year,
                    genome=_canonicalize(parsed),
                )

        _EXTRACT_CACHE[paper_id_or_text] = {
            "paper_id": res.paper_id, "title": res.title, "year": res.year,
            "genome": res.genome, "error": res.error,
        }
        return res

    def extract_batch(self, paper_ids: list[str], *, workers: int = 8,
                      ) -> list[GenomeExtractResult]:
        """Parallel batch extract."""
        self._ensure_client()
        out: list[GenomeExtractResult] = [None] * len(paper_ids)  # type: ignore
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.extract, pid): i for i, pid in enumerate(paper_ids)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out


def format_genome_observation(result: GenomeExtractResult) -> str:
    """Format extract result as the tool observation the agent sees."""
    if result.error:
        return f"[error]: extract_genome failed for {result.paper_id}: {result.error}"
    g = result.genome
    lines = [
        f"[result]: extracted genome for {result.paper_id} ({result.title[:80]}, year={result.year}):",
    ]
    for f in CANON_GENOME_FIELDS:
        v = g.get(f, "")
        if v:
            lines.append(f"  {f}: {v[:300]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import time
    tool = GenomeExtractTool()
    # Use a known-cached OpenAlex paper from web_search smoke earlier:
    test_id = "oa:W4311930535"  # perovskite paper
    print(f"=== extract {test_id} ===")
    t0 = time.time()
    res = tool.extract(test_id)
    print(f"  done in {time.time() - t0:.1f}s  error={res.error}")
    for f, v in res.genome.items():
        print(f"  {f}: {v[:200]}")
    print()
    print("=== observation format ===")
    print(format_genome_observation(res))
