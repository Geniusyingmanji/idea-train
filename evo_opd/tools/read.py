"""ReadTool — paper_id → full GenomeCard text.

Loads tool_corpus.jsonl into memory (855 cards, ~3 MB total) and serves O(1)
lookups by paper_id. Returns a formatted block of:
  - title + year
  - abstract excerpt
  - all 6 genome fields

The output text is what the model sees as the tool observation.
"""
from __future__ import annotations

import json
from pathlib import Path


class ReadTool:
    def __init__(self, tool_corpus_path: str | Path = None):
        if tool_corpus_path is None:
            tool_corpus_path = (
                Path("/home/azureuser/workspace-gzy/zyf/idea_train")
                / "data/agentic_v1/tool_corpus.jsonl"
            )
        self.cards: dict[str, dict] = {}
        with Path(tool_corpus_path).open() as f:
            for line in f:
                d = json.loads(line)
                self.cards[d["paper_id"]] = d

    def _resolve_id(self, paper_id: str) -> str | None:
        """Resolve a possibly-malformed paper_id to a real one in the cache.

        Models often drop the 'paper:' prefix or change separators; we try a
        few variants before giving up. Returns the canonical id or None.
        """
        if paper_id in self.cards:
            return paper_id
        # try prefixing with paper:
        candidate = f"paper:{paper_id}"
        if candidate in self.cards:
            return candidate
        # try matching anything where the slug part appears
        # only do this for ids with at least 1 underscore (signal it's a slug)
        slug = paper_id.split(":")[-1]
        if "_" in slug:
            for pid in self.cards:
                if pid.endswith(":" + slug) or pid.endswith(slug):
                    return pid
        return None

    def __contains__(self, paper_id: str) -> bool:
        return self._resolve_id(paper_id) is not None

    def read(self, paper_id: str) -> str:
        canon = self._resolve_id(paper_id)
        c = self.cards.get(canon) if canon else None
        if c is None:
            return f"[error: paper_id {paper_id!r} not in corpus]"
        g = c.get("genome", {})
        parts = [
            f"TITLE: {c.get('title', '?')}",
            f"YEAR: {c.get('year', '?')}",
        ]
        abstract = c.get("abstract", "")
        if abstract:
            parts.append(f"ABSTRACT: {abstract[:800]}")
        for field in ("niche_genome", "mechanism_genome", "delta_genome",
                      "limitation_genome", "observation_genome", "claim_genome"):
            v = g.get(field, "")
            if v:
                parts.append(f"{field.upper()}: {v[:400]}")
        return "\n".join(parts)

    def read_struct(self, paper_id: str) -> dict | None:
        """Structured access for non-text uses."""
        return self.cards.get(paper_id)


if __name__ == "__main__":
    tool = ReadTool()
    print(f"Loaded {len(tool.cards)} cards\n")

    test_id = list(tool.cards.keys())[0]
    print(f"=== read({test_id!r}) ===")
    print(tool.read(test_id))
    print(f"\n=== read(non-existent) ===")
    print(tool.read("paper:foo:9999"))
