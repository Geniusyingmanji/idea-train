"""Propose action parser — terminal action that emits the final gene_genome.

The model's `propose` tool call carries a gene_genome JSON. We parse it leniently
(JSON, code-fenced JSON, or 'Key: value' plain-text fallback), normalize to the
6 canonical fields, and return as a dict for downstream reward computation.

Canonical fields (same as GenomeCard.genome):
  mechanism_genome, niche_genome, observation_genome,
  limitation_genome, delta_genome, claim_genome
"""
from __future__ import annotations

import json
import re


CANON_FIELDS = [
    "mechanism_genome", "niche_genome", "observation_genome",
    "limitation_genome", "delta_genome", "claim_genome",
]

# common alias → canonical
FIELD_ALIASES = {
    "mechanism": "mechanism_genome",
    "method": "mechanism_genome",
    "core_method": "mechanism_genome",
    "core_idea": "mechanism_genome",
    "approach": "mechanism_genome",
    "niche": "niche_genome",
    "problem": "niche_genome",
    "target": "niche_genome",
    "target_problem": "niche_genome",
    "observation": "observation_genome",
    "expected_observation": "observation_genome",
    "expected_outcome": "observation_genome",
    "result": "observation_genome",
    "expected_result": "observation_genome",
    "limitation": "limitation_genome",
    "limitations": "limitation_genome",
    "acknowledged_limitation": "limitation_genome",
    "gap": "limitation_genome",
    "delta": "delta_genome",
    "innovation": "delta_genome",
    "novelty": "delta_genome",
    "novel_contribution": "delta_genome",
    "claim": "claim_genome",
    "hypothesis": "claim_genome",
    "motivation": "claim_genome",
}

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_KV_LINE = re.compile(r"^\s*([A-Za-z][A-Za-z _]+?)\s*[:=]\s*(.+?)\s*$", re.MULTILINE)


def _canonicalize_keys(d: dict) -> dict:
    out: dict[str, str] = {}
    for k, v in d.items():
        if not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
        k_norm = k.strip().lower().replace(" ", "_")
        canon = FIELD_ALIASES.get(k_norm, k_norm)
        if canon in CANON_FIELDS:
            out[canon] = v.strip()
    # fill missing fields with empty string for downstream compat
    for f in CANON_FIELDS:
        out.setdefault(f, "")
    return out


def parse_propose_action(text: str) -> dict | None:
    """Try multiple parsers; return canonicalized gene_genome dict or None."""
    if not text or not text.strip():
        return None
    text = text.strip()

    # 1. raw JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _canonicalize_keys(parsed)
    except Exception:
        pass

    # 2. fenced JSON
    m = _FENCE_RE.search(text)
    if m:
        try:
            parsed = json.loads(m.group(1))
            if isinstance(parsed, dict):
                return _canonicalize_keys(parsed)
        except Exception:
            pass

    # 3. plain "Key: value" lines
    kv = {}
    for m in _KV_LINE.finditer(text):
        k, v = m.group(1), m.group(2)
        kv[k] = v
    if kv:
        return _canonicalize_keys(kv)

    return None


if __name__ == "__main__":
    cases = [
        # raw JSON
        '{"mechanism_genome": "X", "niche_genome": "Y", "claim_genome": "Z"}',
        # fenced JSON with aliases
        '```json\n{"mechanism": "A1", "novelty": "B1"}\n```',
        # plain key:value
        "Mechanism: M plain text\nLimitation: nothing works\nClaim: it'll work",
        # broken / empty
        "i forgot to use the propose tool, sorry",
    ]
    for c in cases:
        out = parse_propose_action(c)
        print(f"INPUT: {c[:60]!r}")
        print(f"  OUT: {out}\n")
