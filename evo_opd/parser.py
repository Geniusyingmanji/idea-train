"""Parse a student rollout into typed regions for field-gated reverse-KL.

Output:
  ParsedRollout
    .raw_text:         the input string
    .parsed_json:      dict if JSON parsed successfully, else None
    .regions:          list[Region] — non-overlapping character spans with a φ tag
    .schema_valid:     bool — JSON-parseable AND matches expected task schema
    .gold_values:      dict — extracted gold-shape answers (for verifier)
    .evidence_spans:   list[str] — raw quoted strings the model claims are from source

Token-role tagging happens via `tokenize_with_roles(tokenizer)` which aligns
character regions to token offsets (requires HF fast tokenizer w/ offsets).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .schemas import (
    GENE_FIELDS,
    TASK_SCHEMAS,
    TaskSchema,
    canonical_dynamics,
    canonical_field,
    get_schema,
)

PhiTag = str  # one of: boilerplate, content_field, evidence_span, dynamics_label, gold_answer, unknown


@dataclass
class Region:
    """Contiguous character span with a single φ tag."""
    start: int
    end: int
    phi: PhiTag
    field_name: str | None = None  # e.g., "mechanism_genome" when phi=content_field
    text: str = ""

    def __post_init__(self) -> None:
        if not self.text and self.end > self.start:
            self.text = ""  # filled in by parser


@dataclass
class ParsedRollout:
    raw_text: str
    task_type: str | None = None
    schema: TaskSchema | None = None
    parsed_json: dict | None = None
    regions: list[Region] = field(default_factory=list)
    schema_valid: bool = False
    gold_values: dict[str, Any] = field(default_factory=dict)
    evidence_spans: list[str] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------
# JSON extraction (the model wraps JSON in markdown fences or chatter)
# --------------------------------------------------------------------------
_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BARE_JSON = re.compile(r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})", re.DOTALL)  # one level of nesting


def extract_json(text: str) -> tuple[dict | None, tuple[int, int] | None, str | None]:
    """Try fenced ```json``` first, then last bare {...} block.

    Returns (parsed_dict, (start, end) in raw text, error_msg).
    """
    m = _JSON_BLOCK.search(text)
    if m:
        try:
            return json.loads(m.group(1)), (m.start(1), m.end(1)), None
        except json.JSONDecodeError as e:
            return None, (m.start(1), m.end(1)), f"fenced JSON parse failed: {e}"

    # bare JSON: take the LAST balanced {...} (likely the final answer)
    last_obj_match = None
    for m in _BARE_JSON.finditer(text):
        last_obj_match = m
    if last_obj_match:
        try:
            return json.loads(last_obj_match.group(1)), (last_obj_match.start(1), last_obj_match.end(1)), None
        except json.JSONDecodeError as e:
            return None, (last_obj_match.start(1), last_obj_match.end(1)), f"bare JSON parse failed: {e}"

    return None, None, "no JSON object found in text"


# --------------------------------------------------------------------------
# Region tagging
# --------------------------------------------------------------------------
def _find_value_span_in_json_text(json_text: str, key: str, base_offset: int) -> tuple[int, int] | None:
    """Locate the character span of a string value for `key` inside json_text.

    Returns (start, end) in the ORIGINAL text (json_text offsets + base_offset).
    Naive but reliable for the common single-line-value case; falls back to None.
    """
    # Match: "key": "...value..."
    pat = re.compile(r'"' + re.escape(key) + r'"\s*:\s*"((?:[^"\\]|\\.)*)"')
    m = pat.search(json_text)
    if m:
        return base_offset + m.start(1), base_offset + m.end(1)
    return None


def _find_quoted_spans(text: str) -> list[tuple[int, int]]:
    """Find every '"..."' substring; candidate evidence quotes."""
    return [(m.start(1), m.end(1)) for m in re.finditer(r'"((?:[^"\\]|\\.)+)"', text)]


def _is_dynamics_value(s: Any) -> bool:
    return isinstance(s, str) and canonical_dynamics(s) is not None


def parse_rollout(text: str, task_type: str | None = None) -> ParsedRollout:
    """Main entry point. Parse a model rollout and tag its regions."""
    pr = ParsedRollout(raw_text=text, task_type=task_type)
    if task_type:
        pr.schema = get_schema(task_type)

    # 1. extract JSON
    parsed, span, err = extract_json(text)
    if err:
        pr.parse_errors.append(err)
    if parsed is None:
        # Whole text becomes 'unknown'; verifier will hit schema_valid=False
        pr.regions = [Region(0, len(text), "unknown", text=text)]
        return pr
    pr.parsed_json = parsed
    json_start, json_end = span

    # 2. schema validity check
    if pr.schema:
        missing = [k for k in pr.schema.answer_keys if k not in parsed]
        if not missing:
            pr.schema_valid = True
            for k in pr.schema.answer_keys:
                pr.gold_values[k] = parsed[k]
        else:
            pr.parse_errors.append(f"schema missing keys: {missing}")

    # 3. tag the JSON interior
    regions: list[Region] = []
    # pre-JSON prose
    if json_start > 0:
        regions.append(Region(0, json_start, "boilerplate", text=text[:json_start]))
    # JSON body — naive but useful: tag each declared key
    json_text = text[json_start:json_end]
    cursor = json_start
    tagged_intervals: list[tuple[int, int, str, str | None]] = []  # (s, e, phi, field_name)

    # 3a. gene-field content tags
    for fld in GENE_FIELDS:
        span_pos = _find_value_span_in_json_text(json_text, fld, json_start)
        if span_pos:
            tagged_intervals.append((*span_pos, "content_field", fld))
    # alias keys ("mechanism" without _genome suffix) the model might use
    for alias, canon in [("mechanism", "mechanism_genome"), ("niche", "niche_genome"),
                          ("observation", "observation_genome"), ("limitation", "limitation_genome"),
                          ("delta", "delta_genome"), ("claim", "claim_genome")]:
        if not any(t[3] == canon for t in tagged_intervals):
            span_pos = _find_value_span_in_json_text(json_text, alias, json_start)
            if span_pos:
                tagged_intervals.append((*span_pos, "content_field", canon))

    # 3b. dynamics_label — look for "dynamics" key with a recognized value
    if isinstance(parsed.get("dynamics"), str) and _is_dynamics_value(parsed["dynamics"]):
        sp = _find_value_span_in_json_text(json_text, "dynamics", json_start)
        if sp:
            tagged_intervals.append((*sp, "dynamics_label", None))
    if isinstance(parsed.get("driver"), str):
        sp = _find_value_span_in_json_text(json_text, "driver", json_start)
        if sp:
            tagged_intervals.append((*sp, "gold_answer", "driver"))

    # 3c. T1-T4 gold_answer keys → gold_answer φ tag (verifier-anchored)
    if pr.schema:
        for ak in pr.schema.answer_keys:
            if ak in {"mechanism_genome", "niche_genome", "observation_genome",
                      "limitation_genome", "delta_genome", "claim_genome"}:
                continue  # already tagged content_field
            sp = _find_value_span_in_json_text(json_text, ak, json_start)
            if sp:
                tagged_intervals.append((*sp, "gold_answer", ak))

    # 3d. evidence_spans — quoted strings inside content_field regions count as evidence
    #     Heuristic: any '\"...\"' pattern inside a content_field value
    for s, e, phi, fname in tagged_intervals:
        if phi != "content_field":
            continue
        seg = text[s:e]
        for qs, qe in _find_quoted_spans(seg):
            abs_s, abs_e = s + qs, s + qe
            pr.evidence_spans.append(text[abs_s:abs_e])
            tagged_intervals.append((abs_s, abs_e, "evidence_span", fname))

    # 4. merge tagged_intervals into non-overlapping regions; fill gaps with boilerplate
    # sort by start; later (higher-priority) tags overwrite earlier ones in overlap
    priority = {"evidence_span": 4, "dynamics_label": 3, "gold_answer": 2,
                "content_field": 1, "boilerplate": 0, "unknown": 0}
    tagged_intervals.sort(key=lambda x: (x[0], -priority.get(x[2], 0)))

    char_tags: list[tuple[str, str | None]] = [("boilerplate", None)] * (json_end - json_start)
    for s, e, phi, fname in tagged_intervals:
        for i in range(s - json_start, min(e - json_start, len(char_tags))):
            cur_phi, cur_fn = char_tags[i]
            if priority.get(phi, 0) >= priority.get(cur_phi, 0):
                char_tags[i] = (phi, fname)

    # compress runs of same (phi, fname) into Regions
    if char_tags:
        run_start = 0
        run_phi, run_fn = char_tags[0]
        for i in range(1, len(char_tags)):
            if char_tags[i] != (run_phi, run_fn):
                regions.append(Region(json_start + run_start, json_start + i,
                                       run_phi, run_fn, text=text[json_start + run_start:json_start + i]))
                run_start = i
                run_phi, run_fn = char_tags[i]
        regions.append(Region(json_start + run_start, json_end,
                               run_phi, run_fn, text=text[json_start + run_start:json_end]))

    # post-JSON prose
    if json_end < len(text):
        regions.append(Region(json_end, len(text), "boilerplate", text=text[json_end:]))

    pr.regions = regions
    return pr


# --------------------------------------------------------------------------
# Token alignment (used at training time with a HF tokenizer)
# --------------------------------------------------------------------------
def tokenize_with_roles(text: str, tokenizer, task_type: str | None = None,
                        add_special_tokens: bool = False) -> tuple[list[int], list[str], list[str | None]]:
    """Tokenize text and return (input_ids, phi_per_token, field_per_token).

    Requires a fast tokenizer (returns offset_mapping).
    """
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=add_special_tokens)
    offsets = enc["offset_mapping"]
    pr = parse_rollout(text, task_type)
    # for fast lookup: character-indexed (phi, field) array
    char_phi: list[str] = ["unknown"] * len(text)
    char_fld: list[str | None] = [None] * len(text)
    for r in pr.regions:
        for i in range(r.start, min(r.end, len(text))):
            char_phi[i] = r.phi
            char_fld[i] = r.field_name

    phi_per_token: list[str] = []
    fld_per_token: list[str | None] = []
    for (s, e) in offsets:
        if s == e:  # zero-length tokens (special)
            phi_per_token.append("boilerplate")
            fld_per_token.append(None)
            continue
        # plurality vote across chars in this token
        from collections import Counter
        c = Counter(char_phi[s:e])
        f = Counter(char_fld[s:e])
        phi_per_token.append(c.most_common(1)[0][0])
        fld_per_token.append(f.most_common(1)[0][0])
    return enc["input_ids"], phi_per_token, fld_per_token


if __name__ == "__main__":
    # quick smoke
    sample = '''Here is my answer:
```json
{
  "driver": "mechanism",
  "dynamics": "Adaptive Radiation"
}
```
'''
    pr = parse_rollout(sample, "T3-01_single_dynamics")
    print(f"schema_valid={pr.schema_valid}  gold_values={pr.gold_values}")
    for r in pr.regions:
        print(f"  [{r.start:>3}:{r.end:<3}] {r.phi:<14} field={r.field_name!s:<20} text={r.text[:60]!r}")
