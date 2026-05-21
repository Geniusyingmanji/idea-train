"""evo-OPD lineage consistency c(y, p) — self-supervised, teacher-free.

Given a parent gene-card `p` and a student rollout `y` claiming to generate a
child gene-card (with per-field fates + dynamics), score how well y respects
GenomeDiff invariants relative to p.

c(y, p) ∈ [0, 1], mean of four sub-checks (lv_opd_plan.md §3.1(c)):
  - schema_complete       — child has all 6 fields + fates + dynamics
  - fate_content_consistency — claimed fate matches token-overlap evidence
  - dynamics_from_fates   — internal dynamics ↔ fates consistency (reuses verifier)
  - anti_copy_penalty     — child must not just replicate parent verbatim

Inactive when y has no parent context (caller passes p=None → c is 0).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import parse_rollout
from .schemas import GENE_FIELDS, canonical_field
from .verifier import _dynamics_from_fates  # reuse decision-tree implementation


@dataclass
class LineageScore:
    c: float
    schema_complete: float = 0.0
    fate_content_consistency: float = 0.0
    dynamics_from_fates: float = 0.0
    anti_copy: float = 0.0
    per_field_match: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "c": round(self.c, 4),
            "schema_complete": round(self.schema_complete, 4),
            "fate_content_consistency": round(self.fate_content_consistency, 4),
            "dynamics_from_fates": round(self.dynamics_from_fates, 4),
            "anti_copy": round(self.anti_copy, 4),
            "per_field_match": self.per_field_match,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Token-level overlap (cheap, language-agnostic-enough for English science text)
# --------------------------------------------------------------------------
_TOK = re.compile(r"\b[A-Za-z][A-Za-z0-9\-]+\b")


def _tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    return {t.lower() for t in _TOK.findall(s)}


def _jaccard(a: str | None, b: str | None) -> float:
    A, B = _tokens(a), _tokens(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def _is_empty_field(s) -> bool:
    if s is None:
        return True
    if isinstance(s, str):
        return not s.strip() or s.strip().lower() in {"n/a", "na", "none", "null", "-", "—"}
    return False


# --------------------------------------------------------------------------
# Fate-vs-content consistency rules
# --------------------------------------------------------------------------
FATE_RULES = {
    # fate -> (predicate(parent_field, child_field, jaccard) -> bool)
    "INHERITED":  lambda jac, p_emp, c_emp: (not p_emp) and (not c_emp) and (jac >= 0.7),
    "MUTATED":    lambda jac, p_emp, c_emp: (not p_emp) and (not c_emp) and (0.3 <= jac < 0.7),
    "LOST":       lambda jac, p_emp, c_emp: (not p_emp) and c_emp,
    "NOVEL":      lambda jac, p_emp, c_emp: p_emp and (not c_emp),
    # HYBRIDIZED needs external lineage; we accept any case where child has content
    # AND has substantial novel material beyond parent (not pure inheritance)
    "HYBRIDIZED": lambda jac, p_emp, c_emp: (not c_emp) and jac < 0.6,
}


def _extract_parent_fields(parent: dict) -> dict[str, str]:
    """Normalize a parent gene-card dict (which may use legacy_genome or top-level fields)."""
    if not isinstance(parent, dict):
        return {}
    if "legacy_genome" in parent and isinstance(parent["legacy_genome"], dict):
        src = parent["legacy_genome"]
    else:
        src = parent
    out: dict[str, str] = {}
    for f in GENE_FIELDS:
        v = src.get(f)
        if v is None:
            # try alias
            alias = f.replace("_genome", "")
            v = src.get(alias)
        if isinstance(v, str):
            out[f] = v
    return out


def _extract_child_fields(child_parsed: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in GENE_FIELDS:
        v = child_parsed.get(f)
        if v is None:
            alias = f.replace("_genome", "")
            v = child_parsed.get(alias)
        if isinstance(v, str):
            out[f] = v
    return out


def _extract_fates(child_parsed: dict) -> dict[str, str]:
    fates = (child_parsed.get("fates") or child_parsed.get("fates_per_field")
             or child_parsed.get("genome_field_fates") or {})
    if not isinstance(fates, dict):
        return {}
    # canonicalize keys
    out: dict[str, str] = {}
    for k, v in fates.items():
        c = canonical_field(k)
        if c and isinstance(v, str):
            out[c] = v.strip().upper()
    return out


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
def compute_lineage(
    text: str,
    parent: dict | None,
    task_type: str | None = "genome_diff_annotate",
    anti_copy_threshold: float = 0.4,
) -> LineageScore:
    """Score student rollout `text` for lineage consistency given `parent` gene-card."""
    score = LineageScore(c=0.0)
    if parent is None:
        score.notes.append("no parent context provided; lineage score inactive (returns 0)")
        return score

    pr = parse_rollout(text, task_type)
    if not pr.parsed_json:
        score.notes.append("JSON parse failed")
        return score
    parsed = pr.parsed_json

    parent_fields = _extract_parent_fields(parent)
    child_fields = _extract_child_fields(parsed)
    declared_fates = _extract_fates(parsed)

    # --- 1. schema completeness ----------------------------------------
    n_child_fields = sum(1 for f in GENE_FIELDS if f in child_fields)
    n_fates = sum(1 for f in GENE_FIELDS if f in declared_fates)
    has_dyn = bool(parsed.get("dynamics"))
    score.schema_complete = (
        (n_child_fields / len(GENE_FIELDS)) * 0.5
        + (n_fates / len(GENE_FIELDS)) * 0.4
        + (0.1 if has_dyn else 0.0)
    )

    # --- 2. fate-content consistency -----------------------------------
    if declared_fates and child_fields:
        per_field_ok = 0
        per_field_total = 0
        for f in GENE_FIELDS:
            if f not in declared_fates:
                continue
            fate = declared_fates[f]
            rule = FATE_RULES.get(fate)
            if rule is None:
                score.notes.append(f"unknown fate label for {f}: {fate}")
                continue
            p_val = parent_fields.get(f, "")
            c_val = child_fields.get(f, "")
            jac = _jaccard(p_val, c_val)
            ok = rule(jac, _is_empty_field(p_val), _is_empty_field(c_val))
            per_field_total += 1
            if ok:
                per_field_ok += 1
            score.per_field_match[f] = {
                "fate": fate,
                "jaccard": round(jac, 3),
                "parent_empty": _is_empty_field(p_val),
                "child_empty": _is_empty_field(c_val),
                "rule_satisfied": ok,
            }
        score.fate_content_consistency = (per_field_ok / per_field_total) if per_field_total else 0.0
    else:
        score.notes.append("fates or child fields incomplete; fate_content_consistency=0")

    # --- 3. dynamics-from-fates consistency ----------------------------
    if declared_fates and parsed.get("dynamics"):
        inferred = _dynamics_from_fates(declared_fates)
        declared_dy = parsed["dynamics"].strip() if isinstance(parsed["dynamics"], str) else None
        if inferred and declared_dy:
            score.dynamics_from_fates = 1.0 if inferred == declared_dy else 0.0
            if inferred != declared_dy:
                score.notes.append(f"dynamics mismatch: declared={declared_dy} vs inferred={inferred}")
        else:
            score.dynamics_from_fates = 0.0
    else:
        score.dynamics_from_fates = 0.0

    # --- 4. anti-copy: child must differ meaningfully from parent -----
    if child_fields and parent_fields:
        # average jaccard across populated parent fields
        jacs = []
        for f in GENE_FIELDS:
            if not _is_empty_field(parent_fields.get(f)):
                jacs.append(_jaccard(parent_fields.get(f), child_fields.get(f, "")))
        avg_jac = sum(jacs) / len(jacs) if jacs else 0.0
        if avg_jac > (1 - anti_copy_threshold):
            score.anti_copy = 0.0
            score.notes.append(f"child too similar to parent (avg jaccard={avg_jac:.3f})")
        else:
            score.anti_copy = 1.0
    else:
        score.anti_copy = 0.5  # neutral

    # --- aggregate ----------------------------------------------------
    # Equal weight unless schema is broken; broken schema gates the rest.
    if score.schema_complete < 0.3:
        score.c = score.schema_complete * 0.25  # mostly zero
    else:
        score.c = (
            0.20 * score.schema_complete
            + 0.40 * score.fate_content_consistency
            + 0.25 * score.dynamics_from_fates
            + 0.15 * score.anti_copy
        )
    return score


if __name__ == "__main__":
    # toy parent
    parent = {
        "legacy_genome": {
            "mechanism_genome": "Bidirectional transformer encoder with masked language modeling on DNA k-mers.",
            "niche_genome": "Genomic sequence representation; downstream tasks include promoter prediction.",
            "observation_genome": "Outperforms task-specific models by 1-5% AUC.",
            "limitation_genome": "Limited to short context (512 tokens); k-mer tokenization loses single-nucleotide resolution.",
            "delta_genome": "",
            "claim_genome": "",
        }
    }

    # Case 1: well-formed child claiming Adaptive Radiation (mechanism inherited, niche shifted)
    child_text = '''```json
{
  "mechanism_genome": "Bidirectional transformer encoder with masked language modeling on DNA k-mers.",
  "niche_genome": "Single-cell RNA-seq foundation model; downstream tasks include cell-type annotation.",
  "observation_genome": "Achieves competitive zero-shot performance on cell-type classification.",
  "limitation_genome": "Requires large-scale single-cell datasets for pretraining.",
  "delta_genome": "Repurposes BERT-style pretraining from DNA to single-cell expression matrices.",
  "claim_genome": "Foundation models for genomics transfer to single-cell transcriptomics.",
  "fates": {"mechanism_genome": "INHERITED", "niche_genome": "MUTATED",
            "observation_genome": "NOVEL", "limitation_genome": "MUTATED",
            "delta_genome": "NOVEL", "claim_genome": "NOVEL"},
  "dynamics": "Adaptive Radiation"
}
```'''
    s = compute_lineage(child_text, parent, "genome_diff_annotate")
    print("AR (well-formed):", s.to_dict())

    # Case 2: copy-cat (child = parent)
    copy_text = '''```json
{
  "mechanism_genome": "Bidirectional transformer encoder with masked language modeling on DNA k-mers.",
  "niche_genome": "Genomic sequence representation; downstream tasks include promoter prediction.",
  "observation_genome": "Outperforms task-specific models by 1-5% AUC.",
  "limitation_genome": "Limited to short context (512 tokens); k-mer tokenization loses single-nucleotide resolution.",
  "delta_genome": "",
  "claim_genome": "",
  "fates": {"mechanism_genome": "INHERITED", "niche_genome": "INHERITED",
            "observation_genome": "INHERITED", "limitation_genome": "INHERITED"},
  "dynamics": "Mutation"
}
```'''
    s = compute_lineage(copy_text, parent, "genome_diff_annotate")
    print("COPY-CAT:", s.to_dict())

    # Case 3: declared dynamics doesn't match fates
    inco_text = '''```json
{
  "mechanism_genome": "completely new mechanism unrelated to parent",
  "niche_genome": "completely new niche unrelated to parent",
  "observation_genome": "new",
  "limitation_genome": "new",
  "delta_genome": "new",
  "claim_genome": "new",
  "fates": {"mechanism_genome": "NOVEL", "niche_genome": "NOVEL",
            "observation_genome": "NOVEL", "limitation_genome": "NOVEL"},
  "dynamics": "Mutation"
}
```'''
    s = compute_lineage(inco_text, parent, "genome_diff_annotate")
    print("INCONSI :", s.to_dict())
