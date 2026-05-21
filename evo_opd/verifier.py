"""evo-OPD verifier — sparse, deterministic, teacher-free correctness signal.

v(y) ∈ [0,1] with four sub-scores (weights from lv_opd_plan.md §3.1(b)):
  - schema_valid              × 0.20
  - evidence_citation_frac    × 0.30
  - dynamics_consistency      × 0.30
  - exact_match_when_applicable × 0.20

Falls back gracefully when sub-scores don't apply (set weight → 0, renormalize).

NOTE: dynamics_consistency reuses `agent/genome_differ.py` only for its
deterministic decision-tree logic. If that module's structure differs from what
we assume here, the verifier degrades to schema + evidence checks (still useful).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parser import ParsedRollout, parse_rollout
from .schemas import (
    DYNAMICS_LABELS,
    canonical_dynamics,
)


@dataclass
class VerifierScore:
    v: float
    schema_valid: float = 0.0
    evidence_citation_frac: float = 0.0
    dynamics_consistency: float = 0.0
    exact_match: float = 0.0
    notes: list[str] = field(default_factory=list)
    components_active: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "v": round(self.v, 4),
            "schema_valid": round(self.schema_valid, 4),
            "evidence_citation_frac": round(self.evidence_citation_frac, 4),
            "dynamics_consistency": round(self.dynamics_consistency, 4),
            "exact_match": round(self.exact_match, 4),
            "components_active": self.components_active,
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Evidence citation: every quoted span in a gene-field value must appear
# (case-insensitive, whitespace-collapsed) in the source paper text.
# --------------------------------------------------------------------------
def _normalize_for_match(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def evidence_citation_score(pr: ParsedRollout, source_text: str | None) -> tuple[float, list[str]]:
    """Fraction of quoted evidence spans that appear in source_text.

    Returns (frac, [unmatched spans]). If no spans claimed OR no source provided,
    component is INACTIVE (returns 1.0 to be neutral; caller checks active list).
    """
    if not pr.evidence_spans or not source_text:
        return 1.0, []
    source_norm = _normalize_for_match(source_text)
    unmatched: list[str] = []
    n_ok = 0
    for span in pr.evidence_spans:
        # strip the surrounding quote chars the parser captured
        s = span.strip().strip('"')
        if not s:
            continue
        if _normalize_for_match(s) in source_norm:
            n_ok += 1
        else:
            unmatched.append(s)
    total = n_ok + len(unmatched)
    return (n_ok / total if total else 1.0), unmatched


# --------------------------------------------------------------------------
# Dynamics consistency: if the model emitted both per-field fates AND a
# dynamics label, do they match the GenomeDiff decision tree (paper §3.3)?
# --------------------------------------------------------------------------
def _dynamics_from_fates(fates: dict | None) -> str | None:
    """Apply the decision tree from the paper §3.3 / agent/genome_differ.py.

    Inputs: dict {gene_field_name -> 'INHERITED'|'MUTATED'|'LOST'|'NOVEL'|'HYBRIDIZED'}.
    Returns the dominant dynamics label or None if undecidable.

    Decision tree (top-to-bottom, first match wins):
      1. ≥2 genes HYBRIDIZED (from external lineage) → Hybridization
      2. ALL mechanism genes LOST + NOVEL replacements + niche INHERITED/MUTATED → Speciation
      3. mechanism INHERITED/MUTATED + niche MUTATED/LOST/NOVEL → Adaptive Radiation
      4. ZERO genes INHERITED or MUTATED → Niche Competition
      5. default → Mutation
    """
    if not isinstance(fates, dict) or not fates:
        return None
    n_hyb = sum(1 for v in fates.values() if v == "HYBRIDIZED")
    if n_hyb >= 2:
        return "Hybridization"

    mech_keys = [k for k in fates if "mechanism" in k.lower()]
    niche_keys = [k for k in fates if "niche" in k.lower() or "problem" in k.lower()]
    mech_fates = {fates[k] for k in mech_keys}
    niche_fates = {fates[k] for k in niche_keys}

    if mech_keys and all(f in {"LOST"} for f in mech_fates) and any(f == "NOVEL" for f in fates.values()):
        if any(f in {"INHERITED", "MUTATED"} for f in niche_fates):
            return "Speciation"

    if mech_keys and any(f in {"INHERITED", "MUTATED"} for f in mech_fates):
        if niche_keys and any(f in {"MUTATED", "LOST", "NOVEL"} for f in niche_fates):
            return "Adaptive Radiation"

    if not any(f in {"INHERITED", "MUTATED"} for f in fates.values()):
        return "Niche Competition"

    return "Mutation"


def dynamics_consistency_score(pr: ParsedRollout) -> tuple[float, str | None]:
    """Score how well a declared dynamics label agrees with declared fates.

    Returns (score in {0.0, 1.0}, mismatch_msg or None).
    INACTIVE (returns 1.0) if either fates or dynamics not present.
    """
    parsed = pr.parsed_json or {}
    declared = canonical_dynamics(parsed.get("dynamics", ""))
    fates = parsed.get("fates") or parsed.get("fates_per_field") or parsed.get("genome_field_fates")
    if declared is None or fates is None:
        return 1.0, None  # inactive
    inferred = _dynamics_from_fates(fates)
    if inferred is None:
        return 1.0, "fates structure unrecognized"
    if declared == inferred:
        return 1.0, None
    return 0.0, f"declared={declared} but fates imply {inferred}"


# --------------------------------------------------------------------------
# Exact match against gold (composite all-or-nothing per task)
# --------------------------------------------------------------------------
def _eq_ci(a, b) -> bool:
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def exact_match_score(pr: ParsedRollout, gold_answer: dict | None) -> tuple[float, list[str]]:
    """All-or-nothing composite match.

    For each gold key, compare the model's value (deep, case-insensitive for strings).
    Returns (1.0 if all match else 0.0, list of mismatched keys).
    INACTIVE (returns 1.0) if no gold provided.
    """
    if not gold_answer:
        return 1.0, []
    parsed = pr.parsed_json or {}
    mismatches: list[str] = []
    for k, gv in gold_answer.items():
        mv = parsed.get(k)
        if isinstance(gv, list) and isinstance(mv, list):
            if len(gv) != len(mv) or not all(_eq_ci(x, y) for x, y in zip(gv, mv)):
                mismatches.append(k)
        elif isinstance(gv, dict) and isinstance(mv, dict):
            if set(gv.keys()) != set(mv.keys()) or not all(_eq_ci(gv[k2], mv.get(k2)) for k2 in gv):
                mismatches.append(k)
        else:
            if not _eq_ci(gv, mv):
                mismatches.append(k)
    return (1.0 if not mismatches else 0.0), mismatches


# --------------------------------------------------------------------------
# Top-level
# --------------------------------------------------------------------------
def compute_verifier(
    text: str,
    task_type: str | None = None,
    source_text: str | None = None,
    gold_answer: dict | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[VerifierScore, ParsedRollout]:
    """Compute v(y) using the four components, with dynamic weight renormalization."""
    w = weights or {
        "schema": 0.20,
        "evidence": 0.30,
        "dynamics": 0.30,
        "exact": 0.20,
    }
    pr = parse_rollout(text, task_type)
    score = VerifierScore(v=0.0)

    # schema
    s_schema = 1.0 if pr.schema_valid else 0.0
    score.schema_valid = s_schema
    score.components_active.append("schema")

    # evidence (active only if model claimed any spans AND source provided)
    s_ev, unmatched = evidence_citation_score(pr, source_text)
    score.evidence_citation_frac = s_ev
    ev_active = bool(pr.evidence_spans and source_text)
    if ev_active:
        score.components_active.append("evidence")
    if unmatched:
        score.notes.append(f"unmatched evidence: {len(unmatched)} spans")

    # dynamics (active if both fates and dynamics present)
    s_dy, dy_msg = dynamics_consistency_score(pr)
    score.dynamics_consistency = s_dy
    parsed = pr.parsed_json or {}
    dy_active = bool(parsed.get("dynamics") and (parsed.get("fates") or parsed.get("fates_per_field")
                                                  or parsed.get("genome_field_fates")))
    if dy_active:
        score.components_active.append("dynamics")
    if dy_msg:
        score.notes.append(dy_msg)

    # exact match (active if gold provided)
    s_ex, mis = exact_match_score(pr, gold_answer)
    score.exact_match = s_ex
    ex_active = gold_answer is not None
    if ex_active:
        score.components_active.append("exact")
    if mis:
        score.notes.append(f"exact_match mismatches: {mis}")

    # weighted sum with renormalization over active components
    pieces: list[tuple[float, float, bool]] = [
        (w["schema"],   s_schema, True),
        (w["evidence"], s_ev,     ev_active),
        (w["dynamics"], s_dy,     dy_active),
        (w["exact"],    s_ex,     ex_active),
    ]
    active_w = sum(wi for wi, _, act in pieces if act)
    if active_w > 0:
        v = sum(wi * si for wi, si, act in pieces if act) / active_w
    else:
        v = s_schema  # degenerate
    score.v = v
    return score, pr


if __name__ == "__main__":
    # smoke: correct T3-01 answer
    text_ok = '''```json
{"driver": "mechanism", "dynamics": "Adaptive Radiation"}
```'''
    gold = {"driver": "mechanism", "dynamics": "Adaptive Radiation"}
    s, _ = compute_verifier(text_ok, "T3-01_single_dynamics", gold_answer=gold)
    print("CORRECT:", s.to_dict())

    # smoke: schema-valid but wrong dynamics
    text_wrong = '''```json
{"driver": "mechanism", "dynamics": "Mutation"}
```'''
    s, _ = compute_verifier(text_wrong, "T3-01_single_dynamics", gold_answer=gold)
    print("WRONG  :", s.to_dict())

    # smoke: dynamics-fates inconsistency
    text_inco = '''```json
{"fates": {"mechanism_genome": "INHERITED", "niche_genome": "INHERITED"},
 "dynamics": "Speciation"}
```'''
    s, _ = compute_verifier(text_inco, "genome_diff_annotate")
    print("INCONSI:", s.to_dict())
