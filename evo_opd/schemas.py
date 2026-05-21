"""Schemas, field weights, and dynamics enums for evo-OPD.

Centralizes the task-type → expected-output-shape mapping so parser/verifier/lineage
stay consistent. Field weights are the α(φ(t)) values from lv_opd_plan.md §3.1(a).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# --------------------------------------------------------------------------
# Field weights α(φ(t)) — see lv_opd_plan.md §3.1(a)
# --------------------------------------------------------------------------
FIELD_WEIGHT: dict[str, float] = {
    "boilerplate":     0.3,   # JSON brackets, field-name strings, whitespace
    "content_field":   1.0,   # values of the 6 gene fields
    "evidence_span":   1.5,   # quoted text that must appear in source paper
    "dynamics_label":  2.0,   # load-bearing for all-or-nothing composite scoring
    "gold_answer":     0.0,   # decoupled: verifier-anchored (B); handled by v(y)
    "unknown":         0.5,   # fallback if parser can't tag the token
}

# --------------------------------------------------------------------------
# Six gene-card content fields (the "DNA" of a paper)
# --------------------------------------------------------------------------
GENE_FIELDS: tuple[str, ...] = (
    "mechanism_genome",
    "niche_genome",
    "observation_genome",
    "limitation_genome",
    "delta_genome",
    "claim_genome",
)
# Friendly aliases (some prompts use shortened names)
GENE_FIELD_ALIASES: dict[str, str] = {
    "mechanism": "mechanism_genome",
    "niche": "niche_genome",
    "problem": "niche_genome",
    "observation": "observation_genome",
    "limitation": "limitation_genome",
    "delta": "delta_genome",
    "claim": "claim_genome",
}

# --------------------------------------------------------------------------
# Five evolutionary dynamics + Isolation null (matches paper Table 1)
# --------------------------------------------------------------------------
DYNAMICS_LABELS: tuple[str, ...] = (
    "Mutation",
    "Adaptive Radiation",
    "Hybridization",
    "Speciation",
    "Niche Competition",
    "Isolation",  # null baseline; sometimes appears in gold_answers
)
DYNAMICS_ALIASES: dict[str, str] = {
    "mutation": "Mutation",
    "M": "Mutation",
    "adaptive radiation": "Adaptive Radiation",
    "ar": "Adaptive Radiation",
    "AR": "Adaptive Radiation",
    "hybridization": "Hybridization",
    "h": "Hybridization",
    "H": "Hybridization",
    "speciation": "Speciation",
    "s": "Speciation",
    "S": "Speciation",
    "niche competition": "Niche Competition",
    "nc": "Niche Competition",
    "NC": "Niche Competition",
    "isolation": "Isolation",
}

# --------------------------------------------------------------------------
# Drivers / contribution types / gene-field fates (small categorical answer spaces)
# --------------------------------------------------------------------------
DRIVERS: tuple[str, ...] = ("mechanism", "niche", "observation", "limitation")
CONTRIBUTION_TYPES: tuple[str, ...] = ("method", "dataset", "analysis", "system", "theory", "benchmark")
GENE_FIELD_FATES: tuple[str, ...] = ("INHERITED", "MUTATED", "LOST", "NOVEL", "HYBRIDIZED")
GENE_ROLES: tuple[str, ...] = ("driver", "passenger")


# --------------------------------------------------------------------------
# Task → answer schema. Used by parser + verifier.
# Each task_type maps to the keys we expect in the model's JSON answer.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSchema:
    task_type: str
    answer_keys: tuple[str, ...]
    is_composite: bool = False  # if True, all sub-fields must be correct (all-or-nothing)
    notes: str = ""

# Catalog of task schemas — keys harvested from real
# IdeaEvolving/gene_exam/Questions/*/instances.json on 2026-05-17.
# Every task is composite (all keys must be correct) to match the benchmark's
# all-or-nothing scoring.
TASK_SCHEMAS: dict[str, TaskSchema] = {
    # ---------- T1 Genome Abstraction (5) ----------
    "T1-01_contribution_type":         TaskSchema("T1-01_contribution_type",         ("multi_contrib_types",), True),
    "T1-02_genome_field_type":         TaskSchema("T1-02_genome_field_type",         ("GenomeField1Type",), True),
    "T1-03_driver_vs_passenger":       TaskSchema("T1-03_driver_vs_passenger",       ("driver_gene", "passenger_gene"), True),
    "T1-04_lineage_position":          TaskSchema("T1-04_lineage_position",          ("contribution_type", "label"), True),
    "T1-05_cross_lineage_bridge":      TaskSchema("T1-05_cross_lineage_bridge",      ("label",), True),

    # ---------- T2 Inheritance Tracing (12) ----------
    "T2-01_ordering_5":                TaskSchema("T2-01_ordering_5",                ("correct_order",), True),
    "T2-02_ordering_6":                TaskSchema("T2-02_ordering_6",                ("correct_order",), True),
    "T2-03_ordering_7":                TaskSchema("T2-03_ordering_7",                ("correct_order",), True),
    "T2-04_grouping_8":                TaskSchema("T2-04_grouping_8",                ("ordered_group_a", "ordered_group_b"), True),
    "T2-05_grouping_8_medium":         TaskSchema("T2-05_grouping_8_medium",         ("ordered_group_a", "ordered_group_b"), True),
    "T2-06_grouping_9_triple":         TaskSchema("T2-06_grouping_9_triple",         ("ordered_groups",), True),
    "T2-07_lim_delta_match":           TaskSchema("T2-07_lim_delta_match",           ("mapping",), True),
    "T2-08_lim_delta_mixed":           TaskSchema("T2-08_lim_delta_mixed",           ("mapping",), True),
    "T2-09_lim_delta_chain":           TaskSchema("T2-09_lim_delta_chain",           ("mapping",), True),
    "T2-10_genome_field_assign_2p":    TaskSchema("T2-10_genome_field_assign_2p",    ("assignments_with_types", "dynamics"), True),
    "T2-11_genome_field_assign_3p_9a": TaskSchema("T2-11_genome_field_assign_3p_9a", ("dynamics",), True),
    "T2-12_gene_alignment":            TaskSchema("T2-12_gene_alignment",            ("assignments",), True),

    # ---------- T3 Evolutionary Reasoning (17) ----------
    "T3-01_single_dynamics":           TaskSchema("T3-01_single_dynamics",           ("driver", "dynamics"), True),
    "T3-02_genome_field_change":       TaskSchema("T3-02_genome_field_change",       ("driver", "dynamics", "source_genome_status"), True),
    "T3-03_driver_dynamics":           TaskSchema("T3-03_driver_dynamics",           ("driver", "dynamics"), True),
    "T3-04_genome_field_change_shown": TaskSchema("T3-04_genome_field_change_shown", ("driver", "dynamics", "source_genome_status"), True),
    "T3-05_driver_summary":            TaskSchema("T3-05_driver_summary",            ("driver", "dynamics", "source_genome_status"), True),
    "T3-06_dynamics_mech":             TaskSchema("T3-06_dynamics_mech",             ("dynamics",), True),
    "T3-07_blind_change":              TaskSchema("T3-07_blind_change",              ("driver", "dynamics", "source_genome_status"), True),
    "T3-08_driver_unlabeled":          TaskSchema("T3-08_driver_unlabeled",          ("driver", "dynamics", "source_genome_status"), True),
    "T3-09_relation_classify":         TaskSchema("T3-09_relation_classify",         ("dynamics", "label"), True),
    "T3-10_genome_direction":          TaskSchema("T3-10_genome_direction",          ("driver", "dynamics", "label"), True),
    "T3-11_evo_tempo":                 TaskSchema("T3-11_evo_tempo",                 ("label",), True),
    "T3-12_evo_pattern":               TaskSchema("T3-12_evo_pattern",               ("correct_dynamics", "label"), True),
    "T3-13_hidden_gene_fate":          TaskSchema("T3-13_hidden_gene_fate",          ("G1_status", "G2_status", "dynamics"), True),
    "T3-14_hybrid_provenance":         TaskSchema("T3-14_hybrid_provenance",         ("gene_sources",), True),
    "T3-15_gene_tracking":             TaskSchema("T3-15_gene_tracking",             ("correct_dynamics",), True),
    "T3-16_dynamics_boundary":         TaskSchema("T3-16_dynamics_boundary",         ("correct_dynamics",), True),
    "T3-17_multi_citation":            TaskSchema("T3-17_multi_citation",            ("relation",), True),

    # ---------- T4 Lineage Verification (8) ----------
    "T4-01_consistency_check":         TaskSchema("T4-01_consistency_check",         ("contribution_type", "label", "verify"), True),
    "T4-02_intruder_domain":           TaskSchema("T4-02_intruder_domain",           ("intruder", "lineage_members"), True),
    "T4-03_wrong_step":                TaskSchema("T4-03_wrong_step",                ("correct_dynamics", "correct_order", "label"), True),
    "T4-04_next_hop":                  TaskSchema("T4-04_next_hop",                  ("driver", "dynamics", "label"), True),
    "T4-05_parent_genome":             TaskSchema("T4-05_parent_genome",             ("driver", "label"), True),
    "T4-06_missing_link":              TaskSchema("T4-06_missing_link",              ("bridge_paper", "driver", "dynamics"), True),
    "T4-07_gene_bridge":               TaskSchema("T4-07_gene_bridge",               ("correct_dynamics", "label"), True),
    "T4-08_citation_consistency":      TaskSchema("T4-08_citation_consistency",      ("claim_label", "source_genome_status"), True),

    # ---------- Free-form gene-card / SFT training task types (no fixed gold) ----------
    "gene_card_extract":               TaskSchema("gene_card_extract",               GENE_FIELDS, False,
        "JSON with 6 gene fields + evidence quotes; verifier uses citation regex"),
    "genome_diff_annotate":            TaskSchema("genome_diff_annotate",            ("fates", "driver", "dynamics"), True,
        "JSON with per-field fates + dynamics; verifier uses genome_differ.py decision tree"),
    "lineage_trace_reconstruct":       TaskSchema("lineage_trace_reconstruct",       ("ordering", "per_edge_dynamics"), True),
    "lineage_verify":                  TaskSchema("lineage_verify",                  ("valid", "failure_mode", "repair"), True),
    "idea_generate":                   TaskSchema("idea_generate",                   ("problem", "mechanism", "expected_contribution", "lineage_connection"), False),
}


def get_schema(task_type: str) -> TaskSchema | None:
    """Return the registered TaskSchema or None for unrecognized tasks."""
    return TASK_SCHEMAS.get(task_type)


def canonical_dynamics(s: str) -> str | None:
    """Normalize a dynamics string to the canonical capitalized form, or None."""
    if not s:
        return None
    s = s.strip()
    if s in DYNAMICS_LABELS:
        return s
    return DYNAMICS_ALIASES.get(s.lower(), DYNAMICS_ALIASES.get(s))


def canonical_field(name: str) -> str | None:
    """Normalize a gene-field name (e.g., 'mechanism' -> 'mechanism_genome')."""
    if not name:
        return None
    if name in GENE_FIELDS:
        return name
    n = name.lower().strip()
    return GENE_FIELD_ALIASES.get(n) or (n if n in GENE_FIELDS else None)
