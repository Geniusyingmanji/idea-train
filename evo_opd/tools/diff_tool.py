"""genome_diff tool — compute structured parent→proposal alignment.

Wraps IdeaEvolving's `genome_differ.align_gene_cards` + `classify_dynamics`.
The agent calls this AFTER building a proposed genome — to self-check whether
the proposal's gene fates (INHERITED/MUTATED/LOST/NOVEL/HYBRIDIZED) match
the kind of evolutionary dynamics it intended.

Input: parent_id (oa:W... or paper:foo:2024) + proposed_genome (6-field dict)
Output: {
    "alignments": [...],   # per-gene fate + similarity
    "gene_fates_summary": {...},
    "inferred_dynamics": "Mutation" | "Adaptive Radiation" | ...,
    "primary_driver": "mechanism" | "objective" | ...,
    "diagnostics": "human-readable assessment"
}

Allows agent to iterate: propose → diff → if fates don't match intent, revise.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
import sys

# Use upstream genome_differ for the heavy lifting
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
from agent.genome_differ import (
    align_gene_cards, classify_dynamics, lexical_similarity,
)

from .genome_tool import CANON_GENOME_FIELDS, GenomeExtractTool


# stopwords for short-name truncation
_STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
         "paper", "method", "model"}

_TOK_NAME = re.compile(r"[A-Za-z][A-Za-z0-9+\-]*")


def _short_name(text: str, fallback: str, n: int = 6) -> str:
    words = [w for w in _TOK_NAME.findall(text or "") if w.lower() not in _STOP]
    return " ".join(words[:n])[:72] if words else fallback


def _split_mechanism(text: str, max_parts: int = 3) -> list[str]:
    """Split a long mechanism text into 1-3 component sentences."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+|;\s+", text)
    parts = [p.strip(" .;") for p in parts if len(p.strip()) >= 20]
    if len(parts) <= 1:
        parts = re.split(r";\s+|\s+\+\s+|\s+then\s+", text)
    cleaned = [p.strip(" .;") for p in parts if len(p.split()) >= 5]
    return cleaned[:max_parts] or [text]


def genome_to_gene_card(paper_id: str, genome: dict, title: str = "",
                       ) -> dict:
    """Convert a 6-field gene_genome into a multi-gene card consumed by
    `align_gene_cards`. The card has a `genes` list where each gene has
    {gene_id, gene_text, gene_type, gene_role, heritability}.

    Conventions:
      mechanism_genome → 1-3 'mechanism' genes (first is driver, rest passenger)
      niche_genome     → 1 'objective' gene
      limitation_genome → 1 'limitation' gene
      observation_genome → 1 'observation' gene
      delta_genome     → encoded in mechanism (not a separate gene)
      claim_genome     → 1 'claim' gene
    """
    genes = []
    idx = 1

    for i, comp in enumerate(_split_mechanism(genome.get("mechanism_genome", ""), 3)):
        genes.append({
            "gene_id": f"{paper_id}.G{idx}",
            "gene_name": _short_name(comp, f"mech_{idx}"),
            "gene_type": "mechanism",
            "gene_role": "driver" if i == 0 else "passenger",
            "heritability": "high" if i == 0 else "medium",
            "gene_text": comp,
        })
        idx += 1

    for field_name, gtype, role, h in [
        ("niche_genome",      "objective",  "driver_candidate", "medium"),
        ("limitation_genome", "limitation", "future_pressure",  "medium"),
        ("observation_genome", "observation", "passenger",       "low"),
        ("claim_genome",      "claim",      "passenger",        "medium"),
    ]:
        v = (genome.get(field_name) or "").strip()
        if not v:
            continue
        genes.append({
            "gene_id": f"{paper_id}.G{idx}",
            "gene_name": _short_name(v, f"{gtype}_{idx}"),
            "gene_type": gtype,
            "gene_role": role,
            "heritability": h,
            "gene_text": v,
        })
        idx += 1

    return {
        "paper_id": paper_id,
        "title": title,
        "genes": genes,
    }


@dataclass
class DiffResult:
    parent_id: str
    n_genes_parent: int
    n_genes_proposed: int
    gene_fates_summary: dict       # {gene_id_local: "INHERITED"/"MUTATED"/...}
    inferred_dynamics: str
    primary_driver: str | None
    alignments: list[dict]
    diagnostics: str               # human-readable
    error: str | None = None


class GenomeDiffTool:
    """Wraps align_gene_cards + classify_dynamics for the agent."""

    def __init__(self, extract_tool: GenomeExtractTool | None = None):
        # we need extract_tool to get parent's structured genome
        self.extract_tool = extract_tool or GenomeExtractTool()

    def diff(self, parent_id: str, proposed_genome: dict) -> DiffResult:
        # 1) ensure proposed_genome is a dict of strings
        if not isinstance(proposed_genome, dict):
            return DiffResult(
                parent_id=parent_id, n_genes_parent=0, n_genes_proposed=0,
                gene_fates_summary={}, inferred_dynamics="Unknown",
                primary_driver=None, alignments=[],
                diagnostics="proposed_genome is not a dict",
                error="bad_input",
            )

        # 2) extract parent's structured genome (via GPT-5.5, cached)
        parent_res = self.extract_tool.extract(parent_id)
        if parent_res.error:
            return DiffResult(
                parent_id=parent_id, n_genes_parent=0, n_genes_proposed=0,
                gene_fates_summary={}, inferred_dynamics="Unknown",
                primary_driver=None, alignments=[],
                diagnostics=f"failed to extract parent genome: {parent_res.error}",
                error="parent_extract_failed",
            )

        # 3) build gene cards for both
        parent_card = genome_to_gene_card(
            parent_id, parent_res.genome, title=parent_res.title,
        )
        proposed_card = genome_to_gene_card(
            "proposed", proposed_genome, title="(proposal)",
        )

        # 4) align — deterministic, no LLM
        diff = align_gene_cards(parent_card, proposed_card)

        alignments = diff.get("alignments", [])
        gene_fates_summary = diff.get("gene_fates", {})
        primary_driver = diff.get("primary_driver")

        # 5) classify dynamics
        try:
            dynamics = classify_dynamics(
                gene_fates_summary, alignments, parent_card["genes"],
            )
        except Exception as e:
            dynamics = "Unknown"

        # 6) human-readable diagnostics
        fate_counts = {}
        for f in gene_fates_summary.values():
            fate_counts[f] = fate_counts.get(f, 0) + 1
        novel_count = sum(1 for a in alignments if a.get("fate") == "NOVEL")
        lines = [
            f"inferred_dynamics: {dynamics}",
            f"parent genes: {len(parent_card['genes'])}, proposed genes: {len(proposed_card['genes'])}",
            f"fates from parent's perspective: {fate_counts}",
            f"novel genes in proposal: {novel_count}",
        ]
        # heuristic feedback
        if dynamics == "Niche Competition":
            lines.append("⚠ no inheritance — proposal seems disconnected from parent")
        elif dynamics == "Mutation" and fate_counts.get("INHERITED", 0) >= 3:
            lines.append("✓ healthy incremental mutation: most parent genes inherited or mutated")
        elif dynamics == "Adaptive Radiation":
            lines.append("ℹ mechanism kept but niche shifted — re-targeting the same method")
        elif dynamics == "Speciation":
            lines.append("ℹ mechanism replaced while niche kept — new method on same problem")
        elif novel_count >= 3 and fate_counts.get("INHERITED", 0) == 0:
            lines.append("⚠ proposal introduces many novel genes with NO inheritance — may be too disconnected")

        return DiffResult(
            parent_id=parent_id,
            n_genes_parent=len(parent_card["genes"]),
            n_genes_proposed=len(proposed_card["genes"]),
            gene_fates_summary=gene_fates_summary,
            inferred_dynamics=dynamics,
            primary_driver=primary_driver,
            alignments=alignments,
            diagnostics="\n".join(lines),
        )


def format_diff_observation(result: DiffResult) -> str:
    """The text the agent sees as the genome_diff tool observation."""
    if result.error:
        return f"[error]: genome_diff failed: {result.diagnostics}"
    return f"[result]:\n{result.diagnostics}"


if __name__ == "__main__":
    tool = GenomeDiffTool()
    # use the same perovskite paper as parent
    parent = "oa:W4311930535"
    # a deliberately-disconnected proposal
    bad_proposal = {
        "mechanism_genome": "Reinforcement learning for video game playing using deep neural networks.",
        "niche_genome": "Atari game playing.",
        "limitation_genome": "Reward sparsity.",
        "observation_genome": "Higher score than humans.",
        "delta_genome": "Uses neural networks.",
        "claim_genome": "RL can play games.",
    }
    # a closely-related follow-up
    good_proposal = {
        "mechanism_genome": "Extend the stress-normalized indicator with a Bayesian model that predicts long-term degradation curves from short-term accelerated stress tests, calibrated on the open-stability dataset.",
        "niche_genome": "Accelerated stability testing of perovskite solar cells with cross-condition comparability.",
        "limitation_genome": "Bayesian calibration may overfit when stress conditions are sparse.",
        "observation_genome": "Calibrated Bayesian indicator improves predictive accuracy over the simple stress-normalized indicator by 20%.",
        "delta_genome": "Where the prior work proposed a static stress-normalized indicator, this work introduces a Bayesian dynamic predictor of degradation.",
        "claim_genome": "Bayesian dynamic stability prediction is more accurate than static stress normalization.",
    }

    for label, prop in [("bad", bad_proposal), ("good", good_proposal)]:
        print(f"=== diff parent={parent} vs {label} proposal ===")
        res = tool.diff(parent, prop)
        print(format_diff_observation(res))
        print()
