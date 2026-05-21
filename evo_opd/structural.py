"""Structural reward — Layer-1 PES signal, no LLM, schema-agnostic.

This is the missing half of v6's reward composition. The IdeaEvolving paper's
PES has two layers:

  Layer 1 (deterministic): genome-similarity / inheritance-match / novelty-
                            distance computed from the parent paper's gene-card
                            and the proposal text. NO API calls.

  Layer 2 (LLM judge): multi-dim rubric on Heredity/Variation/Selection. v6's
                       tournament/pointwise judge implements this.

v6's first cut only did Layer 2. This module adds Layer 1 — deterministic,
free, and **schema-agnostic** so it transfers to SGI-Bench and other open-ended
idea benchmarks (which use their own schemas like
implementation_steps/related_work/etc).

The signal has 3 sub-scores in [0, 1]:

  inheritance_match — how much of the proposal references the parent (lexical
                      Jaccard-ish overlap with parent's content)

  limitation_chain  — does the proposal's stated limitation/motivation lexically
                      align with the parent's limitations (signals "addresses
                      parent's gap")

  balanced_novelty  — Gaussian-bumped novelty distance: rewards moderate
                      novelty (0.4-0.7 lexical distance from parent), penalizes
                      both pure copy (≈0 distance) and disconnected gibberish
                      (≈1 distance)

Combined as a simple mean → scalar in [0, 1] → group z-normalized → fed to
the per-token reward at weight λ_struct.

All three use the same `lexical_similarity()` primitive ported from
IdeaEvolving/agent/genome_differ.py — a token-overlap geometric-mean (cosine
on raw counters), with science-text-aware stop-word removal.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

# Ported from IdeaEvolving/agent/genome_differ.py - DO NOT MODIFY WITHOUT CHECKING
# the upstream lexical_similarity definition. Identical Jaccard-ish formula.

STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "as", "is", "are", "be", "this", "that", "it", "its", "their",
    "from", "into", "using", "used", "use", "uses", "via", "through",
    "paper", "method", "model", "approach", "system", "framework",
    # extras for generic open-ended scientific text:
    "novel", "new", "propose", "proposed", "proposal", "study", "studies",
    "research", "work", "results", "show", "shows", "shown",
}

_TOK_RE = re.compile(r"[A-Za-z][A-Za-z0-9+\-]{2,}")


def tokenize(text: str | None) -> Counter:
    if not text:
        return Counter()
    words = [w.lower() for w in _TOK_RE.findall(str(text))
             if w.lower() not in STOP_WORDS]
    return Counter(words)


def lexical_similarity(a: str | None, b: str | None) -> float:
    """Counter-cosine over tokens (range [0, 1]).

    Identical formula to IdeaEvolving's genome_differ.lexical_similarity:
        sim = sum(min(ca, cb)) / sqrt(|ca| * |cb|)
    """
    ca, cb = tokenize(a), tokenize(b)
    if not ca or not cb:
        return 0.0
    inter = sum((ca & cb).values())
    denom = (sum(ca.values()) * sum(cb.values())) ** 0.5
    return inter / denom if denom else 0.0


# --------------------------------------------------------------------------
# Generic parent-context extractor — works across schemas
# --------------------------------------------------------------------------

def extract_parent_content(parent_card: dict | None) -> dict:
    """Extract parent text fields used for structural comparison.

    Generic enough to handle:
      - IdeaEvolving GenomeCard fields (driver_genome, mechanism_genome, ...)
      - SGI-Bench question dict (core_idea, related_work, ...)
      - Plain {title, abstract, limitation} dicts
    """
    if not parent_card:
        return {"all_text": "", "limitation": "", "core": ""}
    out = {}

    # gather every string field into 'all_text'
    bits: list[str] = []

    def _walk(obj):
        if isinstance(obj, str):
            bits.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(parent_card)
    out["all_text"] = " ".join(bits)

    # limitation/gap field — common name variants
    for key in ("limitation", "limitation_genome", "gap", "limitations",
                "open_problem", "weakness"):
        v = parent_card.get(key)
        if v and isinstance(v, str):
            out["limitation"] = v
            break
    else:
        # nested: idea_genome.limitation_genome
        ig = parent_card.get("idea_genome") if isinstance(parent_card, dict) else None
        if isinstance(ig, dict):
            out["limitation"] = ig.get("limitation_genome", "") or ig.get("limitation", "")
        else:
            out["limitation"] = ""

    # core mechanism / driver / abstract — used for novelty distance
    for key in ("mechanism", "mechanism_genome", "core_idea", "abstract",
                "driver_genome", "title"):
        v = parent_card.get(key) if isinstance(parent_card, dict) else None
        if v and isinstance(v, str):
            out["core"] = v
            break
    else:
        out["core"] = out["all_text"][:500]

    return out


# --------------------------------------------------------------------------
# Structural sub-scores
# --------------------------------------------------------------------------

def inheritance_match(proposal_text: str, parent_text: str,
                       *, target: float = 0.35, sigma: float = 0.20) -> float:
    """Gaussian-bumped lexical overlap with parent text. [0, 1].

    Pure copy (sim=1) and pure disconnect (sim=0) both score 0. Moderate
    parent grounding (~35% token overlap) is the sweet spot — the proposal
    mentions parent's concepts without regurgitating them. Same shape as
    balanced_novelty but with a different target.
    """
    sim = lexical_similarity(proposal_text, parent_text)
    return math.exp(-((sim - target) ** 2) / (2 * sigma ** 2))


def raw_inherit_sim(proposal_text: str, parent_text: str) -> float:
    """Raw lexical sim (no Gaussian); kept for diagnostic logging."""
    return lexical_similarity(proposal_text, parent_text)


def limitation_chain(proposal_text: str, parent_limitation: str) -> float:
    """How much the proposal addresses parent's stated limitation. [0, 1]."""
    if not parent_limitation:
        return 0.0
    return lexical_similarity(proposal_text, parent_limitation)


def balanced_novelty(proposal_core: str, parent_core: str,
                      *, target: float = 0.55, sigma: float = 0.25) -> float:
    """Gaussian-bumped novelty: rewards moderate distance from parent.

    Reasoning: pure copies score 0 on novelty distance (sim ≈ 1) — bad.
    Pure gibberish/disconnected ideas score 1 on novelty distance (sim ≈ 0) —
    also bad. A good follow-up paper is moderately distant: builds on parent
    while introducing new content. We model the ideal as a Gaussian bump
    around `target` lexical-distance.

    Output ∈ [0, 1].
    """
    if not proposal_core or not parent_core:
        return 0.0
    sim = lexical_similarity(proposal_core, parent_core)
    dist = 1.0 - sim
    # Gaussian, normalized so peak = 1.0 at dist = target
    return math.exp(-((dist - target) ** 2) / (2 * sigma ** 2))


# --------------------------------------------------------------------------
# Combined struct score + group z-normalization
# --------------------------------------------------------------------------

@dataclass
class StructScore:
    s: float                       # combined [0, 1]
    inheritance_match: float       # Gaussian-bumped (peak at moderate sim)
    limitation_chain: float        # linearly positive (addressing limitation = good)
    balanced_novelty: float        # Gaussian-bumped (peak at moderate distance)
    raw_inherit_sim: float = 0.0   # diagnostic only


def compute_struct(proposal_text: str, parent_card: dict | None,
                   *, w_inherit: float = 1.0, w_limit: float = 0.7,
                   w_novelty: float = 1.0) -> StructScore:
    """Compute the 3-piece structural score for one rollout.

    All three sub-scores ∈ [0, 1]. inheritance_match and balanced_novelty are
    Gaussian-bumped so pure copies AND disconnected outputs are both
    penalized. limitation_chain is linear because addressing the parent's
    stated limitation is monotonically good.
    """
    if not proposal_text:
        return StructScore(0.0, 0.0, 0.0, 0.0)
    pc = extract_parent_content(parent_card)
    raw_sim = raw_inherit_sim(proposal_text, pc["all_text"])
    i_m = inheritance_match(proposal_text, pc["all_text"])
    l_c = limitation_chain(proposal_text, pc["limitation"]) if pc["limitation"] else 0.0
    b_n = balanced_novelty(proposal_text, pc["core"]) if pc["core"] else 0.0

    total_w = w_inherit + w_limit + w_novelty
    s = (w_inherit * i_m + w_limit * l_c + w_novelty * b_n) / total_w
    return StructScore(
        s=max(0.0, min(1.0, s)),
        inheritance_match=i_m,
        limitation_chain=l_c,
        balanced_novelty=b_n,
        raw_inherit_sim=raw_sim,
    )


def group_struct_zscore(scores: list[float]) -> list[float]:
    """GRPO-style z-normalization of group struct scores."""
    K = len(scores)
    if K <= 1:
        return [0.0] * K
    mu = sum(scores) / K
    var = sum((s - mu) ** 2 for s in scores) / K
    sigma = max(var ** 0.5, 1e-6)
    return [(s - mu) / sigma for s in scores]


if __name__ == "__main__":
    # Smoke: parent + 4 candidates spanning copy / good / moderate / gibberish
    parent = {
        "title": "Conditional Diffusion for Molecule Generation",
        "abstract": (
            "We propose a conditional diffusion model for 3D molecular "
            "generation. The model uses a U-Net backbone with classifier-"
            "free guidance to generate drug-like molecules conditioned on "
            "protein pockets."
        ),
        "limitation": (
            "The current model has no physical validity constraint and "
            "frequently generates non-physical conformations."
        ),
        "mechanism_genome": (
            "U-Net based conditional diffusion with classifier-free guidance "
            "trained on QM9 and GEOM-Drugs."
        ),
    }
    copy_paste = parent["abstract"] + " " + parent["mechanism_genome"]
    good = (
        "Physics-aware conditional diffusion for molecule generation with a "
        "learned energy-based head that rejects non-physical conformations "
        "during the reverse pass, trained on QM9 and GEOM-Drugs."
    )
    moderate = (
        "Conditional GFlowNet for molecule generation with binding affinity "
        "reward, trained on protein-pocket conditioned datasets."
    )
    gibberish = (
        "Quantum computing for distributed databases using blockchain consensus."
    )

    print(f"{'candidate':<12} {'raw_sim':>8} {'inherit':>8} {'limit':>8} {'novelty':>8} {'TOTAL':>8}")
    for label, text in [
        ("copy_paste", copy_paste),
        ("good",       good),
        ("moderate",   moderate),
        ("gibberish",  gibberish),
    ]:
        out = compute_struct(text, parent)
        print(f"{label:<12} "
              f"{out.raw_inherit_sim:>8.3f} "
              f"{out.inheritance_match:>8.3f} "
              f"{out.limitation_chain:>8.3f} "
              f"{out.balanced_novelty:>8.3f} "
              f"{out.s:>8.3f}")

    # z-norm test
    scores = [compute_struct(t, parent).s for t in [copy_paste, good, moderate, gibberish]]
    print(f"\nz-norm advantage: {[round(z, 2) for z in group_struct_zscore(scores)]}")
