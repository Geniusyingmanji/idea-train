"""Composite reward for agentic-OPD trajectories.

  R_total = α_lineage · R_recall            # gold-ancestor hits
          + α_struct · R_struct             # Layer-1 PES (deterministic)
          + α_arena · R_arena_rank          # Layer-2 PES (tournament, optional)
          + α_efficiency · R_efficiency     # tool-call budget penalty
          + α_format · R_format             # propose emitted + well-formed
          # KL_ref handled in trainer loop, not here

Each component returns a scalar in roughly [-1, 1] or [0, 1]. Arena rank is
the only one that's group-relative; the others are absolute per trajectory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..structural import compute_struct
from .trajectory import Trajectory


@dataclass
class AgenticRewardConfig:
    """Reward weights — wire to trainer via argparse."""
    alpha_lineage: float = 0.3      # gold-ancestor recall; gold labels are sparse, so low weight
    alpha_struct: float = 0.5       # Layer-1 PES (deterministic) — main signal for proposal quality
    alpha_arena: float = 0.7        # Layer-2 tournament — kicks in when judge enabled
    alpha_efficiency: float = 0.1   # tool budget
    alpha_format: float = 0.3       # propose + read + no-malformed — critical for cold-start
    # efficiency budget: penalty kicks in past T tool calls
    efficiency_budget: int = 4
    # format: missing propose, malformed actions, etc.
    format_penalty_unit: float = 0.5


@dataclass
class TrajReward:
    R_total: float                # primary scalar; advantage = (R_total − μ) / σ
    R_lineage: float
    R_struct: float
    R_arena: float                # filled in by group tournament step (NOT here)
    R_efficiency: float
    R_format: float
    diagnostics: dict = field(default_factory=dict)


def compute_lineage_recall(traj: Trajectory, gold_lineage: list[str]) -> float:
    """Fraction of gold ancestors the agent successfully `read` during the trajectory."""
    if not gold_lineage:
        return 0.0
    read_set = set(traj.read_paper_ids)
    hits = sum(1 for g in gold_lineage if g in read_set)
    return hits / len(gold_lineage)


def compute_format_score(traj: Trajectory) -> float:
    """Higher is better. [-1, +1] roughly.

      +1.0 if propose emitted + at least one read
      -0.5 per malformed action
      -0.5 if no propose emitted
      -0.3 if propose emitted but missing fields
    """
    s = 0.0
    if traj.propose_emitted:
        s += 1.0
    else:
        s -= 0.5
    if traj.read_count >= 1:
        s += 0.5
    else:
        s -= 0.3
    s -= 0.5 * traj.malformed_count
    # check proposal completeness
    if traj.final_proposal:
        n_filled = sum(1 for v in traj.final_proposal.values() if str(v).strip())
        if n_filled < 4:
            s -= 0.3
    return max(-1.5, min(1.5, s))


def compute_efficiency(traj: Trajectory, budget: int = 4) -> float:
    """Penalty for going over budget. [0 down to -inf]."""
    over = max(0, (traj.search_count + traj.read_count) - budget)
    return -0.5 * over


def proposal_text_for_struct(final_proposal: dict | None) -> str:
    """Flatten a gene_genome dict into a single text string for struct comparison."""
    if not final_proposal:
        return ""
    bits = []
    for k in ("mechanism_genome", "niche_genome", "delta_genome",
              "limitation_genome", "claim_genome", "observation_genome"):
        v = final_proposal.get(k, "")
        if v:
            bits.append(str(v))
    return " ".join(bits)


def compute_trajectory_reward(
    traj: Trajectory,
    *,
    gold_lineage: list[str],
    parent_card: dict | None,            # for struct computation
    config: AgenticRewardConfig,
    arena_advantage: float = 0.0,         # filled in by group-tournament step
) -> TrajReward:
    """Return per-trajectory reward components + total."""
    # lineage recall (the agent signal)
    R_lineage = compute_lineage_recall(traj, gold_lineage)

    # struct (Layer-1 PES on the final proposal)
    proposal_text = proposal_text_for_struct(traj.final_proposal)
    if proposal_text and parent_card:
        struct = compute_struct(proposal_text, parent_card)
        R_struct = struct.s
        struct_dx = {
            "inheritance_match": struct.inheritance_match,
            "limitation_chain": struct.limitation_chain,
            "balanced_novelty": struct.balanced_novelty,
            "raw_sim": struct.raw_inherit_sim,
        }
    else:
        R_struct = 0.0
        struct_dx = {}

    R_format = compute_format_score(traj)
    R_efficiency = compute_efficiency(traj, config.efficiency_budget)

    R_arena = arena_advantage

    R_total = (
        config.alpha_lineage * R_lineage
        + config.alpha_struct * R_struct
        + config.alpha_arena * R_arena
        + config.alpha_efficiency * R_efficiency
        + config.alpha_format * R_format
    )

    return TrajReward(
        R_total=R_total,
        R_lineage=R_lineage,
        R_struct=R_struct,
        R_arena=R_arena,
        R_efficiency=R_efficiency,
        R_format=R_format,
        diagnostics={
            "n_search": traj.search_count,
            "n_read": traj.read_count,
            "n_malformed": traj.malformed_count,
            "propose_emitted": traj.propose_emitted,
            "lineage_hits": [g for g in gold_lineage if g in traj.read_paper_ids],
            **struct_dx,
        },
    )


if __name__ == "__main__":
    # Smoke: hand-build a fake trajectory and score it
    from .trajectory import Trajectory, ActionStep
    traj = Trajectory(
        prompt_id="smoke", topic="diffusion molecule", discipline="cs",
        read_paper_ids=["paper:foo:2020"],
        search_count=2, read_count=1, propose_emitted=True,
        final_proposal={
            "mechanism_genome": "Physics-aware diffusion with energy head",
            "niche_genome": "molecular generation",
            "delta_genome": "adds physical validity",
            "limitation_genome": "high compute cost",
            "claim_genome": "improves over baseline diffusion",
            "observation_genome": "expected 10% improvement",
        },
    )
    cfg = AgenticRewardConfig()
    parent_card = {
        "title": "Conditional Diffusion for Molecule Generation",
        "abstract": "...",
        "limitation": "no physical validity constraint",
        "mechanism_genome": "Conditional diffusion U-Net with classifier-free guidance"
    }
    r = compute_trajectory_reward(
        traj, gold_lineage=["paper:foo:2020", "paper:bar:2018"],
        parent_card=parent_card, config=cfg,
    )
    print(f"R_total = {r.R_total:+.3f}")
    print(f"  lineage   = {r.R_lineage:.3f} (hits 1/2)")
    print(f"  struct    = {r.R_struct:.3f}")
    print(f"  format    = {r.R_format:+.3f}")
    print(f"  efficiency= {r.R_efficiency:+.3f}")
    print(f"  arena     = {r.R_arena:+.3f}  (group-relative, filled later)")
    print(f"  diagnostics: {r.diagnostics}")
