"""Seeded single-elimination tournament for evo-OPD v6 arena-rank reward.

Algorithm (from ArenaRL, arXiv 2601.06487 §4.2):
  Phase 1 (Seeding):  anchor vs each of the (K-1) exploratory candidates → seed ranks.
                      Anchor is the first candidate (in our setting, the highest-temp=0
                      / greedy rollout if available; otherwise the first sampled
                      rollout). Cost: K-1 pairwise judge calls, all parallelizable.

  Phase 2 (Elim):     bracket = pair (seed 1, seed K), (seed 2, seed K-1), ... — i.e.
                      worst-seed-faces-best. Winners advance. Each round halves the
                      field. Within each round, matches are parallelizable; across
                      rounds they are sequential. Cost: K-1 calls total (rounds 1..r
                      sum to K-1 for any K).

  Final rank: tournament rank == elimination round at which the candidate lost
              (and the champion gets rank=1). Ties broken by total per-dim wins
              over the entire tournament.

Total cost: 2(K-1) judge calls per prompt. For K=8 → 14 calls. The judge calls
within each round are parallelized via judge_pairs_parallel (workers=8 default).
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence

from ..judges.pairwise_pes import (
    JUDGE_DIMS, PairwiseResult, judge_pairs_parallel,
)

log = logging.getLogger(__name__)


@dataclass
class TournamentResult:
    K: int
    tournament_rank: list[int]    # length K; rank[i] ∈ {1..K}; 1 = champion
    quantile_reward: list[float]  # length K; (K - rank) / (K - 1), in [0, 1]
    z_advantage: list[float]      # length K; z-normalized quantile within group
    win_counts: list[int]         # length K; total per-dim wins across tournament
    pair_results: list[PairwiseResult] = field(default_factory=list)
    n_judge_calls: int = 0


def _quantile_then_zscore(ranks: list[int]) -> tuple[list[float], list[float]]:
    """ranks: 1=best, K=worst → returns (quantile_in_[0,1], z_score)."""
    K = len(ranks)
    if K <= 1:
        return [1.0] * K, [0.0] * K
    quant = [(K - r) / (K - 1) for r in ranks]
    mu = sum(quant) / K
    var = sum((q - mu) ** 2 for q in quant) / K
    sigma = max(var ** 0.5, 1e-6)
    z = [(q - mu) / sigma for q in quant]
    return quant, z


def _seed_phase(
    prompt: str, candidates: Sequence[str],
    *, prompt_id: str, anchor_idx: int = 0,
    client=None, workers: int = 8,
) -> tuple[dict[int, int], list[PairwiseResult]]:
    """Return (seed_rank_by_idx, pair_results). seed_rank: 1=best,K=worst."""
    K = len(candidates)
    other_idxs = [i for i in range(K) if i != anchor_idx]
    pairs = []
    for i in other_idxs:
        pairs.append(dict(
            pair_id=f"{prompt_id}::seed::a{anchor_idx}_vs_{i}",
            prompt_id=prompt_id, idx_a=anchor_idx, idx_b=i,
            prompt=prompt,
            idea_a=candidates[anchor_idx],
            idea_b=candidates[i],
        ))
    results = judge_pairs_parallel(pairs, client=client, workers=workers)

    # for each non-anchor i, determine wins against anchor + per-dim score
    # anchor_score = +1 per dim where anchor (A) won, -1 where i (B) won, 0 ties
    anchor_score = 0
    i_score_by_idx: dict[int, int] = {}
    for r in results:
        s = r.a_wins_count - r.b_wins_count  # A=anchor, B=i; positive ⇒ anchor better
        anchor_score += s  # accumulate anchor's net wins across all matches
        i_score_by_idx[r.idx_b] = -s  # i's score against anchor (positive ⇒ i better)

    # Build full ranking: anchor's seed comes from its avg margin; others use their margin
    margin_by_idx = {anchor_idx: anchor_score / max(len(other_idxs), 1)}
    for i, sc in i_score_by_idx.items():
        margin_by_idx[i] = float(sc)  # not averaged; comparable since each i has 1 match
    # Actually need to be careful: anchor played K-1 matches, others played 1. Use:
    #   - anchor_margin = mean per-dim margin across all its matches
    #   - each i's margin = per-dim margin in the 1 match
    # Both have units of "per-dim net wins", so they're comparable enough for seeding.
    ranked_idxs = sorted(range(K), key=lambda i: -margin_by_idx[i])
    seed_rank = {idx: r + 1 for r, idx in enumerate(ranked_idxs)}
    return seed_rank, results


def _elim_phase(
    prompt: str, candidates: Sequence[str], seed_rank: dict[int, int],
    *, prompt_id: str, client=None, workers: int = 8,
) -> tuple[list[int], list[PairwiseResult]]:
    """Run single-elim. Returns (final_rank, all_match_results).

    Tournament rank = K - (round at which they got eliminated)
    Champion gets rank=1; loser of finals = 2; semi-finals losers = 3 (tie);
    quarter-final losers = 5 (tie), ...
    Ties broken later by win counts.
    """
    K = len(candidates)
    # ranks: idx → tournament rank (lower = better). Initialize all to K (worst).
    ranks: dict[int, int] = {i: K for i in range(K)}
    # working_seeds: list of (seed_rank, idx) currently alive. Sort by seed (best first).
    alive = sorted(range(K), key=lambda i: seed_rank[i])  # best-seed first

    all_pair_results: list[PairwiseResult] = []
    round_no = 0
    # rank-of-loser per round = (# of survivors at start of round) + 1
    # actually simpler: loser ranks for a round = current alive count // 2 + 1 onwards
    # We assign loser-rank = number of remaining survivors after this round + 1
    while len(alive) > 1:
        round_no += 1
        n = len(alive)
        # pair best vs worst: alive[0] vs alive[-1], alive[1] vs alive[-2], ...
        pairs = []
        match_pairs = []  # tuples of (seedA_idx, seedB_idx) in alive
        for i in range(n // 2):
            a_idx = alive[i]
            b_idx = alive[n - 1 - i]
            match_pairs.append((a_idx, b_idx))
            pairs.append(dict(
                pair_id=f"{prompt_id}::elim_r{round_no}::{a_idx}_vs_{b_idx}",
                prompt_id=prompt_id, idx_a=a_idx, idx_b=b_idx,
                prompt=prompt,
                idea_a=candidates[a_idx], idea_b=candidates[b_idx],
            ))
        # odd-K: middle one gets a bye
        bye_idx = alive[n // 2] if n % 2 == 1 else None

        results = judge_pairs_parallel(pairs, client=client, workers=workers)
        all_pair_results.extend(results)

        # determine winners + losers
        next_alive = []
        # loser-rank for this round: equals (size after this round) + 1
        next_size = (n + 1) // 2  # accounting for byes
        loser_rank = next_size + 1
        for (a, b), r in zip(match_pairs, results):
            if r.overall == "A":
                next_alive.append(a)
                ranks[b] = loser_rank
            elif r.overall == "B":
                next_alive.append(b)
                ranks[a] = loser_rank
            else:
                # tie → break by per-dim wins; if still tied, break by seed
                if r.a_wins_count > r.b_wins_count:
                    next_alive.append(a); ranks[b] = loser_rank
                elif r.b_wins_count > r.a_wins_count:
                    next_alive.append(b); ranks[a] = loser_rank
                else:
                    # tied on dim count too → better seed wins
                    if seed_rank[a] < seed_rank[b]:
                        next_alive.append(a); ranks[b] = loser_rank
                    else:
                        next_alive.append(b); ranks[a] = loser_rank
        if bye_idx is not None:
            next_alive.append(bye_idx)
        alive = sorted(next_alive, key=lambda i: seed_rank[i])

    # last one alive is the champion
    if alive:
        ranks[alive[0]] = 1

    final_rank = [ranks[i] for i in range(K)]
    return final_rank, all_pair_results


def run_tournament(
    prompt: str, candidates: Sequence[str],
    *, prompt_id: str, client=None, workers: int = 8,
    anchor_idx: int = 0,
) -> TournamentResult:
    """Full seeded single-elimination tournament. Returns rank + reward."""
    K = len(candidates)
    if K < 2:
        return TournamentResult(
            K=K, tournament_rank=[1] * K,
            quantile_reward=[1.0] * K, z_advantage=[0.0] * K,
            win_counts=[0] * K, n_judge_calls=0,
        )

    # phase 1: seeding
    seed_rank, seed_pairs = _seed_phase(
        prompt, candidates, prompt_id=prompt_id,
        anchor_idx=anchor_idx, client=client, workers=workers,
    )

    # phase 2: elimination
    final_rank, elim_pairs = _elim_phase(
        prompt, candidates, seed_rank,
        prompt_id=prompt_id, client=client, workers=workers,
    )

    # win counts: count per-dim wins across all pairs (used for tie-break)
    win_counts = [0] * K
    for r in seed_pairs + elim_pairs:
        win_counts[r.idx_a] += r.a_wins_count
        win_counts[r.idx_b] += r.b_wins_count

    # Break rank ties using win_counts (more wins = better → lower rank number).
    # Single-elim assigns tied round-losers the same rank; we re-rank with win counts
    # to get a strict [1..K] ordering. Critical for GRPO advantage (z-score wants
    # distinct values across the group).
    ordered = sorted(range(K), key=lambda i: (final_rank[i], -win_counts[i]))
    strict_rank = [0] * K
    for new_rank, idx in enumerate(ordered, start=1):
        strict_rank[idx] = new_rank
    final_rank = strict_rank

    quant, z = _quantile_then_zscore(final_rank)

    return TournamentResult(
        K=K,
        tournament_rank=final_rank,
        quantile_reward=quant,
        z_advantage=z,
        win_counts=win_counts,
        pair_results=seed_pairs + elim_pairs,
        n_judge_calls=len(seed_pairs) + len(elim_pairs),
    )


if __name__ == "__main__":
    import time
    # smoke: K=4 with one obviously-best candidate
    prompt = (
        "Parent paper: 'A simple diffusion model for molecule generation.' "
        "Limitation: no physical validity. Propose a follow-up."
    )
    good = (
        '{"idea": "Physics-Aware Diffusion", '
        '"core_method": "Add MD-learned energy head to score denoising steps and reject non-physical confs. Train on QM9+GEOM-Drugs.", '
        '"limitation_addressed": "physical validity"}'
    )
    medium = (
        '{"idea": "Constrained Diffusion", '
        '"core_method": "Add a constraint that the molecule must be valid SMILES.", '
        '"limitation_addressed": "validity"}'
    )
    weak = (
        '{"idea": "Another molecule generator", '
        '"core_method": "Use diffusion.", '
        '"limitation_addressed": "molecule generation"}'
    )
    awful = (
        '{"idea": "Make better molecules", '
        '"core_method": "Train more.", '
        '"limitation_addressed": "stuff"}'
    )
    candidates = [good, medium, weak, awful]

    t0 = time.time()
    out = run_tournament(prompt, candidates, prompt_id="smoke-tourney", workers=4)
    print(f"K={out.K} done in {time.time() - t0:.1f}s, {out.n_judge_calls} judge calls")
    print(f"  tournament_rank: {out.tournament_rank}  (expected ~[1, 2, 3, 4])")
    print(f"  quantile_reward: {[f'{q:.2f}' for q in out.quantile_reward]}")
    print(f"  z_advantage:     {[f'{z:+.2f}' for z in out.z_advantage]}")
    print(f"  win_counts:      {out.win_counts}")
    print(f"  pair count: {len(out.pair_results)}")
    for r in out.pair_results[:6]:
        print(f"    {r.pair_id}: overall={r.overall} a={r.a_wins_count} b={r.b_wins_count}")
