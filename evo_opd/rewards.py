"""evo-OPD per-token reward — glues parser + verifier + lineage + teacher KL.

Implements the equation from lv_opd_plan.md §3.1(d):

  r_t = -α(φ(t)) · kl_t                                         (field-weighted reverse-KL)
        + α(φ(t)) · λ_v · (v(y) - v̄)                            (verifier reward, sparse, broadcast w/ α)
        + α(φ(t)) · λ_c · 1[parent] · (c(y,p) - c̄)               (lineage consistency)

Where:
  kl_t = log π_θ(y_t|·) - log π_T(y_t|·)
  v̄, c̄ are EMA running means (variance reduction)
  φ(t) is the token's schema role from parser
  α(·) is from schemas.FIELD_WEIGHT

Verifier-anchored decoupling (component B): on gold_answer tokens, α=0 so the
reverse-KL is zeroed; only the verifier reward shapes those tokens.
"""
from __future__ import annotations

from dataclasses import dataclass

from .lineage import LineageScore, compute_lineage
from .parser import parse_rollout, tokenize_with_roles
from .schemas import FIELD_WEIGHT
from .verifier import VerifierScore, compute_verifier


@dataclass
class EMAScalar:
    """Exponential moving average with bias correction."""
    alpha: float = 0.05
    _value: float = 0.0
    _count: int = 0

    def update(self, x: float) -> float:
        self._count += 1
        self._value = self.alpha * x + (1 - self.alpha) * self._value
        # bias correction
        return self._value / (1 - (1 - self.alpha) ** self._count)


@dataclass
class EvoOPDRewardConfig:
    lambda_v: float = 0.5
    lambda_c: float = 0.3
    field_weights: dict[str, float] | None = None
    # EMA decay for v̄, c̄ baselines
    ema_alpha: float = 0.05


@dataclass
class PerTokenReward:
    rewards: list[float]               # length = n_tokens
    alphas: list[float]                # length = n_tokens; for diagnostics
    phi: list[str]                     # length = n_tokens
    fields: list[str | None]
    verifier: VerifierScore
    lineage: LineageScore | None
    kl_term_mean: float
    verifier_term_mean: float
    lineage_term_mean: float


class EvoOPDReward:
    """Compute per-token rewards for one rollout.

    Persistent state: EMA baselines for v and c (advantage standardization).
    """

    def __init__(self, config: EvoOPDRewardConfig | None = None) -> None:
        self.cfg = config or EvoOPDRewardConfig()
        self.field_weights = self.cfg.field_weights or FIELD_WEIGHT
        self.v_ema = EMAScalar(alpha=self.cfg.ema_alpha)
        self.c_ema = EMAScalar(alpha=self.cfg.ema_alpha)

    def __call__(
        self,
        text: str,
        per_token_kl: list[float],  # kl_t = log π_θ - log π_T, length n_tokens
        phi_per_token: list[str] | None = None,
        fld_per_token: list[str | None] | None = None,
        task_type: str | None = None,
        source_text: str | None = None,
        gold_answer: dict | None = None,
        parent_card: dict | None = None,
    ) -> PerTokenReward:
        # 1. parse rollout once (used by verifier + tokenization helper)
        # 2. compute verifier
        v_score, _ = compute_verifier(text, task_type, source_text, gold_answer)
        v_adv = v_score.v - self.v_ema.update(v_score.v)

        # 3. compute lineage (only if parent provided)
        l_score: LineageScore | None = None
        c_adv = 0.0
        if parent_card is not None:
            l_score = compute_lineage(text, parent_card, task_type or "genome_diff_annotate")
            c_adv = l_score.c - self.c_ema.update(l_score.c)

        # 4. token-role tags must be supplied by caller (uses HF tokenizer offsets).
        # If not supplied, fall back to character-uniform tags from parse — useful for tests.
        n = len(per_token_kl)
        if phi_per_token is None or len(phi_per_token) != n:
            phi_per_token = ["unknown"] * n
            fld_per_token = [None] * n

        # 5. assemble per-token reward
        alphas: list[float] = []
        rewards: list[float] = []
        kl_terms: list[float] = []
        v_terms: list[float] = []
        c_terms: list[float] = []

        for phi, kl in zip(phi_per_token, per_token_kl):
            alpha = self.field_weights.get(phi, self.field_weights["unknown"])
            kl_t = -alpha * kl                          # -α · (log π_θ - log π_T)
            v_t = alpha * self.cfg.lambda_v * v_adv     # broadcast
            c_t = alpha * self.cfg.lambda_c * c_adv if l_score is not None else 0.0
            r = kl_t + v_t + c_t
            alphas.append(alpha)
            rewards.append(r)
            kl_terms.append(kl_t)
            v_terms.append(v_t)
            c_terms.append(c_t)

        return PerTokenReward(
            rewards=rewards,
            alphas=alphas,
            phi=phi_per_token,
            fields=fld_per_token or [None] * n,
            verifier=v_score,
            lineage=l_score,
            kl_term_mean=sum(kl_terms) / max(n, 1),
            verifier_term_mean=sum(v_terms) / max(n, 1),
            lineage_term_mean=sum(c_terms) / max(n, 1),
        )


# --------------------------------------------------------------------------
# Convenience helper for tests (no tokenizer required)
# --------------------------------------------------------------------------
def char_uniform_phi_tags(text: str, task_type: str | None = None) -> list[str]:
    """Build per-char φ tags from parse_rollout. Useful when no tokenizer is wired."""
    pr = parse_rollout(text, task_type)
    tags = ["unknown"] * len(text)
    for r in pr.regions:
        for i in range(r.start, min(r.end, len(text))):
            tags[i] = r.phi
    return tags


if __name__ == "__main__":
    # Smoke: simulate a teacher that strongly agrees with student except on
    # the dynamics_label, then verify reward correctly downweights gold_answer
    # tokens (α=0) and upweights dynamics_label tokens (α=2.0).

    text = '''```json
{"driver": "mechanism", "dynamics": "Adaptive Radiation"}
```'''
    n_chars = len(text)
    # toy: one "token" per character; uniform small kl=0.1 except boost in middle
    per_token_kl = [0.1] * n_chars
    phi = char_uniform_phi_tags(text, "T3-01_single_dynamics")

    reward = EvoOPDReward()
    out = reward(
        text=text,
        per_token_kl=per_token_kl,
        phi_per_token=phi,
        fld_per_token=[None] * n_chars,
        task_type="T3-01_single_dynamics",
        gold_answer={"driver": "mechanism", "dynamics": "Adaptive Radiation"},
    )
    print(f"verifier.v = {out.verifier.v:.3f}")
    print(f"kl_term_mean       = {out.kl_term_mean:+.4f}")
    print(f"verifier_term_mean = {out.verifier_term_mean:+.4f}")
    print(f"lineage_term_mean  = {out.lineage_term_mean:+.4f}")

    # show breakdown by phi tag
    from collections import defaultdict
    by_phi: dict[str, list[float]] = defaultdict(list)
    for p, r in zip(out.phi, out.rewards):
        by_phi[p].append(r)
    print("\nMean reward per φ tag:")
    for p, rs in sorted(by_phi.items()):
        print(f"  {p:<14} n={len(rs):>3}  α={FIELD_WEIGHT[p]:.2f}  mean_r={sum(rs)/len(rs):+.4f}")
