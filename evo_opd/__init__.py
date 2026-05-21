"""evo-OPD: Evolutionary On-Policy Distillation for GENE-bench-style scientific idea reasoning.

Three deterministic components (no teacher, no training):
  - parser:    student rollout text -> typed regions (φ tags per token)
  - verifier:  v(y) ∈ [0,1] sparse correctness signal
  - lineage:   c(y, p) ∈ [0,1] self-supervised consistency vs parent gene-card

Combined into per-token rewards in `rewards.py` (to be added once teacher logits are wired).
"""
