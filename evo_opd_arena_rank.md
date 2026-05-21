# evo-OPD v6: Arena-Rank Reward (Tournament-Style)

**Status:** spec — not yet implemented. Drafted 2026-05-19 in response to PES findings (v5 compressed −1.56 PES vs v3 SFT) and the ArenaRL paper (arXiv 2601.06487).

**One-line:** Replace the pointwise PES-judge reward with a **seeded single-elimination tournament rank** over the K rollouts in each GRPO group. Diagnostic-validated by ArenaRL: pointwise judge scores cause "discrimination collapse" in open-ended RL; relative ranking inside a group sidesteps this.

---

## 1. Why this matters for us — empirical failure of v5

Our 6-model PES study (n=150 each, 900 ideas total, see `eval/results/OVERNIGHT_REPORT.md`):

| Cluster | Models | PES | vs v3 SFT |
|---|---|---|---|
| Winner | v3 (SFT generalist) | 57.95 | — |
| Specialized ceiling | v8, evo-OPD-v4, evo-OPD-v5 | 56.4-56.7 | −1.3 to −1.6 (p<0.005) |
| Base | Qwen3-8B no-LoRA | 50.96 | −6.99 |

**Three independent specialization paths land in the same compressed cluster** — strict-format SFT (v8), T1-focused SFT (v10 marginally), and verifier-anchored RL (evo-OPD v4 and v5). The PES cost is paid in the first ~100 RL steps and saturates (v4 100-step ≈ v5 200-step on PES). Sub-dim breakdown confirms the cost lives in **creative dims**: `originality` 2.92 vs 3.13, `balanced_novelty` 2.95 vs 3.11, `mechanism_concreteness` 2.45 vs 2.57, `limitation_repair` 3.24 vs 3.43.

The root cause is structural: our current reward in evo-OPD v5 has no gradient toward novelty/originality. The verifier rewards schema and evidence, the lineage signal rewards predecessor-consistency, but **nothing pulls the policy toward creative deviation**. So RL trades creativity for closer fit to the deterministic signal.

This isn't unique to RL — v8 (pure SFT on strict completions) hits the same ceiling. The fix has to be at the **reward design** level, not the algorithm level.

---

## 2. Survey: open-ended RL without ground truth (2024-2026)

Below I summarise the relevant approaches, ranked by applicability to our setup.

### 2.1 ArenaRL (Qiang Zhang et al., arXiv 2601.06487, Jan 2026) — **most applicable**

- **Diagnosis**: pointwise scalar scoring on open-ended tasks causes "discrimination collapse" — judge ratings drift, scale is arbitrary, RL converges to a high-mean safe mode.
- **Fix**: for each prompt, generate K rollouts; rank them via pairwise tournament; convert rank to quantile reward; z-normalize within group as GRPO advantage.
- **Topology**: tried round-robin (K(K-1)/2), Swiss, single-elim, double-elim, **seeded single-elim** (chosen, O(K) comparisons). Phase 1: anchor seeding (greedy vs all exploratory, N-1 calls). Phase 2: bracketed elimination (N-1 calls). Ties broken by accumulated average score.
- **Rubric**: multi-dimensional, process-aware (judge looks at intermediate reasoning, not just final answer). For Open-DeepResearch: 7 dims (Framework, Tool Usage, Coverage, Relevance, Accuracy, Depth, Clarity).
- **Loss**: PPO-clipped with reference-policy KL penalty (same family as GRPO/GSPO).
- **Result**: beats GRPO and GSPO on Open-DeepResearch + Open-Travel. 73.9% human-agreement on tournament outcomes.

### 2.2 Other relevant lines

| Approach | Key idea | Why we don't go here first |
|---|---|---|
| **DPO / IPO / KTO** | Direct optim. from pairwise preferences, no separate reward model | Works on offline pair data; doesn't scale to within-group K-relative signal from a live judge. Could be a follow-up. |
| **RLHF (Bradley-Terry MLE)** | Fit a parametric reward model from pairwise data, then PPO against it | Needs a learned reward model (extra training stage). ArenaRL is simpler and matches BT performance per their ablation. |
| **Spin (Self-Play Fine-Tuning)** | Treat the prior checkpoint's outputs as "loser," current as "winner" | Requires a clear winner-vs-loser assumption. We have a 12-dim PES rubric, not a binary preference. |
| **RLAIF (Bai et al., Constitutional AI)** | Use AI judge for pairwise prefs, train reward model, PPO | Identical to RLHF after the preference collection step. ArenaRL skips the reward-model step. |
| **Process Reward Models (PRMs)** | Per-step reward from a learned model | Useful for multi-step reasoning; our open-ended task is single-shot. |
| **Lit-Search/RAG-RL** | Reward = downstream task success after using a tool | Doesn't apply — no downstream task in idea generation. |
| **MCTS-style search (AlphaZero)** | Tree search at inference, value-head training | Too expensive for free-form text generation. |
| **Diversity-bonus rewards** (Bowman 2025, novelty bonuses) | Add reward ∝ embedding-distance to past outputs | Complementary; can be added to ArenaRL as a tie-breaker. |

**Decision:** ArenaRL is the right port because (a) we already do GRPO-style multi-rollout sampling, so the K rollouts are free, (b) seeded single-elim only needs O(K) judge calls per prompt (we have Azure GPT-5.5 keyless), (c) the rubric maps 1:1 to PES sub-dims so we get a training signal in the same space we're evaluated on, (d) ArenaRL's PPO-clipped loss with ref-KL is drop-in compatible with our existing evo-OPD v3 trainer.

---

## 3. The port — evo-OPD v6 reward composition

### 3.1 Current reward (v5)

```
r_t = α · v_advantage_t          # verifier (schema + evidence + dynamics + judge_PES)
    + γ · c_advantage_t          # lineage-consistency
    − δ · KL[π_θ || π_ref]_t    # reference-policy KL anchor (β_kl_ref = 0.01)
```

where `v_advantage_t` and `c_advantage_t` are group-mean-normalized within the K=8 rollouts (GRPO style). **The `judge_PES` component inside `v` was pointwise (GPT-5.5 absolute rating).**

### 3.2 v6 reward — replace `judge_PES` with arena_rank

```
r_t = α · v_local_t              # LOCAL verifier only: schema + evidence + dynamics (drop pointwise judge)
    + β · arena_rank_advantage_t # NEW: tournament rank-based, replaces judge_PES
    + γ · c_advantage_t          # lineage-consistency (unchanged)
    − δ · KL[π_θ || π_ref]_t     # ref-KL anchor (unchanged)
```

**Hyperparameters (proposed defaults; subject to ablation):**
- `α = 0.25` (verifier still matters for format, but down from 1.0 in v5)
- `β = 0.50` (arena rank is the dominant signal)
- `γ = 0.20` (lineage as in v5)
- `δ = 0.01` (ref-KL as in v5)

### 3.3 `arena_rank_advantage_t` — the new piece

For each prompt `x` in the batch:

1. **Sample K=8 rollouts** `{y_1, ..., y_K}` from `π_θ(·|x)` (already happens in GRPO).
2. **Seed (Phase 1, O(K))**: pick the greedy-decoded `y_anchor` (or `y_1` from sampled rollouts as fallback) and pairwise-judge each `y_i` vs `y_anchor`. Sort by win-rate against anchor → seed ranking `seed_rank(y_i)`.
3. **Eliminate (Phase 2, O(K))**: bracket `(seed 1 vs seed K, seed 2 vs seed K-1, ...)`, winners advance; ties broken by tournament-internal average per-dim score. Final tournament rank `t_rank(y_i) ∈ {1..K}`.
4. **Quantile reward**: `r_arena(y_i) = 1 − (t_rank(y_i) − 1) / (K − 1) ∈ [0, 1]`.
5. **Z-normalize within group**: `arena_rank_advantage(y_i) = (r_arena(y_i) − μ_r) / (σ_r + ε)`.
6. **Broadcast to per-token**: every content-bearing token (non-boilerplate) gets the rollout-level advantage; boilerplate tokens get 0. Same per-token decomposition as v5's `v_advantage_t`.

### 3.4 Pairwise judge rubric

Use GPT-5.5 (Azure keyless) with the **PES sub-dim rubric** restricted to creative dims:

| Dim | Description (1-5 each, then averaged) |
|---|---|
| `originality` | How much does it deviate from obvious extensions of the parent paper? |
| `balanced_novelty` | Is the novelty *measured* (not gratuitous, not incremental)? |
| `mechanism_concreteness` | Is the proposed mechanism specified well enough to implement? |
| `limitation_repair` | Does the proposal address the parent paper's stated limitation? |
| `expected_impact` | Plausible to advance the field if successful? |

Judge prompt: "You see prompt X and two candidate proposals A and B. For each dim, which one is better? Output `{originality: A|B|tie, ...}`." Win = wins majority of dims; ties broken by per-dim average sub-scores requested as a tie-breaker secondary signal.

### 3.5 Cost estimate

| Setting | Judge calls/step | Per training run (200 steps × 4 prompts/batch) | Wall-clock at 8 workers (3-5s/call) |
|---|---|---|---|
| K=8 (default) | 14 × 4 = 56 | 11,200 | ~6-12 hr |
| K=4 (low-cost ablation) | 6 × 4 = 24 | 4,800 | ~2-5 hr |
| K=8, batch=8 (full GRPO) | 14 × 8 = 112 | 22,400 | ~12-24 hr |

The Azure deployment is keyless so no $ cost; only constraint is rate limits + wall-clock. Recommended: K=8 batch=4 for first run (matches v5 settings, ~6-12 hr).

### 3.6 What stays the same vs v5

- GRPO sampling, group structure, K=8 default
- Per-token role parser (boilerplate vs content-field vs gene_genome)
- Lineage-consistency signal `c(y, p)` from GeneTrace cards
- Reference-policy KL anchor (β_kl_ref = 0.01)
- PPO-clipped loss, AdamW LR=1e-5, weight_decay=0.1
- Eval cadence (every 50 steps GENE-Exam 200-instance smoke, GENE-Arena 30-instance PES)

---

## 4. Predicted outcomes & how we'd know it worked

### Success criteria (vs v5 baseline)

1. **PES preserved or improved**: target PES ≥ 57.5 (within v3's range, vs v5's 56.38)
2. **GENE-Exam T3 preserved**: target T3 ≥ 11.5 (vs v5's 12.9, accept small regression)
3. **Sub-dim creativity preserved**: `originality` ≥ 3.05, `balanced_novelty` ≥ 3.05 (vs v5's 2.92, 2.95)
4. **Diversity intact**: rollout-pair embedding distance (within-group, K=8) should NOT collapse over training — measure every 50 steps

### Failure modes to watch for

- **Tournament noise**: if K=8 single-elim has too much variance, rank reward will be a noisy signal. Mitigation: bump to K=16 or add a 2-judge ensemble (still Azure free).
- **Format regression**: dropping pointwise judge_PES means verifier weight α matters for keeping schema right. If schema-valid rate drops below 95% by step 50, bump α to 0.4.
- **Reward hacking on rank-only**: model might learn to make K rollouts maximally diverse (gibberish) to ace ranks. Mitigation: keep verifier as floor — invalid rollouts get rank=K automatically. Also, the lineage signal `c(y,p)` punishes nonsense.

---

## 5. Ablation matrix to ship in the paper

After v6 main run, ship 4 ablations to defend the design:

| Variant | Description | Hypothesis |
|---|---|---|
| **v6 (main)** | α=0.25, β=0.50, γ=0.20 | recovers PES, retains T3 |
| **v6-no-arena** | β=0, α=0.75 → v5 with new α | reproduces v5 collapse |
| **v6-arena-only** | β=1.0, α=γ=0 | rank reward alone enough? probably not — format will drift |
| **v6-K=4** | seeded single-elim with K=4 | cost-effective variant for the next group of researchers |
| **v6-pointwise** | α=0.25, β=0.50, but β-component uses pointwise PES judge (not tournament) | isolates "tournament structure" from "judge identity" — the key question for the paper |

The v6-pointwise ablation is the **must-have** comparison: if pointwise-PES at the same weight performs as well as tournament-rank, then the tournament is unnecessary scaffolding. If tournament wins, that's the strongest evidence we can offer that ArenaRL's central thesis (pointwise → discrimination collapse) generalizes to scientific idea generation.

---

## 6. Implementation plan & checkpoints

| Step | Artefact | Owner | ETA |
|---|---|---|---|
| 1. Pairwise judge implementation | `evo_opd/judges/pairwise_pes.py` — async GPT-5.5 calls, batched, retry-with-backoff | (drop-in next to existing `judges/`) | 0.5 day |
| 2. Tournament module | `evo_opd/trainer/tournament.py` — seeded single-elim, returns rank array | (new) | 0.5 day |
| 3. Reward integrator | extend `evo_opd_loop_v3.py` → `evo_opd_loop_v4_arena.py` to compute `arena_rank_advantage_t` and add to reward | (modify existing) | 1 day |
| 4. Smoke train (K=4, 20 steps) | sanity-check signal, latency, group diversity | (on 1× A100) | 0.5 day |
| 5. Full run (K=8, 200 steps) | v6 main checkpoint | (3× A100 student) | 1-2 days incl. judge time |
| 6. Eval + ablations | GENE-Exam + GENE-Arena PES on v6, v6-no-arena, v6-pointwise, v6-K=4 | (parallel on 4× A100) | 2 days |

**Total**: ~5-6 days for the full v6 cycle including 4 ablations. Compute fits within the 1-week evo-OPD budget already in `plan.md` §7.

---

## 7. Notes on judge methodology (carry over to paper)

- Pairwise judging is **more robust to judge drift** than pointwise: even if absolute scale shifts day-to-day, "A > B on dim X" is stable.
- We're locked to gpt-5.5 (the other 2 of 3 deployments — gpt-5.4, gpt-5.4-nano — return DeploymentNotFound on this Azure endpoint). Single-judge with paired-bootstrap analysis remains valid: judge variance affects within-trace noise, not across-participant bias.
- ArenaRL reports 73.9% human-agreement on tournament outcomes. We should validate this on a 50-pair sample for our domain before paper submission.
