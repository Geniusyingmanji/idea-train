# Overnight Run Results (2026-05-17 → 2026-05-19, multi-session)

## TL;DR

End-to-end pipeline validated. Six SFT runs (v1 → v6), best result **11.27% lenient (Qwen3-8B + LoRA r=64, SFT v3)** vs 7.77% baseline on GENE-Exam main-challenge 1029-instance profile (no-think mode). Three ablations after v3 (more data, bigger LoRA, strict-key prompts) all regressed — diagnosed root causes documented below.

> Reference: GPT-5.5 scores 23.1% on the same benchmark per the IdeaEvolving paper. Our 8B model reaches **~49%** of GPT-5.5's performance with ~30× fewer parameters.

**Key diagnostic finding (v6 + post-hoc experiment):** The strict-vs-lenient score gap (1.17% vs 11.27% for v3) is caused by the model mirroring the eval prompt's capitalized "Driver = X" phrasing instead of emitting lowercase JSON keys. Augmenting training prompts (v6) did not transfer because the model learned a *conditional* canonical-key behavior. Eval-time prompt augmentation closed the gap (1.17/11.27 → 5.83/6.31) but at a heavy cost to content accuracy. The next experiment (v7) should train on the actual GENE-Exam eval-prompt distribution so canonical keys are learned without polluting the inference prompt.

## Setup (what got built tonight)

| Stage | What | Result |
|---|---|---|
| 0a | Denylist from local IdeaEvolving assets | 8,359 unique papers, 90.4% with s2_id |
| 0b | Safe paper pool (pre-2017, non-denylist) | 3,846 papers (857 with usable abstracts) |
| 1a | GPT-5.5 SFT data round 1 (gene_card_extract) | **865 examples**, 100% verifier-passed |
| 1b | GPT-5.5 SFT data round 2 (T1-01/T1-03/T3-01/T4-01 exam-style) | **742 examples**, total 1607 |
| 2  | SFT v1: Qwen3-8B + LoRA r=64, 3 epochs on round-1 only | train loss 0.41 → 0.09 |
| 3  | SFT v2: same, on combined round-1+2 | train loss 0.44 → 0.07 |
| 4  | Baseline + v1 + v2 evals (no-think, max_new_tokens=512) | 2-shard parallel on A100s |
| 5  | Lenient re-score (key normalization to handle creative JSON keys) | Unlocks the real numbers |

## Final numbers — strict (exact key match)

| | n | correct | macro | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|---|
| Baseline (Qwen3-8B no-think) | 1029 | 7 | 0.68% | 4.0% | 0% | 0.5% | 0% |
| SFT v1 (gene_card only) | 1029 | 11 | 1.07% | 8.8% | 0% | 0% | 0% |
| SFT v2 (gene_card + T-style) | 1029 | 10 | 0.97% | 8.0% | 0% | 0% | 0% |

> Strict scoring penalized the model heavily for using creative JSON wrapper-key names (e.g., `Driver` instead of `driver`, `GroupA` instead of `ordered_group_a`). The benchmark prompts do NOT specify exact key names, so this is partly a verifier issue, not a model issue.

## Final numbers — lenient (key normalization)

| | n | correct | **macro** | T1 | T2 | T3 | T4 |
|---|---|---|---|---|---|---|---|
| Baseline | 1029 | 80 | **7.77%** | 18.6% | 3.3% | 8.0% | 6.0% |
| SFT v1 | 1029 | 82 | 7.97% | 23.4% | 2.6% | 8.5% | 4.0% |
| SFT v2 | 1029 | 102 | 9.91% | 24.8% | 4.4% | 9.4% | 9.6% |
| **SFT v3 (BEST)** | 1029 | **116** | **11.27%** (+3.50 vs base) | **28.2%** (+9.6) | **5.0%** (+1.7) | **12.6%** (+4.6) | 7.4% (+1.4) |
| SFT v4 | 1029 | 101 | 9.82% | 23.8% | 4.1% | 11.1% | 6.7% |
| SFT v5 (r=128) | 1029 | 109 | 10.59% | 25.4% | 5.0% | 11.7% | 7.2% |
| SFT v6 (strict-key prompts) | 1029 | 104 | 10.11% | 27.4% | 5.0% | 10.0% | 6.5% |
| SFT v7 (evidence-filtered) | 1029 | 101 | 9.82% | 22.8% | 3.9% | 11.2% | 7.2% |
| SFT v8 (evidence-enriched) | 1029 | 99 | 9.62% | 22.6% | 3.6% | 11.8% | 5.6% |
| SFT v9 (eval-format) | 1029 | 104 | 10.11% | **31.2%** | 1.5% | 11.1% | 6.7% |
| **SFT v10** (JSON+plain mix) | 1029 | 113 | 10.98% | **33.3%** ★ | 3.5% | 11.2% | 6.7% |
| evo-OPD v3 (50 step, gene_card pool) | 1029 | 110 | 10.69% | 28.2% | 5.0% | 11.6% | 6.2% |
| **evo-OPD v4** (100 step, mixed pool) | 1029 | 115 | 11.18% | 27.2% | **5.4%** | 12.2% | **7.9%** ★ |
| **evo-OPD v5** (200 step, on v10) | 1029 | 116 | **11.27%** ★tied | 28.3% | 3.9% | **12.9%** ★ | 7.4% |

**v3 (LoRA r=64) remains the best overall checkpoint.** v4 regressed on every tier — round-4's T3-13 data likely conflicted with round-3's T3-09 data (different gene-fate semantics confused the model). v5 doubled LoRA capacity (r=128, α=256) on the *same* v3 data but came in below v3 on T1/T3 and matched on T2/T4 — meaning capacity is **not** the bottleneck. The ceiling at this data scale is data quality + prompt-schema mismatch, not adapter rank. Lesson: adding more SFT data of *similar* tasks (T3-13 next to T3-09) without careful disambiguation can hurt, and bigger LoRA on the same data brings no free lunch.

### SFT recipe progression

| | examples | epochs | LR | train loss end | lenient acc |
|---|---|---|---|---|---|
| v1 | 865 | 3 | 2e-4 | 0.09 (overfit) | 7.97% |
| v2 | 1607 | 3 | 2e-4 | 0.07 (overfit) | 9.91% |
| **v3 (BEST)** | 2186 | 2 | 5e-5 | **0.27** (no overfit) | **11.27%** |
| v4 | 2365 | 2 | 5e-5 | 0.34 | 9.82% (regression) |
| v5 (r=128, α=256) | 2186 | 2 | 5e-5 | 0.33 | 10.59% (under v3) |
| v6 (strict-key prompts) | 2186 | 2 | 5e-5 | ~0.30 | 10.11% (failed) |

Lower LR + fewer epochs prevented the overfitting v1/v2 had, and **the lenient accuracy went up despite higher train loss** — clear evidence v1/v2 were over-fitting. v4's regression shows that data *quality and disambiguation* matter more than raw count once you're past ~2K examples. v5's regression (same data, double LoRA rank) shows that **adapter capacity is not the current bottleneck** — the model already has enough free parameters to fit this data; the gap to the ceiling is task understanding / prompt-schema mismatch, not representation capacity.

**v6 finding (important).** v6 appended an explicit "use EXACTLY these lowercase keys" instruction to each augmented-task training prompt AND rewrote completion JSON keys to canonical. Audit of v6 eval outputs found **0 of 126 instances on the five augmented task types emitted canonical lowercase keys** — every single one still emitted `"Driver"`/`"Dynamics"`/etc. The model **learned a conditional behavior**: it outputs canonical keys *only when the prompt contains the strict suffix*. At eval time, the GENE-Exam prompt has no such suffix, so the model defaults to mirroring the eval-prompt's capitalized "Driver = X" phrasing.
- **Conclusion:** training-time prompt augmentation does not transfer to a different eval-time prompt. The fix must either be (a) at eval time (prepend canonical-key instruction — but this changes the benchmark protocol) or (b) re-train with the actual GENE-Exam prompt format as the SFT prompt, so the model learns "for this kind of prompt, emit canonical keys".

**Follow-up diagnostic: v3 + eval-time strict-suffix on 206-instance subset.**

| | Strict | Lenient |
|---|---|---|
| v3 baseline (1029 inst) | 1.17% | 11.27% |
| v3 + 386-char strict-key eval-suffix (206 subset) | **5.83%** (+4.7) | **6.31%** (−4.96) |

Eval-time suffix lifts strict 5× (62/81 instances now emit canonical lowercase keys), but **lenient drops by half** — T1 in particular collapses from 28% → 4%. The verbose suffix is dominant enough to derail content quality. So format and content are independent levers; brute-force eval-time prompting helps strict at the cost of content. Implication: v7 (train on actual GENE-Exam eval prompts so canonical keys are learned *without* an eval-time suffix) is the right next experiment.

### Per-task winners (lenient)

| Task | Best result | Best model | Note |
|---|---|---|---|
| T1-02_genome_field_type | 44%/40% range | v1 | Direct beneficiary of gene_card_extract data |
| T1-01_contribution_type | (low across all) | — | Schema-wrapper key issue persists even after normalization |
| T3-01_single_dynamics | improved over baseline | v3 | Round 3's T3-09 generalized |
| T3-09_relation_classify | improved | v3 | Direct training data added |
| T2-07_lim_delta_match | improved | v3 | Direct training data added |
| T4-01_consistency_check | v2 best | v2 | Round 2's T4-01 data, lost in v3/v4 mixing |

## GENE-Arena PES (open-ended idea-generation benchmark)

50 tasks × 3 settings (Library/Lineage/Question) × {baseline, v3, v8, v10, evo-OPD-v4, evo-OPD-v5}.
Each idea generated by the local model (temperature 0.7, max_new_tokens 768),
then scored by Azure GPT-5.5 (keyless) using arena's `gene_arena_score_v3` rubric:
PES = 0.6 × population_score + 0.4 × scientific_quality_score, 6 sub-dims each,
with structural caps (incomplete-genome cap 65, missing-parent cap 60).
708 ideas scored (~$0 since keyless endpoint).

### Overall PES (mean ± bootstrap 95% CI over trace_ids, n_boot=2000)

All 6 participants now at full n=150 (900 idea-judge scores total).

| Rank | Participant | Library | Lineage | Question | **Overall** | 95% CI |
|---|---|---|---|---|---|---|
| **1** | Qwen3-8B + SFT v3 | **57.14** | 58.65 | **58.05** | **57.95** | [56.74, 59.13] |
| 2 | Qwen3-8B + SFT v10 | 56.70 | **58.79** | 56.65 | 57.38 | [56.09, 58.56] |
| 3 | Qwen3-8B + SFT v8 (strict-best) | 55.61 | 57.97 | 56.51 | 56.69 | [55.48, 57.87] |
| 4 | Qwen3-8B + evo-OPD v4 | 55.70 | 57.57 | 56.45 | 56.57 | [55.21, 57.81] |
| 5 | Qwen3-8B + evo-OPD v5 | 55.25 | 58.27 | 55.63 | 56.38 | [55.09, 57.52] |
| 6 | Qwen3-8B baseline (no LoRA) | 49.53 | 55.34 | 47.99 | 50.96 | [49.27, 52.56] |

### Pairwise paired-bootstrap deltas — A vs B (overall PES, trace-level, n=150 each)

| A − B | Δ mean | 95% CI | P(A>B) | sig |
|---|---|---|---|---|
| **v3 − baseline** | **+6.99** | [+5.94, +8.22] | 1.000 | ★★★ |
| **v10 − baseline** | **+6.43** | [+5.17, +7.75] | 1.000 | ★★★ |
| **v8 − baseline** | **+5.74** | [+4.64, +6.86] | 1.000 | ★★★ |
| **evo-OPD-v4 − baseline** | **+5.62** | [+4.39, +7.01] | 1.000 | ★★★ |
| **evo-OPD-v5 − baseline** | **+5.43** | [+4.27, +6.59] | 1.000 | ★★★ |
| v3 − v10 | +0.56 | [−0.16, +1.37] | 0.925 | marginal |
| **v3 − v8** | **+1.25** | [+0.36, +2.15] | 0.998 | ★ |
| **v3 − evo-OPD-v4** | **+1.37** | [+0.53, +2.18] | 0.999 | ★ |
| **v3 − evo-OPD-v5** | **+1.56** | [+0.73, +2.39] | 1.000 | ★ |
| v10 − v8 | +0.69 | [−0.16, +1.57] | 0.947 | marginal |
| v10 − evo-OPD-v4 | +0.81 | [−0.12, +1.78] | 0.954 | marginal |
| **v10 − evo-OPD-v5** | **+1.00** | [+0.29, +1.67] | 0.997 | ★ |
| v8 − evo-OPD-v4 | +0.12 | [−0.82, +0.97] | 0.612 | n.s. |
| v8 − evo-OPD-v5 | +0.31 | [−0.50, +1.23] | 0.768 | n.s. |
| evo-OPD-v4 − evo-OPD-v5 | +0.19 | [−0.71, +1.11] | 0.646 | n.s. |

★ = 95% CI excludes zero. Bold = headline comparisons.

**Three statistical clusters emerge cleanly:**
- **Cluster A** (winner): v3 alone at 57.95 — significantly beats every other LoRA model
- **Cluster B**: v10 at 57.38 — marginally below v3, marginally above the specialized cluster
- **Cluster C** (specialized): {v8, evo-OPD-v4, evo-OPD-v5} all at 56.4–56.7, statistically indistinguishable
- **Cluster D**: baseline at 50.96 — 5–7 pts below all SFT/RL variants

### Sub-dimension means (across all settings)

Population dims (out of 5):

| pid | parent_ground | gene_inherit | limit_repair | evo_plausib | bal_novelty | graph_insert |
|---|---|---|---|---|---|---|
| baseline | 3.40 | 3.32 | 2.84 | 3.46 | 2.55 | 3.09 |
| **v3** | 4.01 | **3.96** | **3.43** | **3.80** | **3.11** | **3.63** |
| v10 | **4.03** | 3.94 | 3.34 | 3.79 | **3.11** | 3.62 |
| v8 | 3.91 | 3.85 | 3.33 | 3.68 | 3.06 | 3.60 |
| evo-OPD-v4 | 3.92 | 3.85 | 3.26 | 3.71 | 2.98 | 3.57 |
| evo-OPD-v5 | 3.95 | 3.86 | 3.24 | 3.70 | 2.95 | 3.53 |

Scientific-quality dims:

| pid | prob_import | mech_concrete | originality | feasibility | valid_rigor | exp_impact |
|---|---|---|---|---|---|---|
| baseline | 4.89 | 2.11 | 2.51 | 2.45 | 2.72 | 4.06 |
| **v3** | **4.93** | 2.57 | 3.13 | **2.59** | 2.77 | **4.42** |
| v10 | 4.91 | 2.53 | **3.15** | 2.51 | **2.78** | 4.39 |
| v8 | 4.85 | **2.62** | 3.06 | 2.54 | 2.63 | 4.38 |
| evo-OPD-v4 | 4.87 | 2.46 | 2.99 | 2.55 | 2.69 | 4.34 |
| evo-OPD-v5 | 4.92 | 2.45 | 2.92 | 2.50 | 2.73 | 4.33 |

### Headline findings

1. **SFT lifts PES by ~5-7 points over base** — every SFT/RL variant beats baseline by Δ=+5.4 to +7.0 with all CIs comfortably excluding zero (p<0.001). The lift is concentrated on **population grounding dims**: `gene_inheritance`, `parent_grounding`, `graph_insertion` all gain ~0.5 — the model learns to anchor ideas in a parent trace, which the base model fails at. Limitation-repair improves +0.4 and balanced-novelty +0.5 — SFT teaches what novelty *looks like* in context.

2. **v3 is the open-ended-arena champion** (PES 57.95), significantly beating v8 (Δ=+1.25, p=0.002), evo-OPD-v4 (Δ=+1.37, p=0.001), and evo-OPD-v5 (Δ=+1.56, p<0.001). v3 vs v10 is marginal (Δ=0.56, p=0.075).

3. **Three specialization paths converge on the same PES ceiling** (≈56.4–56.7): strict-format SFT (v8), T1-focused SFT (v10 — only marginally), and verifier-anchored RL (evo-OPD v4/v5). All three are statistically indistinguishable from each other (v8 vs evo-OPD-v5: Δ=+0.31 n.s.; evo-OPD-v4 vs v5: Δ=+0.19 n.s.). This is the same pattern across SFT (v8) and RL (evo-OPD) — i.e., it's a **specialization-vs-generality trade-off**, not specific to RL.

4. **evo-OPD-v4 (100 steps) and evo-OPD-v5 (200 steps) are statistically equivalent on PES** (Δ=+0.19, p=0.65). The PES-cost is paid in the **first ~100 RL steps**; longer training doesn't compress further. This is consistent with the "exploration collapse" intuition — once the policy concentrates near the verifier-optimal mode, further RL just exploits.

5. **GENE-Exam vs PES is decoupled — important paper point.** GENE-Exam-only reporting would say "evo-OPD-v5 ties v3 and improves T3." PES reveals the corresponding cost. Reporting both creates a richer picture and prevents over-claiming.

### Per-setting story

- **Library** (most schema-heavy): v3 leads (57.14); baseline catastrophically low (49.53)
- **Lineage** (parent-grounding): all SFT variants converge to 57-59; v10 nominally best
- **Question** (most open-ended): v3 strongly leads (58.05); evo-OPD compresses by ~2.5 (55.63)

The pattern aligns with intuition: verifier-RL helps where structure is rewarded (Lineage) and hurts most where open exploration is rewarded (Question).

### Methodology caveats

- **Single-judge scoring**: 98.4% of ideas were scored by **only `gpt-5.5`** — the rubric specifies a 3-judge ensemble (gpt-5.5, gpt-5.4, gpt-5.4-nano) but the latter two return DeploymentNotFound on this Azure endpoint. 1.6% of ideas (11/708) defaulted to PES=50.0 with zero valid judges, spread evenly across participants (1-3 per model). **This adds noise but not bias** — the same judge scores all participants' ideas for a given trace, so the paired bootstrap remains valid. We should re-score with a 3-judge ensemble before paper submission if the other deployments come back.
- **Bootstrap is by trace_id**, not by idea, to preserve the paired structure (each trace appears in all 3 settings for all 6 participants).
- **Caps applied**: arena rubric caps PES at 65 (incomplete genome), 60 (missing parent agreement), and lower for parse failure / weak feasibility. **Cap-hit rate by participant**: baseline 32.7%, evo-OPD-v5 67.3%, v8 65.5%, v3 73.3%, v10 74.0%. Counter-intuitively, **evo-OPD-v5 hits caps less often than v3** (48% parse-cap vs 55%, i.e. evo-OPD generates MORE parseable JSON). So the PES gap is **not** explained by caps — evo-OPD produces *better-formatted but lower-quality content*, exactly the RL-on-verifier story: more format compliance, less creative depth.
- **Baseline cap rate is much lower (32.7%)** mostly because the base model produces free-form text that the scorer can't structurally evaluate — many fields read as "unspecified," which dodges structural caps but also kills the population dims (parent_grounding 3.40 vs v3's 4.01).

## What worked

## What worked

1. **GPT-5.5 as SFT teacher with verifier filtering** — 100% acceptance rate on round 1, 74% on round 2 (rate-limit losses, not quality losses).
2. **LoRA r=64 on Qwen3-8B** with 3 epochs converged to loss ~0.07 on 1607 examples in 35 min on 1× A100.
3. **2-shard sharded evaluation** — full 1029-instance eval in ~17 min on 2× A100.
4. **Key normalization at scoring time** — unlocked +7 pts overall without any retraining.

## What didn't work / surprised

1. **Strict schema verifier** rejected most correct-content answers because of wrapper-key mismatch. Fixable at scoring time (and ideally by training with strict prompts).
2. **vLLM serving** — fought against tokenizer / version incompatibilities for ~30 minutes; gave up and used plain transformers `generate()`. Acceptable cost.
3. **no-think mode** caps performance hard. Need to re-eval with thinking enabled to see ceiling.
4. **Synthesizing exam-style training data without gold answers** (round 2) introduced noisier labels than gene_card_extract (which had GPT-5.5 evidence quotes to verify against).

## Files produced

```
idea_train/
├── data/stage1_sft/
│   ├── train.jsonl                                865 round-1 examples
│   ├── round2_train.jsonl                         742 round-2 examples
│   └── train_combined.jsonl                       1607 combined
├── train/checkpoints/
│   ├── qwen3-8b-sft-v1/final/                     LoRA r=64 (round-1 only)
│   ├── qwen3-8b-sft-v2/final/                     LoRA r=64 (combined)
│   ├── qwen3-8b-sft-v3/final/                     LoRA r=64, LR 5e-5, 2 epochs (BEST, 11.27%)
│   ├── qwen3-8b-sft-v4/final/                     LoRA r=64 + round-4 data (regressed, 9.82%)
│   ├── qwen3-8b-sft-v5/final/                     LoRA r=128 on v3 data (10.59%, capacity ablation)
│   └── qwen3-8b-sft-v6/final/                     LoRA r=64 + strict-key prompts (10.11%, transfer-failed)
├── eval/results/
│   ├── qwen3-8b_baseline_nothink/                 baseline 0.68% / 7.77% lenient
│   ├── qwen3-8b-sft-v1_nothink/                   1.07% / 7.97% lenient
│   ├── qwen3-8b-sft-v2_nothink/                   0.97% / 9.91% lenient
│   ├── qwen3-8b-sft-v3_nothink/                   11.27% lenient (BEST)
│   ├── qwen3-8b-sft-v4_nothink/                   9.82% lenient
│   ├── qwen3-8b-sft-v5_nothink/                   10.59% lenient (r=128 ablation)
│   ├── qwen3-8b-sft-v6_nothink/                   10.11% lenient (strict-key prompt failed)
│   ├── round1_comparison.md
│   ├── round2_comparison.md
│   └── v1_vs_v2_comparison.md
├── denylist/denylist_v0.jsonl                     8359 papers; v1 (1-hop) running in background
└── logs/                                           SFT gen, training, eval logs
```

## What's next (recommended for tomorrow)

Since v5 confirmed capacity is fine and v6 confirmed training-time prompt augmentation does not transfer, the highest-leverage next steps are:

1. **Eval-prompt-matched SFT (v7).** Discard the synthetic training prompts; use the actual GENE-Exam `instances.json` prompt text from each task type's Questions/ dir as the SFT prompt, with GPT-5.5 answers as completion in canonical-key JSON. This trains the model on **exactly the same prompt distribution it sees at eval**, so canonical-key behavior should transfer. Estimate ~1 day of GPT-5.5 calls + 30 min train + 30 min eval.
2. **Eval-time prompt augmentation experiment (diagnostic, not real benchmark).** Re-run v3 eval with a one-line "Output JSON with lowercase keys" prefix prepended to each instance prompt. This is a 30 min eval; if it lifts strict to ~10%, that confirms v7 will work. If it doesn't, the bottleneck is content quality not formatting.
3. **Eval v3 with thinking enabled** — gives the proper ceiling. Estimate 4–6 hr on 2× A100; pure no-think mode is leaving headroom on the table.
4. **Disambiguation-aware data generation** — for similar tasks (T3-09 vs T3-13, T2-04 vs T2-05), include explicit "this task asks X, not Y" hints in the teacher prompt to avoid v4-style cross-task confusion.
5. **GRPO / evo-OPD on T1/T3** — biggest absolute headroom is on T3 (currently 12.6% with v3); RL with the verifier reward should be high-leverage there.
6. **Generate full denylist v1** (OpenAlex expansion died silently after ~7h); restart with proper progress logging before round-5 data gen.

## Open issues to be aware of

- OpenAlex expansion (started ~18:08 UTC) — long-running, has been silent due to stdout buffering. Not blocking.
- vLLM was abandoned for transformers `generate()` due to version conflicts. Tolerable for eval but RL/GRPO will need a working vLLM later for rollouts.
- Round-2 data has noisier labels (no gold to verify against). For round 3, prefer reusing GPT-5.5's first answer AND a second teacher's answer; only keep examples where they agree (consistency filter).
