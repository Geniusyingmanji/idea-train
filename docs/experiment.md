# Experiments

Consolidated results from 2026-05-17 → 2026-05-22 (GENE-Exam SFT phase + Agentic-OPD pivot + 14h data expansion).

## TL;DR

- **Non-agentic SFT v3 = champion**: 11.27% lenient on GENE-Exam (vs 7.77% baseline); 57.95 PES on GENE-Arena (vs 50.96 baseline).
- **Agentic pivot revealed an 18% propose-rate failure** on agentic-v3-sft (27/150 GENE-Arena ideas reach `propose`). Diagnosed: v3 demos had median 10 tools/demo → model over-tools and runs out of `max_turns`.
- **Data expansion (5953 SFT + 2593 DPO + 2930 RL prompts)** inverts the length bias to fix this. Re-SFT pending.

## 1. GENE-Exam main benchmark (1029 instances, lenient scoring)

| Model | n | lenient % | Δ vs base | Notes |
|---|---:|---:|---:|---|
| Qwen3-8B baseline | 1029 | 7.77 | — | no LoRA, no-think |
| SFT v1 (round-1, 865 ex) | 1029 | 7.97 | +0.20 | overfit |
| SFT v2 (round-1+2, 1607 ex) | 1029 | 9.91 | +2.14 | overfit |
| **🏆 SFT v3 (2186 ex, LR 5e-5, 2 epochs)** | 1029 | **11.27** | **+3.50** | no overfit, BEST |
| SFT v4 (+round-4) | 1029 | 9.82 | +2.05 | T3-13 conflicted with T3-09 |
| SFT v5 (r=128 ablation) | 1029 | 10.59 | +2.82 | capacity not the bottleneck |
| SFT v6 (strict-key prompts) | 1029 | 10.11 | +2.34 | training-time aug didn't transfer |
| SFT v7 (evidence-filtered) | 1029 | 9.82 | +2.05 | overfit on filter |
| SFT v8 (evidence-enriched) | 1029 | 9.62 | +1.85 | regression |
| SFT v9 (eval-format) | 1029 | 10.11 | +2.34 | T1 33.3% ★ |
| SFT v10 (JSON+plain mix) | 1029 | 10.98 | +3.21 | T1 33.3% ★ |
| evo-OPD v3 (50 step RL) | 1029 | 10.69 | +2.92 | gene_card pool |
| evo-OPD v4 (100 step) | 1029 | 11.18 | +3.41 | T2 5.4% ★, T4 7.9% ★ |
| evo-OPD v5 (200 step on v10) | 1029 | 11.27 | +3.50 | T3 12.9% ★, tied v3 |

Reference: GPT-5.5 scores 23.1% on the same benchmark (IdeaEvolving paper). Qwen3-8B+LoRA reaches **~49% of GPT-5.5** with ~30× fewer parameters.

**Key lesson (v3 vs v4/v5/v6):** past 2K examples, **data quality + prompt-schema disambiguation** matters more than raw example count or adapter capacity. v4 added similar-task data → cross-task confusion. v5 doubled LoRA r=128 → no gain. v6 trained on strict-key prompts → the model learned a *conditional* behavior, didn't transfer to eval-time prompts.

## 2. GENE-Arena PES (open-ended idea generation, n=150 × 3 settings)

| Rank | Model | Library | Lineage | Question | **Overall PES** | 95% CI |
|---|---|---:|---:|---:|---:|---|
| 🏆 | Qwen3-8B + SFT v3 | 57.14 | 58.65 | 58.05 | **57.95** | [56.74, 59.13] |
| 2 | + SFT v10 | 56.70 | 58.79 | 56.65 | 57.38 | [56.09, 58.56] |
| 3 | + SFT v8 | 55.61 | 57.97 | 56.51 | 56.69 | [55.48, 57.87] |
| 4 | + evo-OPD v4 | 55.70 | 57.57 | 56.45 | 56.57 | [55.21, 57.81] |
| 5 | + evo-OPD v5 | 55.25 | 58.27 | 55.63 | 56.38 | [55.09, 57.52] |
| 6 | baseline | 49.53 | 55.34 | 47.99 | 50.96 | [49.27, 52.56] |

**Pairwise paired-bootstrap deltas:**

| A − B | Δ | 95% CI | sig |
|---|---:|---|---|
| v3 − baseline | +6.99 | [+5.94, +8.22] | ★★★ |
| v3 − v8 | +1.25 | [+0.36, +2.15] | ★ |
| v3 − evo-OPD-v5 | +1.56 | [+0.73, +2.39] | ★ |
| v3 − v10 | +0.56 | [−0.16, +1.37] | marginal |
| v10 − evo-OPD-v5 | +1.00 | [+0.29, +1.67] | ★ |
| evo-OPD-v4 − v5 | +0.19 | [−0.71, +1.11] | n.s. |

**Three statistical clusters:**
- **A (winner)**: v3 alone at 57.95
- **B**: v10 at 57.38 (marginal below v3)
- **C (specialized)**: {v8, evo-OPD-v4, evo-OPD-v5} at 56.4-56.7, indistinguishable
- **D**: baseline at 50.96

**Sub-dim story:** SFT lifts ~+0.5 across population dims (parent_grounding, gene_inheritance, graph_insertion, limitation_repair, balanced_novelty), confirming the model learns to anchor ideas in a parent trace. v3 leads on `mech_concrete` (2.57), `originality` (3.13), `exp_impact` (4.42).

**Decoupling finding (paper-relevant):** GENE-Exam-only reporting would say "evo-OPD-v5 ties v3 and improves T3." PES reveals the cost. Reporting both creates a richer picture. evo-OPD trades PES for tool-task gains → **specialization vs generality trade-off**, paid in the first ~100 RL steps and stable thereafter.

**Methodology caveat:** 98.4% of ideas scored by single judge (gpt-5.5 only; gpt-5.4 / gpt-5.4-nano DeploymentNotFound). 1.6% (11/708) defaulted to PES=50.0 — distributed evenly across participants so paired bootstrap remains valid.

## 3. Agentic propose-rate finding (2026-05-21)

Rollouts on 150 GENE-Arena prompts (`tools/agentic_eval_gene_arena.py` with `--max-turns 9`):

| Adapter | n | propose_emitted | rate | median tokens | median latency |
|---|---:|---:|---:|---:|---:|
| `qwen3-8b-agentic-v2-sft + web` | 116 | 70 | **60%** | — | — |
| `qwen3-8b-agentic-v3-sft + web` | 150 | 27 | **18%** | 1180 | 466 s |

**Diagnosis:** v3 agentic SFT trained on 1032 demos with median 10 tools/demo and 77% long trajectories. Model learned "research = many tool calls", ran out of `max_turns=9` before reaching propose 82% of the time. v2 had shorter demos → higher propose rate but lower quality coverage.

This is the empirical motivation for the v5-v33 data expansion. Combined corpus now has **64% short / 14% med / 22% long** demos (median 3 tools/demo), inverting v3's bias.

**PES on agentic-v3-sft is pending** — Azure rate-limited the judging. Will re-run when quota recovers.

## 4. DPO yield comparison

| Round | Approach | Pairs requested | Pairs delivered | Yield |
|---|---|---:|---:|---:|
| v7 | Tournament (independent chosen+rejected calls) | 270 | 91 | 34% |
| v10 | Focused 2-mode tournament | 240 | 98 | 41% |
| **v13** | **Corruption (chosen=existing SFT, rejected=corrupt it)** | 420 | 398 | **95%** |
| v19 | v13 fresh seed | 600 | 571 | 95% |
| v22 | v13 fresh seed | 600 | 575 | 96% |
| v27 | v13 with 3-mode focus | 300 | 297 | 99% |

**Lesson:** corruption-style generation (single-shot rewrite of an existing demo) has much higher yield than tournament-style (two independent calls must both succeed). Total DPO across 6 rounds = 2593 pairs.

## 5. Methodological lessons from data generation

### What worked

| Approach | Outcome |
|---|---|
| **Corruption-based DPO** | 95% pair yield vs 34-41% tournament-style |
| **Schema-strict system prompts** | v5/v6/v15 hit 95%+ valid; loose prompts (v32 retry-1) returned 0% |
| **Explicit length-tier hints** | v5/v11 produced exactly the short distribution requested |
| **2-turn dialogue with simpler structure** | v14 100% yield vs v9's 22% |
| **OpenAlex disk cache** | Prefetched candidates → A2 phase from ~10 min → 6 sec on repeated rounds |

### What broke (and the fix)

| Bug | Round | Cause | Fix |
|---|---|---|---|
| 0 valid demos | v11 first | `max_tokens=1800` truncated `n=40` prompts mid-JSON | Bumped to 4000 |
| 0 valid demos | v32 first | `max_tokens=2200` truncated `n=15` long-context prompts | n=10, max_tokens=4000 |
| 0 valid demos | v32 retry-1 | Loose SYS let GPT use markdown `Action 1 —` headers | Strict format SYS with literal ` ```action ``` ` example |
| KeyError in `.format()` | v7 | Literal `{6 fields}` in SCHEMA_GUIDE was treated as placeholder | Replaced `{...}` with `[...]` in literal sections |
| ZH demos miss `propose` | v6, v15 | Chinese `propose` text broke ASCII `"propose"` substring check | Kept ASCII propose keyword; rationale stays Chinese |

### What didn't pay off

- **More SFT data of similar tasks** (v4 vs v3): regressed by 1.5%. Disambiguation matters past 2K examples.
- **Bigger LoRA** (v5 r=128 vs v3 r=64): no gain; capacity isn't the bottleneck.
- **Training-time prompt augmentation** (v6): model learned conditional behavior, didn't transfer.
- **Tournament-style DPO**: low yield. Corruption-style dominates.

## 6. Reproduce eval rollouts

```bash
python tools/agentic_eval_gene_arena.py \
  --student-lora <new_lora_path> \
  --participant <participant_name> \
  --gpu <id> --workers 1 --max-turns 9 \
  --max-new-tokens-per-turn 512 --temperature 0.5 \
  --search-backend web \
  --output-dir eval/results/arena_<name>
```

Manifest at `<output-dir>/manifest.jsonl`. PES judging is a separate Azure GPT-5.5 step using the `gene_arena_score_v3` rubric (PES = 0.6 × population_score + 0.4 × scientific_quality_score, 6 sub-dims each, with structural caps).

## 7. Open issues

- **Azure GPT-5.5 rate-limited** intermittently — affects PES scoring of agentic-v3-sft and any new model evals.
- **Single-judge PES**: gpt-5.4 / gpt-5.4-nano return DeploymentNotFound. Should re-score with 3-judge ensemble before paper if other deployments come back.
- **vLLM tokenizer conflicts**: abandoned for transformers `generate()`. Tolerable for eval but RL rollouts will eventually need vLLM.
- **GENE-Exam in no-think mode**: caps performance hard. Re-eval with thinking enabled would give a proper ceiling (4-6 hr).
