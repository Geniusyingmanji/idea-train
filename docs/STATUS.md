# Project Status — 2026-05-18

> Snapshot of what's done, what's blocked, what's next. Refresh whenever you
> spawn a new session and want to skip rebuilding context.

## Headline

- **SFT best (overall): v3 at 11.27% lenient** — Qwen3-8B + LoRA r=64, 2186 examples, LR 5e-5, 2 epochs.
- **SFT best (T1 tier): v10 at 33.3% T1** (mixed JSON+plain-text training, ⬆ from v3's 28.2%).
- **SFT best (strict): v8 at 2.33% strict** (evidence-enriched completions).
- **evo-OPD validated**: 5 iterations, v5 (200 steps on v10 base, mixed pool) **ties v3's 11.27% lenient AND sets new T3 record (12.90%, ⬆ from v3's 12.6)**. Algorithm recovers generalist behavior from a specialized SFT checkpoint while gaining on T3.
- **GENE-Arena PES complete on 6 models** (900 ideas, n=150 each): baseline 50.96 → SFT v3 **57.95** (BEST) > v10 57.38 > v8 56.69 ≈ evo-OPD-v4 56.57 ≈ evo-OPD-v5 56.38. All SFT/RL lift baseline by **Δ=+5-7 pts** (all p<0.001). **v3 > evo-OPD-v5 by +1.56** [0.73, 2.39] (p<0.001), **v3 > evo-OPD-v4 by +1.37** [0.53, 2.18] (p=0.001), **v3 > v8 by +1.25** [0.36, 2.15] (p=0.002), v3 vs v10 marginal (+0.56, p=0.075). The 3 specialized variants (v8, evo-OPD-v4, evo-OPD-v5) are **statistically indistinguishable from each other** — all hit the same ~56.5 PES ceiling. Specialization (strict-format / T1 / verifier-RL) all compress PES by ~1.3pt — **specialization-vs-generality trade-off, not RL-specific**. evo-OPD-v4 (100 steps) ≈ evo-OPD-v5 (200 steps) → PES cost paid in first ~100 RL steps then stable.
- **GeneTrace v0.1 built**: 855 GenomeCards + 300 (synthetic) DynamicsEdges + verifier bundle + dataset card; 0 denylist hits.
- **Paper skeleton compiles**: 8-page PDF, 8 section files, ethics §Min-K table populated with real numbers (baseline + v1-v6 all CLEAN).
- **evo-OPD reward composition validated end-to-end on real data** — 0 errors, 0 NaN/Inf across 209 examples covering all 4 token roles.
- **Contamination guard verified**: Min-20%++ on 50 GENE-Exam papers stays within ±0.13 of baseline across all 6 trained checkpoints; Δ vs reference always strongly negative (−0.87 to −0.96), well below the 0.05 leakage threshold.

## Done

| Workstream | Artefact | State |
|---|---|---|
| SFT iteration v1-v6 | `train/checkpoints/qwen3-8b-sft-v{1..6}/final/` | done, v3 is BEST |
| GeneTrace v0.1 corpus | `data/genetrace_v0_1/{cards,edges,verifier_bundle,dataset_card}` | **done; 824/855 (96.4%) evidence-grounded, mean 9.46 quotes/card** |
| Build script | `tools/build_genetrace_v0_1.py` | done, dry-runnable, 4 stages |
| Paper draft skeleton | `paper/latex/main.tex` + `sections/*.tex` (8 files) | compiles to 8-page PDF; all numbers `\todo{}` |
| Paper positioning | `paper_positioning.md` | done; IdeaEvolving framed as our prior work (per user) |
| Data format spec | `genetrace_data_format.md` | done; 4-level schema |
| evo-OPD open-ended spec | `evo_opd_open_ended.md` | done; verifier→continuous extension |
| Reward composition validation | `tools/validate_evo_opd_rewards.py` | done; 0 errors on 209 examples across 5 task types |
| Min-K%++ leakage script | `tools/min_k_leakage_check.py` | done; **full table run on baseline + v1-v6**, all CLEAN (Δ ≈ −0.87 to −0.96) |
| Contamination guard | denylist (15,698 ids); pre-2017 cut | enforced in build script |
| Plan consolidation | `plan.md` + `lv_opd_plan.md` updated | Stage 2/3 merged → Stage 2 = evo-OPD |

## Blocked

| Item | Blocker | ETA to unblock |
|---|---|---|
| ~~Stage E: GPT-5.5 evidence-quote extraction~~ | GPT-5.5 recovered ~14:30 UTC; ran 3 passes, **824/855 (96.4%) covered**; 31 stubborn content-filter rejections accepted as residual | done |
| Real-citation edges for v0.2 | **OpenAlex 429 rate-limited** (silent denylist expansion ran 19h+ with 10 workers, exhausted quota; killed at 13:46) | likely UTC midnight quota reset (~10h) |
| v0.2 dynamics labeling | Both of the above |
| Min-K v3 result | running now (PID 2301579), ETA ~13:55 UTC |

## Not started (intentionally — needs design decision)

| Item | Why deferred |
|---|---|
| ~~**evo-OPD v6 (Arena-Rank reward)**~~ | Implemented + mid-smoke validated (K=8 20-step on v3, loss 0.40→0.20), but **superseded by agentic pivot on 2026-05-19**. Code preserved at `evo_opd/{judges/*, trainer/tournament.py, trainer/evo_opd_loop_v6.py, structural.py}`. Modules (struct, pairwise judge, tournament) are reused by agentic-OPD as building blocks. |
| **Agentic-OPD (ACTIVE direction)** | Spec: `agentic_opd_plan.md`. Code: `evo_opd/{agentic,tools,trainer/agentic_loop}.py`. Pipeline: 248 prompts + 855-card BM25 search + ReAct rollout + multi-source reward (lineage + struct + format + tournament). Cold-start v3 smoke: 2/5 propose. SFT warm-start (246 GPT-5.5 demos, ~$0) running NOW (PID 2093678, GPU 1, ETA 3 min). Auto-chain launches RL on GPU 0+2 after SFT. Three eval targets: SGI-Bench task_2 (graph_similarity), GENE-Arena PES (existing), ArenaRL Open tasks (deferred). |
| evo-OPD trainer (verl + vLLM teacher) | Big multi-day workstream; needs design alignment on rollout cadence, prompt mix, teacher serving |
| Qwen3-14B teacher serving | vLLM had Qwen3 tokenizer issues earlier; needs retry with newer vLLM build |
| GENE-Arena (open-ended) eval pipeline | Needs PES rubric judge + design for how to score; deferred until v0.2 dataset stable |
| Cross-domain (bio/physics) data | v0.2+ scope |

## Active background tasks

- `b0eusxb1s` — Monitor polling GPT-5.5 every 4 min (will notify when it recovers)
- `bz6ngxpwc` — Wait for Min-K runs on GPU 0 + 2 (~4 min ETA)

## When you wake up — next moves I'd recommend, in order

1. **Read this file + `paper_positioning.md`** to re-load context.
2. **Check `data/genetrace_v0_1/min_k_*.json`** for the leakage numbers — if both baseline and v3 are < 0.05 delta, our contamination guard claim has its first datapoint.
3. **If GPT-5.5 is back:** fire Stage E (`tools/extract_evidence_quotes.py`) on all 855 cards (~30 min, ~$5). This unlocks the v0.1 evidence-grounded release.
4. **If OpenAlex is back:** run `tools/build_real_citation_edges.py` (~20 min) to build the v0.2 unlabeled citation graph.
5. **Independent of APIs:** decide whether to start evo-OPD trainer infra (option C from the earlier conversation). My recommendation: yes, but it's a multi-day workstream.

## Files modified this session (relative to `/home/azureuser/workspace-gzy/zyf/idea_train/`)

```
NEW:
  STATUS.md                                       (this file)
  paper_positioning.md                            (created; flipped IdeaEvolving framing)
  genetrace_data_format.md
  evo_opd_open_ended.md
  paper/latex/main.tex
  paper/latex/sections/{intro,related,genetrace,method,experiments,
                        discussion,limitations,ethics,appendix}.tex
  tools/build_genetrace_v0_1.py
  tools/extract_evidence_quotes.py                (ready, blocked on GPT-5.5)
  tools/build_real_citation_edges.py              (ready, blocked on OpenAlex)
  tools/validate_evo_opd_rewards.py
  tools/min_k_leakage_check.py
  data/genetrace_v0_1/                            (cards, edges, bundle, card)

MODIFIED:
  plan.md                                         (Stage 2/3 collapsed)
  eval/eval_gene_exam_lora.py                     (added --prompt-suffix-file)
  eval/results/OVERNIGHT_REPORT.md                (v4/v5/v6 + diagnostic added)
  tools/launch_sft_train.py                       (added --lora-r/-alpha)
```

## Key numbers to remember

```
GENE-Exam main challenge, 1029 instances, no-think:
  baseline:  0.68% strict  /  7.77% lenient
  v3 (BEST): 1.17% strict  / 11.27% lenient    ← current best
  GPT-5.5:                   23.10% (paper ref)

Compute budget burned this session:
  ~6 SFT runs × ~30 min × 1 GPU  = ~3 GPU-hours
  ~6 eval runs × ~20 min × 2 GPUs = ~4 GPU-hours
  + 1 diagnostic eval, + 2 Min-K runs
  Total: ~10 A100-hours
```
