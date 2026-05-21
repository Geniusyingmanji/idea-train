# Survey: Training an 8B Scientific-Idea LLM for GENE-bench (and Beyond)

> Compiled May 2026. Target: an 8B model trained with **SFT → GRPO → OPD** that improves on GENE-bench (GENE-Exam + GENE-Arena) **and** generalizes to other scientific-idea / reasoning benchmarks, with strict no-cheating on training data.

---

## 0. Problem framing

GENE-bench (Geniusyingmanji/IdeaEvolving) is a benchmark for **scientific lineage reasoning**: it represents each paper as an `IdeaGenome` (mechanism / niche / observation / limitation / delta / claim) plus an `EcologyContext`, and labels parent→child transitions with five evolutionary dynamics (Mutation, Adaptive Radiation, Hybridization, Speciation, Niche Competition) plus an Isolation null.

It has two evaluation tracks:

- **GENE-Exam** — closed-form, 42 main-challenge task types / 1,029 instances over T1–T4 (Genome Abstraction → Inheritance Tracing → Evolutionary Reasoning → Lineage Verification). Anonymous mode + composite all-or-nothing scoring. Current SOTA: GPT-5.5 23.1% exact-match; strongest CLI harness 27.3%.
- **GENE-Arena** — open-ended idea generation under 3 progressive settings (Question / Library / Lineage), scored by **PES** (Heredity / Variation / Selection, 0–100) plus pairwise ELO.

Three structural findings from the paper:
1. The bottleneck is **compositional exactness**, not knowledge — models recognize local pieces but fail to compose a full lineage judgment.
2. **Lineage verification (T4) is the bridge from understanding to generation** — without it, generated proposals don't preserve parents or repair limitations.
3. **Lineage context is a discriminator, not a universal boost** — it changes *which* models can use structured evidence.

Our model has to attack all three. Generality matters: if we only train on GENE-bench-like data, we overfit the genome-card format and lose free-form science Q&A; we must train on a contamination-safe substrate and evaluate on disjoint benchmarks.

### Success criteria (what "win" looks like)

| Track | Metric | Baseline (best public 8B) | Target |
|---|---|---|---|
| GENE-Exam main-challenge | exact accuracy | DeepSeek-R1-Distill-Qwen-7B ≈ 17–20% (interpolated) | **≥ 28%** (matches GPT-5.5+ClaudeCode harness) |
| GENE-Arena PES (Lineage) | 0–100 | direct 8B baselines ~70–75 expected | **≥ 82** |
| GENE-Arena ELO (Lineage) | Bradley-Terry vs frontier panel | ~700–900 | **≥ 1100** (between Claude Opus 4.7 direct and CLI harness) |
| Generality eval (§3) | macro-avg score | starting checkpoint | **no regression > 2 abs pts** on math/code/GPQA |

---

## 1. GENE-bench data map → contamination denylist scope

From inspection of `/IdeaEvolving/`:

| Asset | Count | Field used as ID |
|---|---|---|
| `data/paper_db/paper_db.json` | **29,472 papers** | mixed: `paper:title_slug:year` (25,993) + `s2:<hash>` (3,479) |
| `data/paper_db/json/` | 213 full-text JSONs | normalized title filename |
| `data/三次处理/papers_parsed/` | **~17,515 parsed papers** | per-domain subdirs (CS×26, Science×24) |
| `data/genome_db/paper_gene_cards.json` | **2,076 genome cards** | `paper_id`, `title`, `year`, `trace_id` |
| `data/genome_db/trace_graphs.json` | **90 traces** (~4,500 paper nodes) | each node has `s2_id`, `title`, `year` |
| `gene_arena/task/*.json` | **50 frontier traces** across 15 domains | each lists 5–7 papers with `title`, `year`, `arxiv_id` |
| `gene_exam/Questions/T*` | **1,380 instances**, 56 task types (42 in main-challenge) | papers embedded inline as anonymized genome text |

The 2,076 genome cards + the trace-graph nodes are the canonical "papers in the benchmark." After 1-hop citation closure that becomes roughly **5–7K unique papers** to denylist. Two hops would blow up to 50–200×; we adopt 1-hop hard + 2-hop soft (venue × year × topic).

Domains covered by the arena (informs our domain mix in §8):

```
CS (11): AttentionEfficiency, AutoResearch, DiffusionModels, LLMReasoning,
         MixtureOfExperts, NativeMultimodal, RLAlignment, RobotLearning,
         ScalingLaws, TestTimeCompute, AgentFramework
Biology (3): ProteinDesign, SingleCellFoundation, SpatialTranscriptomics
Chemistry (3): DrugDiscovery, MolecularGeneration, Retrosynthesis
Plus 3 each for: Agriculture, Astronomy, Climate, EarthScience, Ecology,
                 Energy, Materials, Math, Medicine, Neuroscience, Physics
```

There is **no explicit `splits.json` / held-out paper list** in the repo. The denylist must be constructed from the asset files above (see plan.md §2).

---

## 2. Training methods: SFT → GRPO → OPD (2025–2026 state of the art)

### 2.1 On-Policy Distillation (OPD)

**Origin.** Two converging sources: the *Qwen3 Technical Report* (arXiv 2505.09388) introduced "Strong-to-Weak Distillation" with an explicit on-policy phase; the *Thinking Machines Lab* blog "On-Policy Distillation" (Lu et al., Oct 2025) popularized the recipe outside Qwen and gave the cleanest public formulation. Both descend from GKD (Agarwal et al., ICLR 2024) and MiniLLM (Gu et al., ICLR 2024).

**Algorithm (Tinker formulation).** At each step:

1. Sample a trajectory `y ~ π_θ(·|x)` from the **student** (on-policy).
2. For every token `t`, query the **teacher** for `log π_T(y_t | y_<t, x)` (one teacher forward pass over the full sequence).
3. Per-token reward = negative reverse-KL: `r_t = −[log π_θ(y_t|·) − log π_T(y_t|·)]`.
4. Take a policy-gradient step on the student. Objective: `min_θ E_{y~π_θ}[ KL(π_θ(·|y_<t) || π_T(·|y_<t)) ]`.

Reverse-KL is mode-seeking, avoids the exposure bias of forward-KL SFT-on-teacher-traces, and needs no critic.

**Reported wins.** Both Qwen3 and Tinker report ~7–10× fewer gradient steps than GRPO/RLVR to reach the same AIME score, and ~50× total compute when teacher cost is amortized. Critically: higher *Pass@64* than RL, contradicting "RLVR collapses base entropy" — OPD preserves exploration diversity.

**Variants.** GKD (mixture sampling, generalized JSD), MiniLLM (length-normalized reverse-KL), DistiLLM/DistiLLM-2 (skew-KL, 4.3× faster than GKD), Speculative KD (student proposes, teacher replaces poor tokens), Self-Distilled RLVR (frozen prior checkpoint as teacher), Black-Box OPD (top-k only).

**Recipes that work.** Teacher 3–5× student size is the sweet spot. Best-replicated pairs: **Qwen3-32B → Qwen3-8B-Base** and **DeepSeek-R1-671B → Llama-3.1-8B-Base**. Tinker defaults: LR 1e-5 AdamW, batch ~256 prompts × 4 completions, rollout truncated 16–32k tokens, reverse-KL is the only loss (no extra KL-to-reference needed).

**Open-source support (May 2026).** Tinker cookbook (turnkey OPD), TRL's `GKDTrainer`, verl/OpenRLHF/NeMo-RL/SkyRL/AReaL via generic per-token-reward APIs. HuggingFace `huggingfaceh4-on-policy-distillation` Space for cross-tokenizer cases.

### 2.2 GRPO and variants

| Method | What it fixes vs vanilla GRPO | When to use |
|---|---|---|
| **GRPO** (DeepSeekMath, arXiv 2402.03300) | Critic-free PPO, group-normalized advantages | Baseline; β=0.001 default |
| **DAPO** (arXiv 2503.14476) | (1) Clip-Higher (ε_low=0.20, ε_high=0.28) → no entropy collapse on long CoT; (2) Dynamic Sampling (drop all-correct/all-wrong groups); (3) Token-level loss; (4) Overlong reward shaping | **Default choice for long-CoT 8B in 2026.** Beat R1-Zero-Qwen-32B at half the steps |
| **Dr. GRPO** (arXiv 2503.20783) | Removes length normalizer + std normalizer in advantages — both create systematic length and difficulty bias | Two-line patch on top of DAPO; always apply |
| **GSPO** (arXiv 2507.18071) | Sequence-level importance ratio + sequence-level clip; fixes mis-specified token-level ratios | MoE training (Qwen3 235B-A22B). Less critical for dense 8B |
| **VAPO** (ByteDance 2025) | Value-model-based PPO, still beats critic-free on long CoT | Only if you can afford critic memory |
| **VinePPO** (ICML 2025) | Unbiased MC value via branching | 9× faster convergence if rollouts cheap |
| **RLOO** | Low-variance REINFORCE-leave-one-out | Simple alternative to GRPO at small G |

**Reward design for non-verifiable tasks (idea generation).** This is the hardest part for GENE-Arena.

- **Rubrics-as-Rewards** (arXiv 2507.17746): explicit multi-criterion rubric scored by an LLM judge. +31% on HealthBench over Likert. **The right pattern for "is this a good scientific idea."**
- **Rubric-ARM** (arXiv 2602.01511): jointly train rubric generator + judge.
- **ODIN** (arXiv 2402.07319): disentangle length and quality heads to fight length hacking.
- Anti-hacking tricks: reward ensembles, bounded rewards, frozen judge checkpoint, periodic offline judge re-calibration.

**For GENE-bench specifically:** PES already decomposes into Heredity / Variation / Selection (each a 0–100 LLM-judge mean over subitems). We can lift this rubric directly as our reward signal for arena-style RL (see plan §5).

### 2.3 SFT for reasoning — the "less is more" era

| Recipe | Data | Result |
|---|---|---|
| **s1 / s1.1** (arXiv 2501.19393) | 1,000 curated CoT + "Wait"-token budget forcing | Qwen2.5-32B beats o1-preview on AIME/MATH |
| **LIMO** (arXiv 2502.03387) | 817 examples | Qwen2.5-32B → 57.1% AIME, 94.8% MATH |
| **Bespoke-Stratos-17k / 35k** | R1-distilled traces | Sky-T1 reproduction, ~$800 budget |
| **OpenThoughts3-1.2M** (arXiv 2506.04178) | DataComp + Bespoke, QwQ-32B teacher | OpenThinker3-7B: 53% AIME'25 — current SOTA open SFT for 7B |
| **DeepSeek-R1-Distill-Qwen-7B** | 800K CoT (600K reasoning + 200K general) | The single most reliable 7B recipe |
| **DeepSeek-R1-0528-Qwen3-8B** | Same pattern on newer base + R1-0528 teacher | +10pt AIME over Qwen3-8B |

**Key lesson:** quality > quantity for reasoning SFT. DeepSeek explicitly reported direct RL on 7B is *worse* than distillation; **always cold-start an 8B** (SimpleRL-Zoo, arXiv 2503.18892 confirms pure zero-RL on Llama-3-8B-Base diverges).

Standard engineering: loss-on-completion-only (mask prompt with -100), bfd packing, flash-attn 2, document-boundary mask, LR 1–5e-6 cosine, 1–3 epochs. TRL `SFTConfig` and LLaMA-Factory both handle this.

### 2.4 Pipeline shape

The Qwen3-Lite pattern is the most-replicated open recipe:

```
Cold-start SFT (~10–20k curated long-CoT) →
  RLVR/GRPO (200–500 steps, DAPO+Dr.GRPO defaults) →
    OPD (200–300 steps, teacher 3–5× size)
```

Skipping OPD costs ~1 AIME point at 10× the compute. Mixing OPD's dense per-token signal with RLVR's sparse-but-grounded reward gives the highest stable ceiling (Self-Distilled RLVR, arXiv 2604.03128).

---

## 3. Evaluation landscape beyond GENE-bench

To claim generality (not GENE-bench overfit), we need disjoint benchmarks. From the survey of 30+ candidates, the recommended evaluation suite for our 8B model is:

### Recommended eval suite (10 benchmarks)

| # | Benchmark | Axis | Size | License | GENE-bench overlap risk |
|---|---|---|---|---|---|
| 1 | **IdeaBench** (arXiv 2411.02429) | Ideation (biomed) | 2,374 seeds | Public | None — biomed |
| 2 | **ResearchBench** (arXiv 2503.21248) | Ideation (12-discipline, post-2024) | mixed | Public | Low (post-2024 cutoff) |
| 3 | **Nova** (arXiv 2410.14255) | Novelty/diversity (re-run on fresh seeds) | 170 seeds | Public | Controlled |
| 4 | **GPQA-Diamond** | Sci reasoning MCQ | 198 | CC-BY-4.0 (gated) | None |
| 5 | **LAB-Bench** (FutureHouse) | Biology research MCQ | 2,400 (80% public) | Public | None |
| 6 | **BRIGHT** (ICLR 2025) | Reasoning-intensive retrieval | 1,398 queries | Public | None (StackExchange/code/math) |
| 7 | **MultiCite** (NAACL 2022) | Citation intent (lineage at sentence level) | small, multi-label | CC-BY | Low |
| 8 | **MATH-500** | Math no-regression | 500 | MIT | None |
| 9 | **MMLU-Pro** | Broad knowledge no-regression | 12K | MIT | Low |
| 10 | **Arena-Hard-Auto** (science-seed variant) | LLM-judge arena | 500 hard prompts | Public | Controllable |

### Benchmarks to **avoid as primary** (high GENE-bench overlap risk)

LitSearch, QASPER, QASA, SPIQA, AI-Idea-Bench-2025, CoI-Agent Idea Arena seed set, SciMON — all draw heavily from ICLR / NeurIPS / CVPR / ACL / EMNLP papers, which is the same corpus family as GENE-bench. If reported, must publish intersection size with GENE-bench's 1,085 papers.

### Inside GENE-bench: where our model has headroom

From the paper's diagnostic tables:
- T4 Lineage Verification is the weakest tier (GPT-5.5: 16.0%). Most gain available here.
- Tool use helps T2 most (Inheritance Tracing: 25.7→37.9% for GPT-5.5+ClaudeCode).
- PES Heredity dimension benefits most from explicit lineage context. Variation and Selection improve modestly.

This suggests the model should be RL-tuned with a *Heredity-weighted* arena reward and a *T4-heavy* exam reward.

---

## 4. Base model: Qwen3-8B-Base vs DeepSeek-R1-0528-Qwen3-8B

The 8B landscape in May 2026:

| Model | Context | Pretrain | MATH-500 | GPQA | License | Notes |
|---|---|---|---|---|---|---|
| **Qwen3-8B-Base** | 32K (128K YaRN) | 36T tokens, heavy STEM | ~62 base | ~40 base | Apache-2.0 | First-class verl/vLLM/SGLang support |
| **Qwen3-8B (hybrid think/no-think)** | 32K (128K) | post-trained | 88–91 | 55–60 | Apache-2.0 | Already a reasoner |
| **DeepSeek-R1-0528-Qwen3-8B** | 32K | R1-distilled on Qwen3-8B | ~94+ | +10 over Qwen3-8B | MIT (commercial + distill OK) | Best 8B reasoner today |
| **DeepSeek-R1-Distill-Qwen-7B** | 32K | 800K CoT from R1 | ~83 | ~49 | MIT | Older but very stable |
| **Llama-3.1-8B** | 128K | 15T | ~50 | ~32 | Llama-3 community | Lags Qwen3 by 10–30pts |
| **Granite 3.3 8B** | 128K | — | ~70 | mid | Apache-2.0 | GRPO+TPO already applied |
| **OLMo 2 13B** | 4K (extended) | 5–7T (Dolma 2) | mid | mid | Apache-2.0 + ODC-BY data | Pick only if "fully open data" required |

**Recommendation.** Two viable starting points:

- **A: DeepSeek-R1-0528-Qwen3-8B** — best public 8B reasoner, MIT-licensed, distillation permitted. Best if our goal is to *push further* from a strong reasoning prior. Risk: it's already heavily SFT'd on R1 traces, so further SFT mostly overwrites; we'd jump directly to GRPO + OPD on top.
- **B: Qwen3-8B-Base** — clean canvas, full control over the SFT mix, best tokenizer for LaTeX/code/Chinese scientific text, best ecosystem (verl, vLLM, SGLang, LLaMA-Factory, Liger Kernel, TRL). Best if we want to *teach* genome-style structured reasoning from scratch.

**We pick B (Qwen3-8B-Base)** for full pipeline control and clean ablations. Option A becomes the upper-bound comparison baseline.

---

## 5. Existing scientific 7–8B models (what we're comparing against)

- **SciGLM-6B** (NeurIPS D&B 2024) — ChatGLM3-6B + SciInstruct.
- **SciLitLLM-7B / -14B** (ICLR 2025) — Qwen2.5-{7B,14B} with 12.7B-token CPT over scientific literature + SciLitIns SFT. **Their CPT corpus draws from arXiv/PMC** → non-trivial contamination risk for ML papers; ICLR-cohort papers may be in their CPT.
- **Llemma-7B** — CodeLlama-7B + Proof-Pile II (55B tokens). Math-heavy, no biomed.
- **ChemLLM-7B** — InternLM2-7B + ChemData.
- **BioMistral-7B** — Mistral-7B + PubMed Central OA.
- **PMC-LLaMA-7B**, **MedAlpaca-7B**, **ChemDFM**, **Mol-Instructions tuned models** — domain-specific 7B; not idea-generation focused.
- **OpenScholar (Llama-3.1-8B + 45M-paper index)** — retrieval-augmented, not weights-tuned for ideation per se.

**Critical finding:** No public 8B model in May 2026 is explicitly tuned for "scientific idea reasoning" / lineage understanding. This is the niche we're filling.

---

## 6. Public scientific corpora & contamination risk vs GENE-bench

| Corpus | Size | License | Contamination risk |
|---|---|---|---|
| **S2ORC** (arXiv 1911.02782) | 81M papers / 8.1M full-text | ODC-By | **HIGH** — almost certainly contains the 1,085 seed papers + neighbors |
| **peS2o v2** (`allenai/peS2o`) | ~40M papers / ~50B tokens | ODC-By | **HIGH** — S2ORC-derived |
| **arXiv (RedPajama / Dolma)** | ~1.5M papers / ~28B tokens | arXiv non-exclusive (many CC-BY) | **HIGH** for CS / physics / math |
| **PMC OA** | ~4.5M | CC0/CC-BY (commercial bucket) | **MEDIUM-HIGH** for bio / med seeds |
| **OpenReview / PeerRead / ORB** | 14K–89K reviews | mixed; PeerRead NC-research | **VERY HIGH** — ICLR/NeurIPS reviews are the most likely benchmark seed pool |
| **S2 abstracts (S2AG)** | 91M abstracts | ODC-By | HIGH (abstract overlap) |
| **OpenAlex** | 477M works, 107M abstracts | CC0 | MEDIUM (metadata + abstracts only) |
| **DCLM-baseline / FineWeb-Edu** | hundreds of B tokens | CC | LOW-MEDIUM (web, may cache arXiv HTML) |
| **DBLP** | ~7M metadata records | ODC-By | LOW (metadata only) |
| **SciInstruct** (arXiv 2401.07950) | 254K, physics/chem/math/proofs | Apache-ish | **LOW-MEDIUM** — textbook-driven |
| **AbsInstruct** (arXiv 2402.10646) | ~10–20K abstraction items | research-only | LOW |
| **Camel-AI Science** | ~20K per domain (GPT-4 synthetic) | CC-BY-NC | LOW direct; distillation leakage |
| **OpenScholar-200K** (arXiv 2411.14199) | ~200K synthesized scientific QA | ODC-By | **HIGH** — datastore is peS2o-style |
| **Tulu-3 SFT mixture / SciRIFF subset** | 939K total | ODC-By | MEDIUM — SciRIFF draws on S2 papers |
| **OpenThoughts3 / OpenThinker** | 1.2M CoT traces | mixed | LOW direct, but contains math/code/STEM seeds |
| **NaturalReasoning** (arXiv 2502.13124) | 2.8M back-translated reasoning Qs | open | LOW (synthetic) |

**The asymmetry:** the corpora cheapest to use (S2ORC, peS2o, OpenScholar) carry the highest contamination risk. Our SFT pool must be a *filtered* subset of these plus synthetic data generated on a guaranteed-safe paper pool.

---

## 7. Contamination control protocols

### Recipes from production-grade open work

- **Tulu 3** (arXiv 2411.15124): 8-gram token matching. Contamination = >50% token-8-gram match against any training instance. Reject training sets where >2% of any eval suite is matched. Reference code: `allenai/open-instruct/decontamination`.
- **DataComp-LM, Dolma**: MinHash + LSH at 13-grams (128 hash fns, 9 bands × 13 rows per RedPajama recipe), plus Bloom-filter n-gram exclusion.
- **DeepSeek-R1 / V3 tech reports**: describe decontamination of both pre- and post-training data; specific cutoffs not public.
- **Qwen3 tech report**: "anomalous files and instances overlapping with downstream tasks" removed; numeric criterion not specified.

### Tooling

- **NeMo Curator** (NVIDIA-NeMo/Curator): `prepare_task_data`, `find_matching_ngrams`, `remove_matching_ngrams` — GPT-3 / Turing-NLG-style task decontamination at GPU scale. Also exact (MD5), fuzzy (MinHash-LSH), and semantic (embedding) dedup.
- **DataTrove** (HuggingFace): MinHash, sentence dedup, URL filtering.
- **Datasketch**: reference MinHash/LSH Python lib.
- **D4** (arXiv 2308.12284, NeurIPS 2023): embedding-cluster-based dedup; good for semantic dedup against benchmark texts.

### Post-hoc contamination probes (white-box on our own checkpoint)

- **Canary GUIDs** (BIG-Bench style).
- **Min-K% Prob** (ICLR 2024): membership inference via per-token loss.
- **Time Travel** guided-completion (Golchin & Surdeanu, ICLR 2024).
- **ConStat**: performance gap on rephrased benchmark samples.
- Explicit "complete this exact phrase" probes against denylist content.

---

## 8. Synthetic genome-style data generation

GENE-bench's atomic training units are:
- `paper_gene_card` (6 fields: mechanism / niche / observation / limitation / delta / claim)
- `GenomeDiff` (gene-by-gene alignment of a parent–child pair)
- Five evolutionary dynamics labels + Isolation null

We need similar training data **from non-GENE-bench papers**. Approach:

1. **Safety paper pool.** S2ORC/peS2o/arXiv intersected with:
   - Paper year ≤ 2017 (eliminates almost all GENE-bench seeds), OR
   - Domains *outside* GENE-bench's 15 (linguistics, sociology, economics, law, history of science, archaeology, control theory, mechanical engineering — but kept scientific enough that gene cards make sense).

2. **Gene-card extraction prompts** (closest published templates: OpenScholar literature-synthesis prompts; SciInstruct self-reflective annotation prompts). Teacher = Claude Opus 4.6 / GPT-5.5 / Qwen3-235B. Required: every field quotes evidence spans from the source paper.

3. **GenomeDiff pair annotations.** Walk the citation graph in the safety pool; for each cited→citing pair, ask teacher to write a 6-field diff and pick a dynamics label.

4. **Quality verification (no human adjudication available).**
   - Round-trip: second teacher reconstructs the abstract from the card; require BERTScore ≥ 0.7 on held-out 1%.
   - Adversarial swap detection: shuffle one field across two papers, judge model must catch the swap with ≥ 80% accuracy (vs ≤ 25% random).
   - Citation-consistency check: every quoted evidence span must appear verbatim in the source text.
   - Human spot-check: ~200 cards per domain at Likert 1–5.

5. **Risk: teacher LLM has memorized GENE-bench cards.** Mitigation: run Time Travel / Min-K% probes on the teacher *for known GENE-bench papers* before annotating. If a teacher reproduces benchmark phrasing, either route those papers to a less-contaminated teacher (e.g., Llama-3.3-70B) or drop them.

---

## 9. Tooling stack (May 2026)

| Stage | Primary | Alternative |
|---|---|---|
| SFT | **LLaMA-Factory + Liger Kernel** (fastest 8B SFT throughput; ~20% throughput / 40–60% memory savings vs vanilla FSDP) | Axolotl + sequence parallelism (for >32K context) |
| GRPO/RL | **verl** (largest hyperparameter surface, DAPO/Dr.GRPO/GSPO/RLOO/ReMax built-in, FSDP2/Megatron, vLLM/SGLang rollouts) | **OpenRLHF** (cleanest codebase ~8.5K LOC; 1.22–1.68× faster than verl on small models per published benchmarks; simpler Ray+vLLM hybrid) |
| OPD | **Tinker cookbook** (turnkey) or **TRL `GKDTrainer`** | Custom hooks in verl/OpenRLHF (per-token reward API) |
| Inference (rollouts) | **vLLM ≥ 0.7** (explicit `start_weight_update`/`finish_weight_update` IPC, KV-cache invalidation) | SGLang (~29% faster on 7–8B; best for structured outputs); avoid TGI (maintenance mode Dec 2025) |
| Eval | lm-eval-harness, math-evaluation-harness | GENE-bench's own `gene_exam/evaluators/eval_benchmark.py` (already in repo) |

**Memory check.** 8B + GRPO at G=8, ~8K context fits on **2× H100 80GB** with FSDP2 (no offload, no LoRA) at ~32 prompts/batch. 8× A100 80GB needs ZeRO-3 offload or LoRA. LoRA rank-32 fits on **1× H100**. Long-context (32K) doubles memory — consider sequence parallel CP=2 in verl/Megatron.

---

## 10. Risk register (consolidated)

| Risk | Mitigation |
|---|---|
| Direct paper leakage from training corpus | Hard exclude by (s2_id / arxiv_id / doi / normalized title) + MinHash-LSH at Jaccard ≥ 0.7 + Tulu-3 8-gram with > 50% threshold |
| Indirect citation leakage (training paper X cites benchmark paper Y) | Strip related-work / references sections; denylist 1-hop neighbors; soft-exclude 2-hop via venue×year×topic |
| Distillation leakage (teacher LLM memorized GENE-bench) | Probe teacher with Time Travel / Min-K% on known GENE-bench papers; route memorized cases to fallback teacher |
| Format overfit to gene cards | Mix in 30% free-form science QA + 10% summarization + 10% review writing + ≥10% generic chat |
| Hidden test set from ClassifierAgent expansion | Soft exclude venue × year × topic of all 15 covered domains for 2018–2025 ML/Nature/Science venues |
| Reward hacking on PES rubric judge | Frozen judge checkpoint, ODIN-style length disentanglement, reward ensembles, 2-judge cross-check |
| RL collapse on 8B (entropy → 0) | DAPO Clip-Higher (ε_high=0.28), dynamic sampling, β=0.001 KL-to-reference, cold-start SFT first |
| Math/code regression after science tuning | Mix MATH/AIME/GSM8K into SFT and RL prompt sets (~15% weight); track MATH-500 + HumanEval each checkpoint |
| License contamination | Pin every doc to license tag; train public checkpoints only on CC0 / CC-BY / ODC-By / CC-BY-SA |
| Teacher ToS (Claude/GPT) | Use under research carve-outs; document model+date per annotation batch; keep Llama-3.3-70B fallback teacher |

---

## 11. Open questions (decision points for plan.md)

1. **Base model:** Qwen3-8B-Base (clean) vs DeepSeek-R1-0528-Qwen3-8B (head start)? Plan picks Qwen3-8B-Base, with R1-Distill as upper-bound baseline.
2. **OPD position:** before or after GRPO? Qwen3 puts OPD *after* RL; Thinking Machines uses OPD as a replacement. We propose SFT → GRPO → OPD (Qwen3 pattern) and ablate against SFT → OPD only.
3. **Teacher model for SFT + reward judge:** ~~Qwen3-235B vs Claude/GPT~~ **resolved** — local Azure keyless GPT-5.5 is available (`https://t2vgoaigpt4o3.openai.azure.com/`, `azure_cli` auth, $0 cash, ~3.8s verified latency). GPT-5.5 is the top-scoring model on GENE-Exam in the paper itself, so it's the strongest possible teacher *and* the same judge family as the published 3-judge panel. New risk: higher distillation-leakage probability (GPT-5.5 may have memorized benchmark-like content) → must probe before bulk annotation.
4. **Teacher model for OPD:** GPT-5.5 is closed-weights (top-20 logprobs only via API). Two variants kept in plan: A) Qwen3-32B-Thinking served locally for canonical per-token reverse-KL (primary); B) Black-Box OPD (arXiv 2511.10643) on GPT-5.5 top-20 logprobs (frontier-teacher ablation if rate-limit quota permits).
5. **Reward judge:** GPT-5.5 (frozen snapshot) directly — no judge-finetuning cost. Fallback: GPT-5.4 (second Azure endpoint) with deterministic correction.
6. **Denylist closure radius:** 1-hop hard / 2-hop soft. 3-hop is too aggressive (loses ~60% of post-2018 ML corpus).
7. **Held-out internal arena:** generate 30 mock-arena tasks from a 7th-hop citation set (never used in SFT/RL) for our own contamination self-test before reporting GENE-bench numbers.

---

## Key sources

**Methods:**
- Thinking Machines, "On-Policy Distillation" (Oct 2025): https://thinkingmachines.ai/blog/on-policy-distillation/
- Qwen3 Tech Report (arXiv 2505.09388)
- DAPO (arXiv 2503.14476), Dr. GRPO (arXiv 2503.20783), GSPO (arXiv 2507.18071)
- GKD (arXiv 2306.13649), MiniLLM (arXiv 2306.08543)
- s1 (arXiv 2501.19393), LIMO (arXiv 2502.03387), OpenThoughts (arXiv 2506.04178)
- Rubrics-as-Rewards (arXiv 2507.17746), ODIN (arXiv 2402.07319)
- Llama-Nemotron (arXiv 2505.00949), JustRL (arXiv 2512.16649)

**Eval:**
- IdeaBench (arXiv 2411.02429), ResearchBench (arXiv 2503.21248), Nova (arXiv 2410.14255)
- LAB-Bench (FutureHouse), BixBench (arXiv 2503.00096), BRIGHT (ICLR 2025)
- GPQA, MATH-500, MMLU-Pro, Arena-Hard

**Data & contamination:**
- S2ORC (arXiv 1911.02782), peS2o (`allenai/peS2o`), Dolma (arXiv 2402.00159)
- Tulu 3 (arXiv 2411.15124, `allenai/open-instruct/decontamination`)
- DCLM (arXiv 2406.11794), D4 (arXiv 2308.12284)
- NeMo Curator: https://docs.nvidia.com/nemo/curator/
- OpenScholar (arXiv 2411.14199), SciInstruct (arXiv 2401.07950)

**Tooling:**
- verl: https://github.com/volcengine/verl
- OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
- Tinker: https://thinkingmachines.ai/news/announcing-tinker/
- LLaMA-Factory: https://github.com/hiyouga/LLaMA-Factory
- Liger-Kernel: https://github.com/linkedin/Liger-Kernel
- vLLM: https://github.com/vllm-project/vllm

**Unverified claims to flag in writeups:**
- OPD's "9–30× cheaper than RL" comes from Thinking Machines' own math benchmarks — not yet replicated on non-math (scientific ideation) tasks.
- "8B can reach frontier reasoning via distillation" — well-supported for math/code; **unverified for open-ended scientific idea generation**.
- DeepSeek's "RL on 7B is worse than distillation" was made for math/code, not science ideation.
