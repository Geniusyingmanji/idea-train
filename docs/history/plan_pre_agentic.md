# Plan: 8B Scientific-Idea Model — SFT → evo-OPD

> **PLAN UPDATE (2026-05-18).** Stages 2 (GRPO) and 3 (evo-OPD) are **collapsed into a single Stage 2: evo-OPD**. evo-OPD's "verifier-anchored decoupling" component is functionally GRPO with a verifier reward, so the previous two-stage GRPO→OPD pipeline was double-counting that signal. The new flow is **SFT → evo-OPD**. Standalone GRPO is kept as one of the pipeline-level ablations (Variant 2 below), and the SFT→GRPO→evo-OPD pipeline is kept as another ablation (Variant 3), so we can measure whether the GRPO warm-up actually helps before evo-OPD. Old sections 4 (GRPO standalone) and 5 (vanilla OPD) below are preserved verbatim as **reference for the ablation variants** — they are not the main path.

> Companion to `survey.md`. This is the actionable training plan.
> Target: 8B scientific-idea reasoning model that improves on GENE-bench (Exam ≥ 28% exact, Arena PES ≥ 82 in Lineage setting) and does not regress on math/code/GPQA, with strict no-cheating against GENE-bench's ~5–7K paper denylist.

---

## 0. Decisions locked in

| Decision | Choice | Rationale (one line) |
|---|---|---|
| Base model | **Qwen3-8B-Base** (Apache-2.0, 32K native / 128K YaRN) | Clean canvas, best 8B STEM pretraining, best ecosystem |
| Upper-bound baseline | DeepSeek-R1-0528-Qwen3-8B | Already-distilled reasoner; tells us the ceiling of "just take an existing R1 distill" |
| SFT trainer | LLaMA-Factory + Liger Kernel | ~20% throughput / 40–60% memory savings on 8B FSDP |
| RL trainer | verl (primary), OpenRLHF (fallback) | DAPO/Dr.GRPO/GSPO built-in; vLLM+SGLang rollouts |
| OPD framework | Tinker cookbook recipe ported to verl per-token-reward API | Tinker pattern is the cleanest public spec |
| Inference (rollouts) | vLLM ≥ 0.7 | Explicit weight-swap IPC for async RL |
| Teacher — SFT synthetic data | **GPT-5.5 via Azure keyless** (`https://t2vgoaigpt4o3.openai.azure.com/`, `azure_cli` auth) | Free (Azure managed identity), strongest model on GENE-Exam (23.1% direct, 27.3% w/ harness), reuse `IdeaEvolving/agent/llm_client.py` |
| Teacher — OPD per-token | **Qwen3-14B-Thinking** (open weights, full logits, ~1.75× student), served locally via vLLM on 1× A100 80GB | Canonical Tinker reverse-KL recipe; avoids Black-Box truncation (~30% sample-efficiency hit) and Azure rate limits. Teacher/student ratio is below the recommended 3-5× — see `lv_opd_plan.md` §3.1 for the trade-off note. GPT-5.5 Black-Box kept as evo-OPD ablation L7 |
| Reward judge (Arena RL) | **GPT-5.5 (frozen snapshot)** directly | Same judge family as the paper's 3-judge panel; no judge-finetuning cost; just freeze checkpoint date |
| Backup teacher / cross-model probe | GPT-5.4 (Azure keyless, second endpoint `t2vgoaigpt4o.openai.azure.com`) + Claude Opus 4.7 via v3cm | Used for contamination probing and 5% calibration |
| Compute target | **4× A100 80GB (shared node, 2 GPUs currently busy with other workloads)** | Reality, not the 8× H100 originally assumed. Forces: smaller teacher (14B), SFT ctx 8K (not 16K), longer wall-clock (~2× vs H100), and GPU-coordinated scheduling with other tenants. ~10K GPU-hours total (was ~5K) |

---

## 1. Goals & success metrics

**Primary (must hit):**
- GENE-Exam main-challenge: **≥ 28%** exact-match (matches GPT-5.5+ClaudeCode harness; current best 8B baseline likely ~17–20%).
- GENE-Arena PES (Lineage setting, direct LLM): **≥ 82** (between DeepSeek-V4-Pro 82.7 and Claude Opus 4.7 82.6).
- GENE-Arena ELO (Lineage, cross-model battle vs DeepSeek-V4-Pro/Claude/Gemini/Qwen): **≥ 1100**.

**Secondary (must not regress > 2 abs pts vs Qwen3-8B-Base + RL-only baseline):**
- MATH-500, AIME-2024, GSM8K (math preservation)
- GPQA-Diamond (scientific reasoning generality)
- MMLU-Pro (broad knowledge)
- HumanEval, MBPP (code preservation)

**Generality (report, don't gate):**
- IdeaBench Insight Score, ResearchBench three sub-tasks, Nova diversity, LAB-Bench LitQA2+ProtocolQA, BRIGHT, MultiCite, Arena-Hard-Auto.

**Honesty (must publish):**
- Min-K% Prob hit rate on the 5–7K denylist; Time-Travel guided-completion accuracy on 100 random gene-card fields. Target: zero exact memorization, ≤ 2% performance gap on a paraphrased internal mirror.

---

## 2. Stage 0 — Contamination firewall (1 week, 1× A100 + CPU)

### 2.1 Build the denylist

Single script `tools/build_denylist.py` reads from `/IdeaEvolving/`:

```python
sources = [
    "data/genome_db/paper_gene_cards.json",        # ~2,076 papers
    "data/genome_db/trace_graphs.json",            # 90 traces × ~50 nodes
    "data/genome_db/trace_gene_graphs.json",
    "gene_arena/task/*.json",                       # 50 frontier traces
    "data/paper_db/paper_db.json",                 # 29,472 paper registry
]
```

Output: `denylist_v1.parquet` with columns
`{normalized_title, year, s2_id, arxiv_id, doi, openalex_id, source_file, domain}`.

Expand via OpenAlex API: for each paper, pull `referenced_works` (1-hop). Optionally `related_works` and `cited_by` (1-hop). Final hard denylist ≈ 5–7K papers; soft denylist (2-hop venue×year×topic) ≈ 80–150K.

### 2.2 Filtering cascade for any external corpus

Apply in order, log every drop reason:

1. **Hard exclude by ID.** Drop any record whose `(s2_id | arxiv_id | doi | openalex_id)` is in `denylist_v1`. Also drop by normalized-title (lowercase, strip punctuation/whitespace) edit distance ≤ 2.
2. **MinHash-LSH exclude.** 128 perms, Jaccard ≥ 0.7 over abstract+intro vs denylist abstract+intro. Use `datasketch` or NeMo Curator.
3. **Tulu-3-style 8-gram exclude.** Drop documents with > 50% token overlap (8-gram) against any denylist paper's abstract+conclusion. Reference: `allenai/open-instruct/decontamination`.
4. **Soft exclude.** ICLR / NeurIPS / ICML / Nature / Science / Cell / PRL papers from 2018–2025 in the 15 GENE-bench domains, *unless* explicitly outside the 2-hop neighborhood. Toggleable; default ON for first SFT run.
5. **Semantic dedup against arena card text.** Embed each `paper_gene_card` content (`nomic-embed-text-v1.5`); drop training docs at cosine ≥ 0.85 against any card.

### 2.3 Safety paper pool

After filtering, retain a **two-pool** training corpus:

- **Pool A — High safety, broad coverage.** S2ORC/peS2o/arXiv papers with `year ≤ 2017` OR domain ∉ GENE-bench's 15. Estimated 30–40M papers. Used for: gene-card extraction, GenomeDiff pairs, free-form science QA.
- **Pool B — Domain-matched, post-filter.** Post-2018 papers in the 15 covered domains that survive the entire cascade + soft exclude. Estimated 1–3M papers. Used sparingly — only for "domain familiarity" SFT samples (≤15% of mix).

### 2.4 Audit hooks

- Reject the entire SFT mix if > 2% of GENE-Exam instances match training docs at 8-gram > 50%.
- Reject if any single training doc shares an `s2_id` with the denylist.
- Manual spot-check 200 random training docs per domain, by human annotator (graduate-level).

---

## 3. Stage 1 — SFT data curation (2–3 weeks, mostly CPU + teacher API)

### 3.1 Data mixture (target: ~80K total examples — "less is more" regime)

| Component | Share | Source | Format | Notes |
|---|---|---|---|---|
| **Genome-card extraction** | 32% (~25.6K) | Pool A (domain-balanced per §3.4) | `{paper_text} → {6-field genome card with evidence spans}` | Teacher = Qwen3-235B-Thinking; quality-verified per §3.3 |
| **GenomeDiff pair annotation** | 16% (~12.8K) | Pool A citation pairs | `{paper_A_card, paper_B_card} → {gene-by-gene fates + dynamics label + rationale}` | Citation pairs entirely within Pool A |
| **Lineage trace reconstruction** | 8% (~6.4K) | Pool A citation chains of length 3–6 | `{shuffled genome cards} → {ordered lineage + per-edge dynamics}` | Direct T2/T3 capability training |
| **Lineage verification** | 8% (~6.4K) | Pool A traces with synthetic intruders/wrong-step/missing-link/citation-conflict perturbations | `{proposed lineage} → {valid? if not, what's wrong + repair}` | Direct T4 capability training — highest paper headroom |
| **Lineage-grounded idea generation** | 8% (~6.4K) | Pool A traces with explicit "open question" prompts | `{trace + question} → {structured proposal: Problem/Mechanism/ExpectedContribution + lineage_connection field}` | Direct Arena prep |
| **Free-form science QA** | 15% (~12K) | OpenScholar-200K **after §2.2 cascade** + SciInstruct + SciToolBench | open chat QA | Anti-format-overfit |
| **General reasoning (math/code/STEM)** | 10% (~8K) | OpenThoughts3 STEM subset + NaturalReasoning sciences subset | long-CoT | No-regression on MATH/AIME/GPQA |
| **General chat / instruction following** | 3% (~2.4K) | Tulu-3 SFT mixture **minus SciRIFF** | various | Preserves base instruction-following |

Total ≈ 80K examples. Following s1 / LIMO evidence, quality is much more important than quantity at this scale.

### 3.1.5 Teacher infrastructure (Azure keyless GPT-5.5)

Reuse the existing IdeaEvolving plumbing instead of building new clients:

```python
# uses /IdeaEvolving/agent/llm_client.py + /IdeaEvolving/config.py
from agent.llm_client import LLMClient
teacher = LLMClient(
    provider="azure",
    model="gpt-5.5",
    azure_endpoint="https://t2vgoaigpt4o3.openai.azure.com/",
    azure_auth_mode="azure_cli",          # managed identity, no key
    api_version="2024-12-01-preview",
    max_output_tokens=8192,
)
```

Operational notes:
- **Auth:** `az login` once (managed identity already trusted on this tenant: `AACars-IMG-MSRA`); token auto-refreshes.
- **Latency:** ~3.8s for short responses (verified via `scripts/preflight_api.py`); longer for full gene-card extraction. Plan for ~6–15s per call at our prompt sizes.
- **Concurrency:** start at 32 parallel calls, tune up to rate-limit ceiling. Total ~50K teacher calls for SFT (cards + diffs + traces + verification + generation) — at 32 concurrency and 10s avg latency, ≈4.3 hours wall-clock per pass.
- **Fallback:** GPT-5.4 endpoint (`t2vgoaigpt4o.openai.azure.com`) for retries on 429/5xx.
- **Cost:** $0 cash; only opportunity cost on the shared Azure quota — coordinate with other workspace users (gca/gca-skill/etc. also use this endpoint).
- **Logging:** every call writes `{prompt, response, model, timestamp, input_tokens, output_tokens, finish_reason}` to `data/teacher_logs/` so we can replay or audit.

### 3.2 Teacher prompts (canonical templates we'll author)

The repo has no pre-built genome-extraction prompt for arbitrary papers. We author 5 reusable templates:

1. `prompt_genome_extract.md` — extract 6-field gene card with evidence spans (every field ≤ 40 words; must quote source).
2. `prompt_genome_diff.md` — gene-by-gene fate (`INHERITED | MUTATED | LOST | NOVEL | HYBRIDIZED`) + primary driver + dynamics label.
3. `prompt_trace_reconstruct.md` — given shuffled cards, return ordering + per-edge dynamics with rationale.
4. `prompt_lineage_verify.md` — given proposed lineage, return valid/invalid + specific failure mode (intruder / wrong-step / missing-link / citation-conflict).
5. `prompt_idea_generate.md` — given trace + open question, return structured proposal with mandatory `lineage_connection` field naming parent(s) + repaired limitation.

All prompts embed the same dynamics decision tree and field schema as `agent/genome_differ.py` and the GENE-Exam evaluator, so downstream answer formats match.

### 3.3 Quality verification pipeline (no human adjudication needed for the bulk)

Every generated example goes through:

- **Schema check.** Strict JSON validation against `paper_gene_card`, `GenomeDiff`, `lineage_trace` schemas.
- **Evidence-span citation check.** Every quoted span in a field must appear verbatim in the source text (regex match). Drop if any span fails.
- **Round-trip reconstruction.** Second teacher reconstructs the abstract from the card; require BERTScore ≥ 0.7 vs original abstract on 1% sample.
- **Adversarial swap detection.** Shuffle one field across two papers; judge model must detect with ≥ 80% accuracy on real cards (vs ≤ 25% random). Run on 5% sample.
- **Difficulty calibration.** Use Qwen3-8B-Base zero-shot to answer the SFT prompt; keep only examples where base is wrong (rejection sampling — this prevents "easy filler" examples).
- **Human spot-check.** ~200 cards per domain (~2000 total) by graduate-level annotator at Likert 1–5; required mean ≥ 4.0.

### 3.4 Domain mix (forces generality, prevents CS overfit)

Per §6 of survey.md:

```
Biology (PMC OA, non-denylist):                18%
Chemistry (PMC OA + RSC OA):                   10%
Physics (arXiv ≤ 2017):                        10%
Math (arXiv ≤ 2017):                            8%
Medicine (PMC OA non-clinical-trial):          10%
Neuroscience (PMC OA):                          8%
Materials/Energy/Climate/Earth/Agri/Astro mix: 18%
CS/ML (pre-2018 ACL/EMNLP/JMLR + non-denylist post-2018): 10%
OOD sanity (econ/linguistics/social sci):       8%
```

Note CS share is intentionally **only 10%**, even though GENE-bench's CS coverage is 11/50 arena tasks (22%) and most exam papers are CS. This is deliberate: training on bio/chem/physics lineage patterns and *evaluating* on CS lineage tests the abstraction. If the model only learns CS-shaped genome cards, it overfits.

### 3.5 SFT training

- **Recipe:** LLaMA-Factory + Liger Kernel + FlashAttention-2.
- **Hyperparameters:** LR 2e-5 cosine, warmup 3%, batch 64 (micro 1, grad-accum), 2 epochs, max_seq_len 16,384, packing with bfd, loss-on-completion-only.
- **Hardware:** **4× A100 80GB FSDP2** (2 effective if other workloads occupy 2 GPUs). Likely needs ZeRO-3 offload or LoRA for 16K context; default to 8K context with full-param finetuning. Estimated ~40-60 hours wall-clock on 4× A100 vs the original 24h on 8× H100.
- **Checkpoints:** every 500 steps; eval on a held-out 1% slice of each component + GENE-Exam main-challenge sample of 200 instances.

---

## 4. Stage 2 — GRPO training (1.5–2 weeks, 4× A100) — ABLATION ONLY, see header

> Per the plan update, **this section's pipeline is no longer the main path.** Stage 2 in the production pipeline is evo-OPD (§5). The reward design, prompt sets, and anti-hacking diagnostics described here ARE reused inside evo-OPD's verifier-anchored decoupling component, so this section is still load-bearing as the spec for that component. The "Algorithm: DAPO + Dr. GRPO patches" subsection only fires if you run the standalone-GRPO ablation (Variant 2).

### 4.1 Prompt sets

Three RL prompt pools, sampled in equal proportion per batch:

1. **Verifiable GENE-Exam-style prompts** (~5K, from Pool A annotations). Format identical to T1–T4 instances; answer extracted by regex; reward = 1.0 if exact match else 0.0. **All-or-nothing scoring matches the benchmark.** Includes composite suffixes (Dynamics, Driver, GenomeFieldFate, TF-Verify, ContribType).
2. **Verifiable math/code preservation** (~2K, MATH/AIME/HumanEval). Same exact-match reward. Prevents reasoning regression.
3. **Arena-style open-ended prompts** (~3K, generated proposal tasks on Pool A traces). Reward = rubric-as-reward (§4.2).

### 4.2 Reward design

Three reward heads, summed with fixed weights `w_exam=0.4, w_math=0.2, w_arena=0.4`:

- **Exam reward.** Exact match (0/1). All-or-nothing across composite suffixes. Same scoring rule as `gene_exam/evaluators/eval_benchmark.py`.
- **Math reward.** sympy / numeric tolerance for math; pytest for code (HumanEval). 0/1.
- **Arena reward.** PES-style rubric scored by **GPT-5.5 (Azure keyless, frozen snapshot date)** as the judge — same model family as the paper's 3-judge panel:
  - 3 dimensions (Heredity, Variation, Selection), each 0–10 via 4 subitems (PES schema lifted from `gene_arena/`).
  - **Heredity weight × 1.5** (PES diagnostic in the paper shows lineage context primarily improves Heredity — that's the lever we want to pull).
  - Frozen judge: pin `model=gpt-5.5`, `api_version=2024-12-01-preview`, snapshot the prompt template + few-shot exemplars in a hash-tagged file; never edit during the run.
  - Cross-model recalibration every 200 RL steps: run 100 held-out completions through both GPT-5.5 and Claude Opus 4.7; fit a single per-dimension affine to keep judge stable. Drift > 0.5 abs triggers refresh.
  - ODIN-style length disentanglement: regress length out of judge score before training step.
  - **Rate-limit budget:** ~2K judge calls per RL step (256 prompts × 8 samples ÷ batching of 2) × 500 steps = ~1M calls. Plan for 64-concurrency on GPT-5.5; verify quota before kickoff. If quota insufficient, swap to GPT-5.4 judge with a fixed 0.95× score correction.

### 4.3 Algorithm: DAPO + Dr. GRPO patches

- **DAPO** defaults: Clip-Higher (`ε_low=0.20, ε_high=0.28`), Dynamic Sampling (drop all-correct/all-wrong groups), Token-level loss, Overlong reward shaping (16K hard cap, 4K soft buffer).
- **Dr. GRPO patches**: remove length normalizer, remove std normalizer in advantages.
- **KL coefficient** β = 0.001 vs SFT checkpoint (reference policy).
- **Group size G = 8.**
- **Batch size** = 256 prompts × 8 samples = 2048 rollouts per step.
- **Mini-batch** = 64 prompts per gradient update; 4 PPO epochs per batch.
- **LR** = 5e-7 (lower than s1's 1e-5 because we're past cold-start).
- **Max prompt** = 8K, **max response** = 16K.
- **Total** ≈ 300–500 RL steps. Stop when GENE-Exam held-out plateaus.

### 4.4 Engineering

- **Trainer:** verl with FSDP2 + vLLM rollouts (separate process group on dedicated GPUs).
- **Rollout:** vLLM 0.7 + `start_weight_update` / `finish_weight_update` IPC for in-place weight sync; `clear_cache=False` for throughput.
- **Async:** turn on verl's async-RL mode (rollouts overlap with training).
- **Memory (revised for 4× A100):** 8B × FSDP2 + G=8 rollouts at **8K context** fits 4× A100 80GB at batch 128 (half the original plan). For 16K context, switch to ZeRO-3 offload or LoRA rank-32 (~30% throughput cost).

### 4.5 Anti-hacking diagnostics (every 50 steps)

- Track entropy of policy distribution; alert if drops > 30% in 50 steps (sign of collapse → reduce learning rate or increase `ε_high`).
- Track avg response length; alert if grows > 50% (length hacking → re-tune ODIN regression).
- Track judge score vs Claude re-calibration set every 200 steps; alert if drift > 0.5 abs.
- Track held-out GENE-Exam every 100 steps; alert if drops > 1 abs pt (overfit to RL prompts).

---

## 5. Stage 2 — evo-OPD (MAIN, 1.5–2 weeks, 3× A100 student + 1× A100 teacher)

> Renumbered from "Stage 3" per the plan update. evo-OPD subsumes GRPO via its verifier-anchored decoupling component, so this is the only post-SFT stage in the production pipeline. See `lv_opd_plan.md` for the algorithm spec; this section keeps the operational details (reward, prompt sets, schedule).

### 5.1 Why OPD here

After GRPO, the policy is sharp on the exam reward but may have token-level distribution drift (modes pruned too aggressively). OPD with a strong teacher restores teacher-grade per-token entropy at low compute, and Qwen3 tech report + Thinking Machines blog both report PES-like preservation of Pass@K.

### 5.2 Teacher choice

**Locked: Qwen3-14B-Thinking** served locally via vLLM on **1× A100 80GB** (TP=1, bf16) on the shared 4× A100 node. Open weights → full per-token log-probs → canonical Tinker reverse-KL recipe works as-is. Teacher is **~1.75× student** — below the 3-5× sweet spot; see `lv_opd_plan.md` §3.1 for trade-off note. Same tokenizer family as Qwen3-8B-Base, so no cross-tokenizer alignment needed.

GPT-5.5 Black-Box (top-20 logprobs, arXiv 2511.10643) is kept only as **evo-OPD ablation L7**, to answer "is the frontier teacher worth the truncation hit + contamination risk?". GPT-5.5 remains the SFT-data teacher (Stage 1) and the GRPO Arena reward judge (Stage 2); only the OPD teacher uses the open model.

### 5.3 OPD recipe (Tinker-style)

For each prompt `x` sampled from a held-out Pool A genome-task prompt set:

1. Student `π_θ` (post-GRPO checkpoint) generates `y ~ π_θ(·|x)`.
2. Teacher `π_T` runs a single forward pass to get `log π_T(y_t | y_<t, x)` for every token `t`.
3. Per-token reward `r_t = −[log π_θ(y_t|·) − log π_T(y_t|·)]`.
4. Policy-gradient step on student. No critic, no extra reference KL.

### 5.4 Hyperparameters

- **LR** 1e-5 (AdamW, β1=0.9, β2=0.95, weight_decay=0.1).
- **Batch** ≈ 256 prompts × 4 completions.
- **Rollout truncation** 16K tokens (matching GRPO).
- **Steps** 200–300.
- **Prompt set:** **separate** from the GRPO prompt set — pull from the same Pool A genome-task templates but a different random split, plus 20% Arena-style open prompts so the model keeps the proposal-writing skill.

### 5.5 Ablations to ship in the paper

> **Stage 3 now uses evo-OPD** (see `lv_opd_plan.md`). The 7 main ablations below are the *pipeline-level* ablations; evo-OPD's internal component ablations (L1 − A / L2 − B / L3 − C / L4 + AR / L5 + MT / L7 GPT-5.5 Black-Box teacher) live in `lv_opd_plan.md` §5.

1. SFT only.
2. SFT → GRPO.
3. SFT → GRPO → **evo-OPD** (Qwen3-14B teacher, all 3 components on) — main result.
4. SFT → GRPO → vanilla OPD (Qwen3-14B teacher) — baseline B0 from `lv_opd_plan.md`; isolates evo-OPD's added components from the teacher choice.
5. SFT → evo-OPD (no GRPO middle stage) — tests Thinking Machines' "OPD replaces RL" claim on a non-math task.
6. Reward-weight ablation: w_exam=1.0/0.0, w_arena=0.0/1.0, w_heredity=1.0/1.5/2.0.
7. Domain-mix ablation: 50% CS vs 10% CS in SFT data (tests the generality hypothesis).
8. **Teacher contamination ablation:** SFT with GPT-5.5-generated cards vs Qwen3-235B-generated cards on the same Pool A. If GPT-5.5 has memorized GENE-bench-like content, the GPT-5.5-trained variant should show higher Min-K% scores on the denylist — directly measures the leakage.

### 5.6 evo-OPD v6 — Arena-Rank reward (post-PES-collapse pivot, 2026-05-19)

After the 900-idea PES study showed evo-OPD-v5 hits the same compressed ceiling as v8 strict-SFT (PES ≈ 56.5, −1.3 to −1.6 vs v3 SFT, p<0.005), we identified the root cause as a missing creativity-direction reward — `judge_PES` inside `v(y)` was **pointwise** and ArenaRL (arXiv 2601.06487) shows that pointwise scoring on open-ended tasks → "discrimination collapse" → policy converges to a safe high-mean mode.

**Fix in v6**: replace pointwise judge_PES with a **seeded single-elimination tournament rank** over the K=8 rollouts in each GRPO group. O(K) GPT-5.5 calls per prompt; quantile-converted rank → z-normalized → fed as additional advantage signal alongside existing verifier (schema/evidence/dynamics) and lineage-consistency. Full spec, cost model, and ablation matrix in `evo_opd_arena_rank.md`. Adds ~5-6 days of work + ~10-20 hr of judge wall-clock (Azure keyless, $0).

**Headline ablations specific to v6** (run after the 8 pipeline-level ablations above):

| Variant | What it isolates |
|---|---|
| v6 main (α=0.25, β=0.50, γ=0.20) | recovers PES + retains T3 |
| v6 − arena (β=0) | reproduces v5 collapse — confirms arena reward is the active ingredient |
| v6 − pointwise (replace tournament with pointwise PES @ same β) | **the key ablation**: isolates "tournament structure" from "judge identity" — if tournament wins here, it's the strongest evidence we have that ArenaRL's thesis ports to scientific idea gen |
| v6 K=4 | cost-effective recipe for follow-on researchers |

---

## 6. Evaluation protocol

### 6.1 In-loop eval (every checkpoint)

Cheap, runs in < 60 min on 1× A100 with vLLM:
- GENE-Exam main-challenge **200-instance smoke** (10% sample stratified across T1–T4).
- MATH-500 (full, 500 problems).
- GPQA-Diamond (full, 198 problems).
- A held-out 100-instance internal arena (auto-PES via Qwen3-14B judge).

### 6.2 Milestone eval (after each stage)

Full benchmark suite (8–16 hours on 1× A100):
- **GENE-Exam:** full main-challenge profile (1,029 instances) via `gene_exam/evaluators/eval_benchmark.py`. Also `full` profile (1,380) for the appendix.
- **GENE-Arena:** all 30 active tasks × 3 settings (Question / Library / Lineage), full PES + ELO via `gene_arena/run_arena.py`. Battle against the published frontier panel (GPT-5.5, Qwen3.6-Max-Preview, Kimi-K2-Thinking).
- **Generality suite (10 benchmarks):** IdeaBench, ResearchBench, Nova, GPQA-Diamond, LAB-Bench, BRIGHT, MultiCite, MATH-500, MMLU-Pro, Arena-Hard-Auto (science seed).
- **No-regression:** AIME-2024, HumanEval, MBPP.

### 6.3 Contamination self-test (before any public checkpoint)

- **Canary probe:** insert 100 canary GUIDs into the denylist papers' synthetic gene cards; checkpoint must not reproduce them.
- **Min-K% Prob** on full denylist (5–7K papers). Target: distribution indistinguishable from a random sample of Pool A.
- **Time Travel guided-completion** on 100 random gene-card fields from the denylist (using held-out abstract intros as prompts). Target: zero exact reproductions.
- **Paraphrased mirror test:** rewrite 100 GENE-Exam instances by hand, ask the model both versions; performance gap must be ≤ 2 abs pts.

If any test fails → re-run §2.2 cascade with stricter thresholds, regenerate SFT data, restart from Stage 1.

---

## 7. Compute budget & timeline

**Revised for 4× A100 80GB shared node (2 GPUs occupied by other workloads, so effective: 2× A100 baseline, peaks to 4 when other tenants idle).** A100 ≈ 0.5× H100 throughput in bf16, so GPU-hours roughly double vs the original H100 plan.

| Stage | Wall-clock | GPU-hours | Notes |
|---|---|---|---|
| 0. Denylist + filter pipeline | 1 week | ~50 (mostly CPU) | OpenAlex API + MinHash/8-gram on filtered S2ORC subset; almost no GPU |
| 1a. Synthetic data generation (GPT-5.5 Azure keyless) | 1.5–2 weeks | — | **$0** cash; bound by Azure rate limit. ~50K calls × ~10s avg ÷ 32 concurrency ≈ 4-6h pure compute per pass |
| 1b. Quality verification + dedup | 3-5 days | ~100 | Round-trip + adversarial swap on 5% sample |
| 1c. SFT training | 2-3 days | ~500 | 80K examples × 2 epochs × 4× A100 at 8K context |
| 2. GRPO training | 1.5-2 weeks | ~1500 | 300-500 steps, async rollouts; throughput cut roughly in half vs 8× H100 |
| 3. evo-OPD | 1 week | ~900 student + ~250 teacher | 200-300 steps; 3× A100 student + 1× A100 Qwen3-14B teacher (vLLM TP=1) |
| 4. Full eval suite | 1-2 days | ~80 | 10 generality benchmarks + GENE-bench full |
| 5. Ablations (5 prioritised: SFT-only, vanilla-OPD baseline B0, SFT→evo-OPD, domain-mix, teacher-contamination) | 3-4 weeks | ~5000 | Each: partial SFT + GRPO + evo-OPD on a smaller scale; LoRA-only variants to cut cost |
| **Total (main + 5 priority ablations)** | **~9-11 weeks** | **~8000-9000 GPU-hours** | **~$0 API cash** (Azure keyless), only rate-limit coordination |

If only the main run is needed: ~5-6 weeks, ~3500 GPU-hours, $0 API cash. Full 8-ablation budget would push to ~14 weeks / ~12K GPU-hours on this hardware — recommend trimming ablations rather than blocking the main run.

**Rate-limit pre-flight (Day 0):** before kickoff, measure GPT-5.5 sustained tokens-per-minute on `t2vgoaigpt4o3` with `scripts/preflight_api.py` at 32, 64, 128 concurrency. Estimate maximum sustainable rate × required call volume for each stage. If Stage 2 reward judge (~1M calls) exceeds quota, fall back to GPT-5.4 judge + per-step deterministic correction.

---

## 8. Risk mitigations (consolidated from survey §10)

| Risk | Detection signal | Mitigation |
|---|---|---|
| Direct paper leakage | s2_id/arxiv_id match in dataset | Stage 0 §2.2 cascade; reject SFT batch if violated |
| Indirect citation leakage | Min-K% > random baseline on 1-hop neighbors | Strip references sections; denylist 1-hop hard, 2-hop soft |
| Distillation leakage | Teacher reproduces denylist phrasing | **GPT-5.5 is the top GENE-Exam scorer in the paper → high a priori risk.** Probe teacher before bulk annotation (Time-Travel completion on 100 denylist papers; canary GUID test; Min-K% of teacher logprobs on denylist abstracts). If teacher reproduces benchmark phrasing > 5% of probes, swap to Qwen3-235B-Thinking for that subset and re-probe |
| Format overfit (only gene cards) | Drop on free-form QA benchmarks | 28% non-genome share in SFT mix; track Arena-Hard each checkpoint |
| Reward hacking on PES judge | Judge score climbs while held-out Exam stalls | Frozen judge + weekly Claude calibration + ODIN length head |
| RL entropy collapse | Policy entropy drops > 30% in 50 steps | DAPO Clip-Higher; β=0.001 KL; reduce LR |
| Math/code regression | MATH-500 / HumanEval drops > 2 abs | 30% math/code/STEM in SFT; 20% in RL prompts |
| Teacher ToS (Claude/GPT) | N/A (legal) | Use only under research carve-outs; document model + date; keep open fallback |
| Hidden test set from ClassifierAgent | Sudden GENE-Exam jump > 5 abs without method change | Soft-exclude venue×year×topic 2018–2025 in 15 domains |

---

## 9. Deliverables

By end of project:

1. **Model checkpoints** (HuggingFace, MIT or Apache-2.0):
   - `idea-8b-sft` (after Stage 1)
   - `idea-8b-grpo` (after Stage 2)
   - `idea-8b-opd` (final, main release)
2. **Training data** (HuggingFace, ODC-By):
   - `idea-train-80k` — full SFT mix with provenance + denylist proof
3. **Code repo** (`idea_train/`):
   - `tools/build_denylist.py` — denylist construction
   - `tools/filter_corpus.py` — 5-stage cascade
   - `tools/generate_genome_data.py` — teacher prompts + verification
   - `train/sft.yaml` — LLaMA-Factory config
   - `train/grpo.yaml` — verl config
   - `train/opd.yaml` — Tinker-style OPD loop
   - `eval/run_full_suite.sh` — 10-benchmark generality eval
   - `eval/contamination_probe.py` — canary + Min-K% + Time-Travel + paraphrase-mirror
4. **Paper appendix material**:
   - Complete denylist (paper IDs + sources)
   - Per-stage held-out GENE-Exam curve
   - 7 ablations (§5.5)
   - Contamination probe results

---

## 10. Open decisions (defer until Week 1 of execution)

1. **Final closure radius:** 1-hop hard / 2-hop soft is the default; revisit if denylist filtering removes > 80% of post-2018 ML papers (may relax 2-hop to venue+year only, drop topic).
2. **Teacher choice for synthetic genome cards:** GPT-5.5 (Azure keyless) is default; if Stage-0 contamination probe shows > 5% memorization on denylist papers, route those papers to Qwen3-235B-Thinking. Run a small head-to-head on 100 papers before committing.
3. **GPT-5.5 rate-limit sufficiency:** Day-0 quota measurement decides whether Stage-2 reward judge can be GPT-5.5 or must fall back to GPT-5.4.
4. **OPD teacher (locked):** Qwen3-14B-Thinking served locally (full logits, canonical reverse-KL). GPT-5.5 Black-Box kept only as evo-OPD ablation L7.
5. **GRPO mini-batch ratio:** if entropy collapses with mini-batch 64, drop to 32 (more PPO updates per batch).
6. **OPD prompt set composition:** start with 80% genome-task / 20% Arena open prompts; if Arena PES drops during OPD, raise to 40% Arena prompts.
7. **Ablations to actually run:** prioritize 1 (SFT-only), 3 (full pipeline, Variant A), 5 (SFT→OPD), 7 (domain mix), 8 (teacher contamination). Drop 2, 4, 6 if compute budget tight.

---

## 11. Reproducibility checklist (NeurIPS-style)

- [ ] Public checkpoint (HuggingFace)
- [ ] Full training data with provenance + denylist
- [ ] Denylist published with paper IDs
- [ ] All hyperparameters in `train/*.yaml`
- [ ] Eval scripts produce same numbers given same seed
- [ ] Contamination probe results in appendix
- [ ] Teacher model + date + prompt templates published
- [ ] Cost / GPU-hour table in appendix
- [ ] License inheritance documented per data source
