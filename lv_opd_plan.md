# evo-OPD: Evolutionary On-Policy Distillation

> A new OPD variant tailored to the **open-academic / structured-lineage** setting (GENE-bench and similar). Companion to `plan.md` — **replaces** Stage 3 there (vanilla OPD is retained as baseline B0 only).
>
> **Locked decisions (user 2026-05-17):**
> - Name: **evo-OPD**
> - Position in pipeline: **replaces vanilla OPD as Stage 3**
> - Primary teacher: **Qwen3-14B-Thinking** served locally via vLLM (open weights → full per-token logits; canonical reverse-KL recipe). GPT-5.5 Black-Box demoted to ablation L7.
> - Anonymous-Robust pairing (L4): **ablation only, not in primary run**
> - Component scope (A field-gating / B verifier-anchoring / C lineage-consistency): **pending user decision** — recommendation: keep all 3 (even with cleaner Qwen3-14B-Thinking teacher, B still defends against the ~20% of cases where Qwen3-14B-Thinking is wrong on GENE-Exam)

---

## 1. Why vanilla OPD leaves value on the table here

Vanilla OPD (Tinker / Qwen3 strong-to-weak) is **schema-blind, verifier-blind, and lineage-blind**. It treats every token as an equally-weighted reverse-KL target against a single teacher's distribution. That works well for math/code, where the answer is a free-form trace and the teacher is approximately correct. The open-academic / GENE-bench setting violates three of those assumptions:

| Property of vanilla OPD | What GENE-bench actually has | Cost of ignoring it |
|---|---|---|
| Tokens are equally weighted | Output is **schema-structured**: 6 gene fields + dynamics label + evidence spans + boilerplate. Different regions carry wildly different semantic load | Gradient wasted on `{`, `,`, `"mechanism":`; load-bearing dynamics labels under-trained |
| Single teacher = ground truth | A **cheap deterministic verifier** exists (`agent/genome_differ.py` decision tree + evidence-span regex + T1–T4 gold exact-match). When teacher and verifier disagree, **teacher is the one likely wrong** (especially GPT-5.5, which scored only 23.1% on the benchmark itself) | Distill from a teacher that has *memorized* benchmark-adjacent content but composes it incorrectly |
| Each prompt is independent | Paper genomes form a **lineage graph** with parent-child constraints. A "good" child card must be GenomeDiff-consistent with its parent | Miss the self-supervised consistency signal that scales without extra teacher cost; under-train T4 verification — *the* paper-identified bridge from understanding to generation |

evo-OPD is the minimal modification that fixes all three.

---

## 2. Design-space exploration

Before locking the algorithm, here are the distinct axes we considered and the ones evo-OPD adopts.

| Axis | Option | evo-OPD's choice | Why |
|---|---|---|---|
| **Token weighting** | Uniform / field-gated / PES-routed | **Field-gated** with 4-bucket weights | Simple, exploits schema, doesn't require classifying tokens into PES dimensions |
| **Teacher trust** | Always trust / verifier-anchored / multi-teacher agreement | **Verifier-anchored** | Directly addresses GPT-5.5 contamination; verifier disagreement → teacher is muted on that token |
| **Self-supervised signal** | None / lineage consistency / anonymous robustness / transition distill | **Lineage consistency** (primary), anonymous robustness (optional layer) | T4 is the paper-identified bottleneck; gets free supervision from `genome_differ.py` |
| **Teacher source** | Single / ensemble / strong-to-weak | **Single teacher Qwen3-14B-Thinking** (open weights, full logits, ~4× student) | Avoids Black-Box truncation hit (~30% sample-efficiency loss), Azure rate-limit risk, and the worst of GPT-5.5's leakage risk; canonical Tinker reverse-KL recipe works as-is |
| **Distillation atom** | Token / transition / sequence | **Token** (field-gated) | Standard, well-tooled in verl; transition-level too coarse for `agent/genome_differ.py` reward density |
| **Rollout pairing** | Single rollout / paired (anonymized) | Single primary; paired as optional regularizer | Paired anonymized rollouts double rollout cost; ablation only |

Five distinct directions we **rejected** for the primary algorithm (kept as ablations):

- **PES-Routed OPD** — three loss heads for Heredity/Variation/Selection. Theoretically clean but requires per-token routing into PES dimensions, which has no clean labeling function.
- **MT-OPD** (multi-teacher agreement-weighted) — useful, but adds 2× teacher compute and only marginally reduces leakage relative to verifier-anchoring.
- **TR-OPD** (transition-level) — distill per-(parent,child) pair instead of per-token. Loses gradient density.
- **AR-OPD** (anonymous-robust pairing) — strong invariance prior but doubles rollout cost; promoted to optional layer §3.3.
- **Pure verifier-only RLVR** — no teacher at all. Drops sample efficiency vs OPD by ~3-5× per Thinking Machines numbers.

---

## 3. Algorithm: evo-OPD

### 3.1 Setup

For each batch of prompts `x`, optionally with parent gene-card context `p(x)`:

1. Student `π_θ` rolls out `y ~ π_θ(·|x)`.
2. Teacher `π_T` = **Qwen3-14B-Thinking**, served locally via vLLM on **1× A100 80GB** (TP=1; bf16 ≈ 28GB weights + KV) on the shared 4× A100 cluster. Open weights → full per-token log-probs `log π_T(y_t | y_<t, x)` directly. Same tokenizer family as the Qwen3-8B-Base student → no cross-tokenizer alignment.

> Teacher/student ratio is 1.75× (14B / 8B), below the 3–5× sweet spot reported by Tinker/Qwen3. This is a deliberate trade for the 4× A100 budget. Mitigations: (a) component B (verifier-anchoring) helps when teacher is wrong; (b) ablation L7 (GPT-5.5 Black-Box) gives an upper-bound check on what a stronger teacher would buy.
3. Parser identifies each token's **schema role** `φ(t) ∈ Φ`:
   - `boilerplate` — JSON brackets, field-name strings, whitespace
   - `content_field` — text inside one of the 6 gene-field values (mechanism / niche / observation / limitation / delta / claim)
   - `dynamics_label` — the one-of-five categorical dynamics label
   - `evidence_span` — quoted span that must appear verbatim in the source paper
   - `gold_answer` — exam-style discrete answer field (T1–T4 multiple-choice / multi-label)
4. Three reward signals are computed:

#### (a) Field-weighted teacher KL (dense, content-aware)

```
α(φ(t)) =  0.3  if boilerplate
           1.0  if content_field
           1.5  if evidence_span
           2.0  if dynamics_label  ← load-bearing for all-or-nothing scoring
           0.0  if gold_answer     ← decoupled, handled by verifier (3.b)

kl_t      = log π_θ(y_t | y_<t, x) − log π_T(y_t | y_<t, x)   ← full-vocab logits from Qwen3-14B-Thinking
L_KL_t    = −α(φ(t)) · kl_t          ← canonical reverse-KL, mode-seeking
```

Setting `α = 0` on `gold_answer` tokens is the **verifier-anchored** part: if the teacher itself is contaminated and produces a wrong gold answer, we don't distill that bias. The next term handles it.

#### (b) Structured verifier reward (sparse, exact, teacher-free)

```
v(y) = 0.20 · 1[schema parses]
     + 0.30 · evidence_citation_frac(y)         ← from regex against source paper
     + 0.30 · dynamics_consistency(y)            ← agent/genome_differ.py decision-tree gate
     + 0.20 · exact_match_when_applicable(y, gold)

L_VR = β · (v(y) − v̄)                           ← scalar advantage, broadcast over tokens
```

- `evidence_citation_frac` = fraction of quoted spans that appear verbatim in source paper. Already implementable from the existing repo.
- `dynamics_consistency` = run `agent/genome_differ.py` on the student's gene-card; check whether the chosen dynamics label is the one the decision tree would assign. **This makes verifier and decision-tree teach consistent dynamics labels, even when the LLM teacher disagrees.**
- `exact_match_when_applicable` is the standard T1–T4 all-or-nothing score. For non-exam prompts, this term contributes 0.

`v̄` is an EMA running mean (variance reduction).

#### (c) Lineage consistency self-signal (medium-density, teacher-free)

When prompt `x` includes a parent context `p`:

```
c(y, p) = 1[GenomeDiff(p, parse(y)) is well-formed and respects predecessor->successor invariants]
        · gene_field_fate_consistency(p, parse(y))

L_LC = γ · (c(y, p) − c̄)
```

`gene_field_fate_consistency` = fraction of gene-field fates (INHERITED / MUTATED / LOST / NOVEL / HYBRIDIZED) that the student's child-card claims AND that `genome_differ.py` would assign given `(p, child)`. **This trains T4 lineage verification directly from GENE-bench's own logic, with zero teacher cost.**

When prompt has no parent context, `γ = 0` for that example.

#### (d) Combined per-token objective

```
r_t   = L_KL_t  +  α(φ(t)) · [ λ_v · (v(y) − v̄) + λ_c · 1[parent] · (c(y, p) − c̄) ]
```

Note: the sparse rewards `v` and `c` are **broadcast onto tokens with the field weight `α(φ(t))`**, so they primarily reinforce content-bearing tokens (not boilerplate). This is the second use of the schema.

Default coefficients: `λ_v = 0.5`, `λ_c = 0.3`. Tune in pilot run.

#### (e) Update

Standard policy gradient on `r_t`. No critic. Reference-policy KL `β_ref = 0` (the field-gated reverse-KL acts as its own anchor); add a small `β_ref = 1e-4` if entropy collapses.

### 3.2 Pseudocode

```python
for batch in dataloader:
    x_batch, p_batch = batch.prompts, batch.parents  # p may be None
    y_batch = student.generate(x_batch)              # on-policy rollout
    
    teacher_logp = teacher.score(y_batch, x_batch)   # per-token log p_T
    student_logp = student.score(y_batch, x_batch)   # per-token log p_theta
    
    phi = parse_schema_roles(y_batch)                 # token -> role
    alpha = field_weight(phi)                         # token -> alpha
    
    # (a) field-weighted reverse-KL
    L_KL = -alpha * (student_logp - teacher_logp)
    
    # (b) verifier reward
    v = structured_verifier(y_batch, sources)         # scalar per example
    v_adv = (v - v_baseline.ema())
    
    # (c) lineage consistency
    c = lineage_consistency(y_batch, p_batch)         # scalar per example
    c_adv = (c - c_baseline.ema())
    
    # broadcast to tokens, weighted by alpha
    L_VR = alpha * lambda_v * v_adv[:, None]
    L_LC = alpha * lambda_c * c_adv[:, None] * (p_batch is not None)
    
    r = L_KL + L_VR + L_LC
    loss = -(r * advantages).mean()
    loss.backward()
    optimizer.step()
```

### 3.3 Optional add-on layers (ablation candidates)

- **AR layer (Anonymous Robustness):** for each prompt, sample a paired rollout `y'` on the anonymized variant `x'`; add `δ · KL(π_θ(·|x) ‖ π_θ(·|x'))` averaged over content-field tokens. Forces the model to be invariant to surface identity (titles/years/author names). Doubles rollout cost.
- **MT layer (Multi-Teacher Agreement):** weight `L_KL_t` additionally by ensemble agreement `1 − H_norm(top-k votes across teachers)`. Triples teacher forward-pass cost.
- **Transition packing:** every example is a (parent, child) pair concatenated with a separator; the `c` term gets denser supervision. Easier than per-token routing.

---

## 4. Implementation over verl

verl supports per-token rewards via its generic `reward_fn` interface, so evo-OPD lands as a custom reward module + a custom rollout postprocessor.

### 4.1 Module layout (under `idea_train/lv_opd/`)

```
evo_opd/
├── config.py             # weights, hyperparameters
├── parser.py             # JSON gene-card parser + token-role tagger φ
├── verifier.py           # v(y): wraps agent/genome_differ.py + evidence regex
├── lineage.py            # c(y, p): GenomeDiff well-formedness check
├── rewards.py            # combines KL + v + c into per-token reward
├── teachers/
│   ├── qwen32b_thinking.py    # PRIMARY — vLLM-served locally, full per-token logits
│   ├── gpt55_blackbox.py      # ablation L7 — Azure keyless, top_logprobs=20, truncated-tail correction
│   └── ensemble.py            # MT add-on (L5)
├── trainer/
│   └── verl_recipe.yaml       # verl GRPO recipe with custom reward
└── eval_hooks.py          # per-checkpoint GENE-Exam smoke + ablation logging
```

### 4.2 verl integration points

- Replace verl's default `reward_model` with `lv_opd.rewards.reward_fn`.
- Hook `lv_opd.parser.tokenize_with_roles` into the rollout postprocessor so each rollout carries a `φ` tag tensor.
- Teacher service runs in a separate vLLM process; `lv_opd.teachers.qwen32b_thinking.score(y)` returns full per-token log-probs over the student's vocabulary (same tokenizer family — no cross-tokenizer projection needed).

### 4.3 Cost vs vanilla OPD

| Component | Vanilla OPD | evo-OPD | Overhead |
|---|---|---|---|
| Student rollout | 1× | 1× | 0% |
| Teacher forward | 1× | 1× | 0% |
| Schema parsing | 0 | 1 CPU pass | ~negligible |
| Verifier (regex + decision tree) | 0 | 1 CPU pass | ~0.5% of step |
| Lineage check | 0 | 1 call to genome_differ | ~1% of step |
| **Total wall-clock overhead** | — | — | **~2%** |

Effectively free additional signal density.

---

## 5. Baselines and ablations

The headline experiment is evo-OPD vs vanilla OPD with identical SFT initialization, teacher, prompts, and step budget.

| ID | Variant | What it tests |
|---|---|---|
| **B0** | Vanilla OPD (Tinker recipe, Qwen3-14B-Thinking teacher) | Headline baseline — same teacher as L0, so the comparison isolates the evo-OPD components, not the teacher |
| **B1** | RLVR-only (DAPO + verifier, no teacher) | Tests whether the OPD signal helps at all on top of pure RL |
| **B2** | SFT → evo-OPD (no GRPO middle stage) | Tests Thinking Machines "OPD replaces RL" claim on a non-math task |
| **L0** | **evo-OPD full** (A field-gated + B verifier-anchored + C lineage-consistent, Qwen3-14B-Thinking teacher) | **Main proposed method** |
| **L1** | evo-OPD − A (uniform α=1) | Isolates schema awareness |
| **L2** | evo-OPD − B (α≠0 on gold, no verifier-decoupling) | Isolates teacher-contamination defense; with Qwen3-14B-Thinking this should show *smaller* gap than it would with GPT-5.5, but still positive — Qwen3-14B-Thinking is also only ~17–20% on GENE-Exam |
| **L3** | evo-OPD − C (γ=0) | Isolates the T4 self-supervision |
| **L4** | evo-OPD + AR anonymous layer (paired rollout) | **Ablation only** per user decision; tests anonymous-mode invariance |
| **L5** | evo-OPD + MT multi-teacher agreement (Qwen3-14B-Thinking + GPT-5.5-top20 + Claude top-20) | Tests whether ensemble adds anything beyond single open teacher |
| **L7** | evo-OPD with GPT-5.5 Black-Box teacher (top-20 logprobs, truncated-tail) | Frontier teacher under closed-weight constraint; if L7 > L0 by a clear margin → revisit the teacher choice |

Headline metrics per variant:
- GENE-Exam main-challenge exact (per tier: T1, T2, T3, **T4** especially), Arena PES per setting, denylist Min-K% Prob gap, MATH-500 / GPQA / HumanEval no-regression.

Predicted ordering (claims we'd defend):
- L0 > B0 on **T4 first** (lineage consistency directly targets it), Arena Heredity (verifier-anchored evidence spans), and Min-K% gap (verifier-anchored decoupling reduces teacher leakage).
- L0 ≈ B0 on math/code (no LV-specific signal there).
- L3 ≈ B0 on T4 (predicts L3 reverts to B0's T4 weakness, confirming γ-term value).

---

## 6. Risks and open questions

| Risk | Detection signal | Mitigation |
|---|---|---|
| **Parser fragility** — student's JSON breaks → φ undefined for all tokens | parsing-failure rate per checkpoint | (a) penalize via `schema_valid` term in v(y); (b) fallback regex parser; (c) early in training set α uniform for malformed outputs |
| **Verifier false negatives** (`genome_differ.py` is itself approximate) | judge-vs-verifier disagreement on Arena samples | Periodic sample audit; treat v(y) as soft (not hard 0/1); add affine recalibration vs human spot-checks |
| **Lineage signal saturates** (model trivially satisfies GenomeDiff by copying parent) | c(y,p) → 1 with student outputs becoming uninformative | Add anti-copy penalty: `c` capped if normalized edit distance to parent < 0.4 |
| **Teacher field-tagging mismatch** | content-field tokens misclassified as boilerplate or vice versa | Use teacher's tokenizer (same as student, since Qwen3 family) + character-level alignment of parsed JSON span boundaries to token offsets |
| **Reward weight sensitivity** | unstable PG variance with `λ_v`/`λ_c` ≠ default | Reward normalization (per-batch z-score); start with conservative `λ_v=0.3, λ_c=0.2` then sweep |
| **Composite reward gaming** | v(y) climbs via schema validity alone, content stays weak | Sub-decompose v(y) and require all 4 components ≥ 0.3 floor for credit |
| **Domain skew** | evo-OPD specialized to GENE-style structure → drops on free-form QA | Mix 20% of OPD steps on free-form science QA (no φ tagging, vanilla OPD on those) — preserves generality |
| **"Just an engineering combination"** critique in paper review | reviewer sees field-gating + verifier-anchoring + lineage-consistency as known components recombined | Argue novelty = first OPD variant for *structured-lineage* domains; show each ablation removes a specific known failure mode of vanilla OPD; release code as a reusable framework |
| **Qwen3-14B-Thinking serving node fails / VRAM contention with student** | Teacher latency p99 > 2× p50; OOM | Dedicated 1× A100 80GB for teacher (TP=1); preflight vLLM serve at expected concurrency; if shared-node VRAM tight, fall back to Qwen3-8B-Thinking (teacher = student size, mostly useful as a baseline only) |
| **Qwen3-14B-Thinking is itself wrong on benchmark-style prompts** (only ~17–20% on GENE-Exam) | Student L0 plateaus at teacher's exam score | Component B (verifier-anchoring) defends here too: on the ~80% of cases where teacher is wrong on a verifiable gold answer, decouple α=0 and let verifier reward dominate |
| **Tokenizer drift across Qwen3 versions** | Logprob index mismatch between teacher and student tokenizers | Pin both teacher and student to identical Qwen3 tokenizer revision; assert vocab equality at startup |

Open questions to resolve before kickoff:

1. **Parser implementation:** custom JSON parser with span-to-token alignment is the trickiest engineering piece. Estimate: 3–5 days. We could simplify by using a fixed prompt template that produces newline-delimited fields instead of nested JSON.
2. **Lineage-consistency formal spec:** the GenomeDiff decision tree in `agent/genome_differ.py` is partially deterministic, partially LLM-assisted. Use only the deterministic part for `c(y, p)` to keep training signal teacher-independent.
3. **Should evo-OPD run after GRPO (replacing vanilla OPD as Stage 3) or replace GRPO entirely?** Default: replace Stage 3. Justified by ablation B2 (SFT → OPD direct), which Thinking Machines shows competitive on math.
4. **Cross-tokenizer fallback** if we ever swap to Llama/Mistral teacher: use string-level alignment + greedy match (cf. HF `huggingfaceh4-on-policy-distillation` Space).

---

## 7. Integration with main `plan.md`

evo-OPD replaces **Stage 3** in `plan.md` § 5. Other stages unchanged:

```
Stage 0  Contamination firewall
Stage 1  SFT cold-start (80K examples, GPT-5.5 teacher)
Stage 2  GRPO/DAPO (DAPO + Dr.GRPO, GPT-5.5-judged Arena reward)
Stage 3  ──> evo-OPD (this doc) ──   replaces vanilla OPD
Stage 4  Eval + 8 ablations
```

The 8 ablations in plan.md § 5.5 stay, plus evo-OPD adds:
- L1 (− field gating)
- L2 (− verifier anchoring)
- L3 (− lineage consistency)

These directly become Table-2 ablation columns in the paper.

---

## 8. Next concrete steps

If we commit to evo-OPD, the implementation order (parallelizable):

1. **Day 1–3:** parser + verifier prototypes on `gene_exam/Questions/` outputs (no training yet, just to confirm parsing accuracy ≥ 95%).
2. **Day 4–5:** lineage-consistency module on `data/genome_db/gene_diffs.json` (verify it agrees with stored labels ≥ 90% — sanity check on `agent/genome_differ.py` extraction).
3. **Day 6–8:** verl integration + custom reward function; smoke-run on a 100-example subset with student = SFT checkpoint, teacher = Qwen3-14B-Thinking.
4. **Day 9–14:** full evo-OPD pilot run (50 steps) on `idea-8b-sft` → `idea-8b-lvopd-pilot`; eval on GENE-Exam 200-instance smoke + Arena 5-task smoke.
5. **Week 3:** full evo-OPD run (200–300 steps) + 3 internal ablations (L1, L2, L3).
6. **Week 4:** add-ons (L4 AR, L5 MT) if compute permits.

Total evo-OPD-specific engineering: ~2 weeks of focused work, mostly on parser + lineage module + verl reward hook.

---

## 9. Decisions log

| # | Decision | Value | Date |
|---|---|---|---|
| 1 | Name | **evo-OPD** | 2026-05-17 |
| 2 | Component scope (A field-gating / B verifier-anchoring / C lineage-consistency) | **all 3** (A + B + C). Rationale: each removes a distinct vanilla-OPD failure mode; engineering overhead ~2%; B remains valuable even with Qwen3-14B-Thinking teacher because Qwen3-14B-Thinking is also ~17–20% on GENE-Exam (i.e., wrong on ~80% of cases the verifier can adjudicate). | 2026-05-17 |
| 3 | Stage 3 position | **replace vanilla OPD**; vanilla OPD survives only as baseline B0 | 2026-05-17 |
| 4 | Primary teacher | **Qwen3-14B-Thinking** served locally via vLLM (open full logits; canonical reverse-KL). GPT-5.5 Black-Box demoted to ablation L7 | 2026-05-17 |
| 5 | Anonymous-Robust pairing (L4) | **ablation only** | 2026-05-17 |

**Day-0 kill switches for the Qwen3-14B-Thinking teacher choice:**
1. Confirm Qwen3-14B-Thinking weights downloadable + license compatible (Apache-2.0 expected). If access issue → fall back to Qwen3-8B-Thinking (teacher = student size — useful only as a baseline check).
2. Pre-flight vLLM serve at TP=1 on 1× A100 80GB: measure p50 / p99 latency at 32-64 concurrency. If p99 > 5s, drop concurrency.
3. Sanity-check teacher quality on 50 GENE-Exam main-challenge instances: Qwen3-14B-Thinking should score within ±3 abs of its published Qwen3 leaderboard number. If far off → wrong checkpoint / config.

**Reuse of GPT-5.5 elsewhere in `plan.md` is unchanged:**
- SFT synthetic gene cards: still **GPT-5.5** (strongest available; quality verification + denylist filter contain the contamination risk at SFT stage).
- GRPO Arena reward judge: still **GPT-5.5** frozen snapshot.
- Only the *OPD teacher* changed from GPT-5.5 Black-Box → Qwen3-14B-Thinking open.
