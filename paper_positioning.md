# Paper Positioning: GeneTrace + evo-OPD

> Strategic framing for the two contributions we will claim. Drives the data
> schema (`genetrace_data_format.md`), the training-method spec
> (`lv_opd_plan.md`, `evo_opd_open_ended.md`), and the paper draft
> (`paper/latex/main.tex`). Companion to `plan.md`.
>
> **Framing update (2026-05-18, user confirmation):** IdeaEvolving is our
> lab's prior work — it *defined* the genome paradigm and built GENE-bench as
> a proof-of-concept. This paper is the **formalisation + scaling-up**: we
> codify the paradigm into a release-grade schema (GeneTrace), prove it
> trainable with a tailored algorithm (evo-OPD), and ship code + data + model
> together. The "differentiation vs IdeaEvolving" framing is dropped — we
> cite it as foundational. The remaining differentiation is vs Intern-Atlas
> (concurrent, document-centric) and the OPD/GRPO methods literature.

## 1. The two contributions

We will write the paper as a **method + dataset double contribution**, with the
method as primary and the dataset as enabling infrastructure (mirroring how
InstructGPT / Tulu / OLMo present their releases).

### C1 — evo-OPD (primary, method contribution)

A lineage-aware on-policy distillation algorithm with three coupled signals
(reverse-KL to teacher, verifier reward, lineage-consistency self-signal), each
gated by per-token role (`φ(t)`). Generalises to **open-ended generative
tasks** (proposal writing, idea synthesis) by replacing the 0/1 verifier with a
continuous verifier-as-reward — same algorithm, no architecture change.

Why this is novel beyond vanilla OPD (Tinker, Qwen3 strong-to-weak):
- Vanilla OPD is schema-blind, verifier-blind, and lineage-blind (`lv_opd_plan.md` §1).
- Existing RL-from-verifier (DAPO, Dr. GRPO) assume one scalar reward per
  trajectory and ignore token-level structure.
- evo-OPD couples both signals at the per-token level via `α(φ(t))` weighting,
  and adds a self-supervised lineage-consistency signal that **does not require
  a gold label** — applicable to open-ended tasks where exact-match doesn't fire.

### C2 — GeneTrace (enabling, data contribution)

The **first release-grade, training-scale instantiation** of the genome paradigm
introduced in our prior work (IdeaEvolving / GENE-bench). GeneTrace upgrades
the paradigm from a proof-of-concept eval set into a corpus designed for
*training* generative models:

1. **6-field genome cards** per paper (mechanism / niche / observation / limitation / delta / claim), each grounded in evidence quotes from the source text — IdeaEvolving introduced the schema; we make it evidence-grounded and release-quality;
2. **Categorical evolutionary-dynamics labels** on each lineage edge (5 modes: Mutation, Adaptive Radiation, Hybridization, Speciation, Niche Competition) — same taxonomy as IdeaEvolving, now applied at corpus scale rather than only inside the 1,029-item exam;
3. **Automated verifier scores** per annotation (schema validity, evidence groundedness, dynamics consistency) — *new in this paper*; turns the annotations into a directly RL-usable reward;
4. **Reasoning traces** from a strong teacher (GPT-5.5) showing how each annotation was derived from the source text — *new in this paper*; enables CoT-style SFT and serves as the OPD warm-up.

We position GeneTrace as the **training-grade primitive layer** under arbitrary
downstream tasks — GENE-bench's 42 exam tasks (the original IdeaEvolving
evaluation), idea generation, paper retrieval, literature review — and release
it under a permissive license with explicit contamination guards.

---

## 2. Differentiation landscape

| Work | Scale | Per-paper representation | Edge semantics | Has verifier? | Has RL/OPD method? | Open-released? |
|---|---|---|---|---|---|---|
| **IdeaEvolving / GENE-bench** (ours, prior) | ~2K papers, 1,029-item exam | 6-field genome card (introduced here) | 5-mode dynamics (introduced here) | Partial (exam-only) | No (eval only) | Code yes, training data partial |
| **Intern-Atlas** (Apr 2026, arXiv 2604.28158, concurrent) | 1M papers | Method-entity name | Generic lineage + bottleneck | No | No | Reportedly yes |
| **AutoSurvey / ResearchAgent** | varied | Free-text summaries | Citation only | No | No | Yes |
| **Tinker OPD recipe** | n/a | n/a | n/a | n/a | OPD (single-signal) | Recipe yes |
| **DAPO / Dr. GRPO** | n/a | n/a | n/a | Yes (math/code) | RL (scalar reward) | Code yes |
| **GeneTrace + evo-OPD (this paper)** | 5K–50K (tiered) | **6-field card + evidence + reasoning trace** | **5-mode dynamics + per-step verifier** | **Yes, per-annotation** | **Yes (evo-OPD)** | **Yes, MIT** |

**Position relative to IdeaEvolving (our prior work).** IdeaEvolving introduced
the 6-field genome card and the 5-mode dynamics taxonomy and validated them via
a hand-curated 1,029-item exam. The paradigm was *defined and shown plausible*,
but the corpus was eval-only — not designed for training. This paper takes the
paradigm to the next stage:
- We add per-field **evidence quotes** that ground each genome claim in source text.
- We attach an **automated verifier** to every annotation, turning the paradigm into a directly RL-usable reward.
- We add **teacher reasoning traces** that enable CoT SFT and OPD warm-up.
- We scale from ~2K hand-curated papers to **5K–50K** annotated by GPT-5.5 + Qwen3-14B with consistency filtering.
- We design **evo-OPD**, the first training algorithm that exploits all three genome-paradigm primitives (card / dynamics / lineage).

**Position relative to concurrent Intern-Atlas.** Intern-Atlas is also a
methodological evolution graph, but is *document-centric*: it identifies method
entities (names) and connects them via generic lineage edges + bottleneck
labels. The two efforts are complementary — Intern-Atlas covers breadth (1M
papers, method-entity granularity), GeneTrace covers depth (5K–50K papers, full
genome-card decomposition + dynamics taxonomy + verifier + reasoning traces).
GeneTrace cards could in principle be attached to Intern-Atlas method entities
as a richer per-paper annotation layer.

**Defensible wedges in this paper:**
1. **First training-grade release of the genome paradigm** (vs IdeaEvolving's eval-only release; vs Intern-Atlas's coarser method-entity representation).
2. **evo-OPD is the first OPD variant that uses lineage structure and verifier rewards together**, with a principled extension to open-ended generation.
3. **Contamination guard at the dataset level:** we ship the denylist with the data, document the safe-pool construction, and report Min-K% leakage diagnostics in the dataset card. No prior paper-lineage corpus has done this.

---

## 3. Why "genome-centric" is the right primitive (and how we say it in the paper)

Three layers of representation are possible for a paper:

| Layer | Example resource | What you can do with it | What you can't |
|---|---|---|---|
| **Document-centric** | Semantic Scholar abstracts, ArXiv full-text | Read, embed, retrieve | Reason about *what changed* between two papers |
| **Method-entity-centric** | Intern-Atlas, Paper-with-Code | Trace technique evolution | Distinguish *why* a method evolved (limitation? hybridization? speciation?) |
| **Genome-centric (ours)** | GeneTrace | Diff papers across 6 axes, label transition modes, verify groundedness | n/a (this is the most structured layer) |

The thesis: **method-entity is not granular enough for a learned model to reason about
*why* one paper followed another**. A method-entity edge says "B uses A's technique",
but doesn't say whether B fixed A's limitation, hybridised A with another lineage,
or speciated into a new niche. The 5-mode dynamics taxonomy captures that, and the
6-field card gives the model the evidence to make the call. This is what a model
needs to *propose* the next paper, not just to *retrieve* the previous one.

We expect reviewers to push back: "isn't 6 fields arbitrary?". Defence:
- The 6 fields are a Pareto cut of what IdeaEvolving's annotators actually used (verified by us via the GENE-Exam Questions/ files).
- Ablating fields (drop `delta_genome`, drop `claim_genome`) is one of the dataset-side ablations.
- We do NOT claim the 6-field decomposition is universal — we claim it's *sufficient* for the lineage-reasoning tasks we evaluate, and *strictly richer* than method-entity.

---

## 4. Open-source plan (concrete)

Three tiered releases over the course of the project:

### GeneTrace v0.1 — "starter set" (target: paper submission)
- **Cards:** ~5K (round-1 + round-3 SFT pool, all pre-2017, all evidence-quote backed, all verifier-passed).
- **Lineage pairs:** ~2K with dynamics labels.
- **Lineage chains:** ~200 (length 3–5).
- **Reasoning traces:** GPT-5.5 chain-of-thought stored per annotation.
- **Verifier code:** open with the data (so consumers can re-score on their own splits).
- **Contamination assets:** denylist_v0 (8,359 papers), safe-pool construction code, Min-K% leakage report.
- **License:** MIT for code, CC-BY-4.0 for data (compatible with downstream RL training).

### GeneTrace v0.2 — "scale-up" (target: 1 month post-acceptance)
- Expand cards to ~20K via consistency-filtered teacher voting (GPT-5.5 + Qwen3-14B-Thinking; only keep examples where both agree on dynamics label).
- Expand denylist via OpenAlex 1-hop (already in progress).
- Add Arena-style open-ended prompts with PES rubric annotations.

### GeneTrace v1.0 — "production-grade" (long term)
- 50K+ cards, multi-domain (CS + bio + physics), multi-teacher consensus, ISO-639 license + DUA.
- Hosted on HuggingFace with revision pinning.

---

## 5. Paper outline (proposed, to refine)

1. Introduction — the document/method/genome layering argument
2. Related work — Intern-Atlas, IdeaEvolving, Tinker OPD, DAPO
3. **GeneTrace** (C2):
   - 3.1 Genome-card schema (6 fields + evidence)
   - 3.2 Dynamics taxonomy (5 modes + verifier)
   - 3.3 Lineage chain construction
   - 3.4 Contamination guard
   - 3.5 Dataset statistics + comparison table
4. **evo-OPD** (C1):
   - 4.1 Algorithm (three signals, per-token gating)
   - 4.2 Open-ended generalisation (continuous verifier)
   - 4.3 Implementation (data shape, rollout loop, teacher serving)
5. Experiments:
   - 5.1 GENE-Exam main result (Qwen3-8B + SFT + evo-OPD vs baselines)
   - 5.2 GENE-Arena PES result (open-ended)
   - 5.3 Pipeline ablations (Variants 1–8 from `plan.md` §5.5)
   - 5.4 evo-OPD component ablations (L1–L7 from `lv_opd_plan.md` §5)
   - 5.5 Cross-domain transfer (bio papers from GeneTrace v0.2)
6. Discussion — limitations, ethical considerations, what 1M-scale unlocks
7. Conclusion

---

## 6. What this positioning REJECTS, and why

- **"Differentiate from IdeaEvolving" framing** — rejected. IdeaEvolving is our prior work. We cite it as foundational and frame this paper as the formalisation + scaling-up, not a competing approach.
- **"Scale-the-corpus" framing** — rejected. We will lose to Intern-Atlas on raw paper count and shouldn't compete there. Our claim is **structured-annotation quality + trainability**, not breadth.
- **"GENE-bench is the benchmark" framing** — rejected. GENE-bench is one downstream view (and it's our prior work's eval); framing the paper around it makes us look benchmark-overfit. The corpus + method are the contribution; GENE-bench main-challenge + GENE-Arena are two of several evaluations.
- **"evo-OPD is just GRPO + OPD" framing** — rejected. The novelty is the per-token role gating + lineage-consistency self-signal, both of which require the genome-card structure. We make this explicit by showing the algorithm degenerates to vanilla OPD when `α≡1` and `λ_c=0` (ablation L1).
- **"Beat GPT-5.5 on GENE-bench" framing** — rejected. We are at ~49% of GPT-5.5 with 30× fewer params; that's the *headline number*, not the headline *claim*. Headline claim is "evo-OPD gives consistent wins over SFT / GRPO / vanilla-OPD at the same compute, on both closed and open-ended lineage tasks".
