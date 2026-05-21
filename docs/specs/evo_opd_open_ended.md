# evo-OPD: Open-Ended Generalisation

> Companion to `lv_opd_plan.md` (the algorithm spec). This document covers (a)
> how evo-OPD extends from closed/verifiable tasks to open-ended generation
> (proposal writing, idea synthesis, Arena-style outputs), and (b) the
> concrete data shape during training. Aligned with `paper_positioning.md` C1.

## 1. The closed-vs-open task spectrum

| Task class | Example | Has gold? | Verifier signal | OPD compatibility |
|---|---|---|---|---|
| **Closed-form, exact-match** | GENE-Exam T3-01 ("driver = mechanism") | yes | 0/1 exact match | Trivial — verifier ≈ reward |
| **Closed-form, multi-field** | GENE-Exam T2-04 (group 8 papers into 2 lineages) | yes | structured exact match | Trivial |
| **Closed-form, schema-only** | "extract a 6-field gene card from this paper" | partial (evidence quotes verifiable, free-text not) | schema + evidence groundedness | Vanilla OPD works; evo-OPD adds verifier head |
| **Open-ended, lineage-conditioned** | "given papers A and B, propose paper C that extends them" | no | proxy: novelty + plausibility + lineage consistency | Vanilla OPD has nothing to anchor on; evo-OPD's lineage-consistency self-signal carries the load |
| **Open-ended, free** | "propose a follow-up to this paper" | no | rubric (PES-style) | evo-OPD with continuous verifier; reduces to RLHF-with-judge if `λ_c = 0` |

Vanilla OPD covers row 1-3 only (closed). evo-OPD's contribution is rows 4-5
without losing performance on 1-3 — that's the whole pitch.

---

## 2. Algorithmic recap

Per-token reward, applied during on-policy distillation:

```
r_t  =  − α(φ(t)) · KL[π_θ(·|y_<t,x) ∥ π_T(·|y_<t,x)]      ←  field-gated reverse-KL (A)
        + α(φ(t)) · λ_v · v_adv(y, t)                       ←  verifier-anchored decoupling (B)
        + α(φ(t)) · λ_c · c_adv(y, p, t)                    ←  lineage-consistency self-signal (C)
```

Where:
- `φ(t)` is the token-role tag from `evo_opd/parser.py`
  (boilerplate / content_field / evidence_span / dynamics_label / gold_answer / unknown).
- `α(φ(t))` is the per-role weight from `evo_opd/schemas.py::FIELD_WEIGHT`.
- `v(y) ∈ [0, 1]` is the verifier score from `evo_opd/verifier.py`.
- `c(y, p) ∈ [0, 1]` is the lineage-consistency score from `evo_opd/lineage.py`.
- `v_adv`, `c_adv` are token-distributed advantages: each token in a content
  span gets the (v − baseline_v) signal; tokens outside content spans get 0.

`λ_v` and `λ_c` are loss weights (defaults: `λ_v = 1.0`, `λ_c = 0.5`; tuned in
the L2/L3 ablations).

---

## 3. The open-ended generalisation

The key claim: **the same three signals apply to open-ended tasks, with no
algorithmic change. The verifier just becomes continuous.**

### (A) Field-gated reverse-KL — UNCHANGED.
Teacher distribution always exists; per-token reverse-KL works regardless of
whether the task has a gold answer. For open-ended tasks the teacher (Qwen3-14B-Thinking
or GPT-5.5 Black-Box) provides stylistic / fluency anchoring.

### (B) Verifier-anchored decoupling — REPLACE `v(y)` WITH A CONTINUOUS REWARD.

For closed tasks, `v(y)` is composed of:
- `schema_valid` (1 if JSON matches schema, else 0)
- `evidence_grounded` (fraction of cited quotes that match source text)
- `dynamics_valid` (1 if dynamics label ∈ enum, else 0)
- `exact_match` (1 if gold matches, else 0)

For open-ended tasks, drop `exact_match` and replace it with **judge-based
continuous reward**, structured by the GeneTrace primitives:

| Component | Closed form | Open-ended form |
|---|---|---|
| `schema_valid` | JSON parses | Proposal follows the required template (title / motivation / proposed_mechanism / expected_observation / acknowledged_limitation) — checked by a regex + light parser, 0/1 |
| `evidence_grounded` | Quotes match source | Proposal's "motivation" and "limitation" sections cite *quotes from the predecessor paper*, verified by string match against the GeneTrace card's `evidence` field — fractional |
| `dynamics_valid` | Dynamics ∈ enum | Proposal claims a dynamics (e.g. "this is an Adaptive Radiation") and the claim is *consistent* with which gene-fields the proposal changes — checked by `verifier.py::check_dynamics_consistency` — fractional |
| `exact_match` | Gold compare | **PES-style rubric judge** (Heredity / Variation / Selection, 0–10 each, by GPT-5.5 as judge) → normalise to `[0,1]` |

The four subscores are weighted into `v(y) ∈ [0,1]` with the same weights as
the closed verifier. Per-token decomposition `v_adv(y, t)` is unchanged
(content tokens carry the advantage, boilerplate doesn't).

### (C) Lineage-consistency self-signal — STRONGER ON OPEN-ENDED THAN CLOSED.

`c(y, p)` measures whether the model's output is consistent with the lineage
structure of paper `p`:

- Closed tasks: `c(y, p)` is computed by re-rolling the model on a SECOND
  paper from the same lineage and checking if the predictions are mutually
  consistent (e.g. if `dynamics(p₁→p₂) = Mutation`, the model should also
  predict `dynamics(p₂→p₃) = something compatible`).
- Open-ended tasks: `c(y, p)` is computed by asking the model to *predict the
  predecessor paper's genome card* given the proposed successor, then
  comparing to the ground-truth predecessor card from GeneTrace. High `c(y,p)`
  means the proposal is "lineage-faithful": it could plausibly have evolved
  from `p` under one of the 5 dynamics modes.

This is the **self-supervised signal that vanilla OPD lacks**. It requires no
human or judge — only the GeneTrace structure. It's the main contribution of
evo-OPD on the open-ended axis.

---

## 4. Training-time data shape (the actual JSONL fed to the trainer)

Each row in the rollout buffer (re-generated every train step):

```json
{
  "rollout_id":     "step_0042::prompt_0007::sample_2",
  "prompt_id":      "evo::open_ended::p_0007",
  "prompt":         "<paper P's gene card + 'propose a follow-up paper'>",
  "task_kind":      "open_ended_proposal",       // or "closed_exam"
  "source_papers":  ["paper:foo_2015"],          // from safe_pool
  "lineage_anchor": {                            // for component (C)
    "paper_id":     "paper:foo_2015",
    "predecessor":  "paper:earlier_2013",        // for c(y, p) backward prediction
    "card":         { ... GenomeCard ... }
  },

  "completion_y":   "<student rollout, ~512-2048 tokens>",
  "token_ids":      [101, 245, 1923, ...],
  "per_token_phi":  ["boilerplate","content_field",...],  // from parser.py

  "teacher_logprobs": [                          // for component (A)
    {"token_id": 245, "logprob": -1.23},
    ...
  ],

  "verifier_v": {                                // for component (B)
    "schema_valid":     1.0,
    "evidence_grounded": 0.62,
    "dynamics_valid":   1.0,
    "judge_rubric":     0.71,                    // PES-style if open-ended
    "v_total":          0.83
  },
  "v_token_adv":      [...],                     // per-token (v − baseline_v)

  "lineage_consistency_c": {                     // for component (C)
    "predecessor_card_pred": { ... },            // model's prediction
    "predecessor_card_gold": { ... },            // from GeneTrace
    "field_agreement_per_genome": {"mechanism": 0.8, "niche": 0.5, ...},
    "c_total":          0.65
  },
  "c_token_adv":      [...],

  "final_per_token_reward": [...]                // r_t for the PG update
}
```

This is the **training data of evo-OPD** — what the user asked about. To
reiterate the distinction:

- **SFT data** is offline, static: `(prompt, gold_completion)` pairs.
- **evo-OPD data** is online, regenerated each step: `(prompt, student_rollout, teacher_logprobs, verifier_subscores, lineage_signals)` tuples — the rollout itself varies because the student keeps changing.

The only persistent on-disk artefacts during evo-OPD training are:
- `prompts.jsonl` — the prompt pool (~10K, sampled per step), built from
  GeneTrace + math/code + Arena open prompts.
- `lineage_index.jsonl` — for each prompt, which GeneTrace paper(s) it
  references and what their lineage neighbours are (used for the `c(y, p)`
  computation without an extra teacher call).
- Per-step rollout buffers (transient, ~10K records each, flushed after the
  gradient update).

---

## 5. Why this scheme genuinely generalises (and where it doesn't)

**Generalises to:**
- Any structured-output task with a verifiable schema (math, code, JSON tasks).
- Any task that has an associated graph / lineage (paper proposal, drug discovery, code-refactor history, mathematical-proof tree).
- Any task where a strong teacher's distribution is available (everything where
  vanilla OPD applies).

**Does NOT generalise to:**
- Pure-creative open-ended tasks with no verifiable structure (free-form
  poetry, dialogue without a topic). For these, only component (A) fires;
  (B) and (C) degrade to 0 and the algorithm degenerates to vanilla OPD.
- Tasks where lineage cannot be defined (single-instance Q&A). Same degeneracy.

We will make the degeneracy explicit in the paper: evo-OPD = vanilla OPD when
`(B)` and `(C)` are off, so the worst-case behaviour is "as good as vanilla OPD,
not worse". This is the *non-inferiority* claim, separate from the
*superiority* claim on lineage-rich tasks.

---

## 6. Connection to existing literature

| Existing method | What it does | What evo-OPD adds |
|---|---|---|
| **Tinker OPD** (Thinking Machines, 2025) | Per-token reverse-KL to teacher, no reward | Adds verifier reward + lineage consistency |
| **DAPO / Dr. GRPO** | Per-trajectory scalar reward, no teacher | Adds teacher distribution + per-token role gating |
| **Decoupled-PPO** (TRL) | Reward-model + KL-to-ref-policy decoupling | Same decoupling principle, applied at token-role level with task-specific verifier |
| **RLAIF-V** | LLM-as-judge reward | Same judge schema, but anchored to GeneTrace's verifier-checkable schema (not free-text judge) |
| **PES rubric** (IdeaEvolving, 2025) | Heredity/Variation/Selection eval for proposals | Used as the judge_rubric subscore in our open-ended verifier |

So evo-OPD is positioned as the **per-token combination** of Tinker-style
distillation, DAPO-style RL, and a novel self-supervised lineage signal. The
novelty is the *combination + gating*, not any single component.

---

## 7. Implementation status (today)

| Component | Status | File |
|---|---|---|
| Parser (φ(t) tagging) | ✅ exists | `evo_opd/parser.py` |
| Verifier (closed-form v(y)) | ✅ exists | `evo_opd/verifier.py` |
| Lineage consistency (c(y,p)) | ✅ exists, closed-form only | `evo_opd/lineage.py` |
| Token-level reward composer (r_t) | ✅ exists | `evo_opd/rewards.py` |
| Teacher client (Qwen3-14B-Thinking via vLLM) | ⏭ not yet | TBD `evo_opd/teachers/qwen3_local.py` |
| Open-ended verifier extension | ⏭ not yet | TBD extend `verifier.py` |
| Lineage backward-prediction (`c(y,p)` open-ended) | ⏭ not yet | TBD extend `lineage.py` |
| Rollout loop / trainer | ⏭ not yet (will use verl) | TBD `train/evo_opd_loop.py` |

Earliest training kickoff: when (a) v7 SFT data is in (closes the prompt-format
gap), (b) Qwen3-14B teacher serving is up, (c) verl integration patch is
landed. Estimated 1 week of engineering.

---

## 8. Open questions

- **Teacher staleness during rollouts.** Tinker recipe re-queries the teacher
  every rollout. With Qwen3-14B on one A100 (TP=1), throughput is the bottleneck
  — we may need to amortise teacher queries (e.g. one teacher pass per K rollouts
  with the same prompt). Empirical question; budget 1 week of tuning.
- **`λ_v` vs `λ_c` schedule.** Should `λ_c` ramp up over training (early steps
  rely on `λ_v` more, late steps on `λ_c`)? Hypothesis: yes, because early
  rollouts are not lineage-faithful enough to extract meaningful `c(y,p)`.
- **Open-ended verifier calibration.** The PES rubric judge can drift; we
  recalibrate every 200 steps (already specified in `plan.md` §4.2). Should
  also pin the judge prompt to a hash and forbid edits during a run.
- **Negative `c(y,p)` cases.** If the model proposes something INCONSISTENT
  with the predecessor lineage, should `c(y,p)` go to 0 (current default) or
  *negative* (active penalty)? Negative might over-penalise novelty.
  Conservatively default to 0; revisit in L3 ablation.
