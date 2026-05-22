# Data Format Spec

All artifacts across Stage 0 → Stage 4 use line-delimited JSON (`.jsonl`) as the primary on-disk format. Parquet/CSV are derived views for inspection only.

Common conventions:
- All text fields are UTF-8.
- `null` is allowed and meaningful (e.g., `s2_id: null` ≠ `s2_id: ""`).
- `year` is integer (e.g., `2024`), not string.
- `normalized_title`: lowercase, punctuation stripped to spaces, whitespace collapsed (see `tools/build_denylist.py:norm_title`).
- Every record carries a stable `source` or `sources` tag so we can trace provenance.

---

## Stage 0 — Contamination firewall

### `denylist/denylist_v0.jsonl`
One paper per line. The seed denylist (no OpenAlex expansion yet).

```json
{
  "s2_id": "33161a5a9b5dcb635b5a97475e6a6209a69ada7d",       // Semantic Scholar SHA; null if unknown
  "arxiv_id": null,                                            // e.g., "2503.14476"; usually null
  "doi": null,                                                  // when populated
  "title": "The AI Scientist: Towards Fully Automated...",     // canonical title; cross-resolved when possible
  "year": 2024,
  "venue": null,                                                // when populated
  "internal_paper_id": "paper:the_ai_scientist...:2024",       // IdeaEvolving's internal key
  "trace_id": "cs_AutoResearch",
  "domain": "cs",
  "subfield": "AutoResearch",
  "role": "lineage",                                            // for trace_graph nodes: lineage|branch|foundation|competitor
  "s2_id_via": "paper_db_xref",                                 // present only if s2_id was cross-resolved
  "sources": ["gene_card", "trace_graph"]                       // which IdeaEvolving asset(s) referenced it
}
```

### `denylist/denylist_v1.jsonl`
v0 plus OpenAlex 1-hop closure. Adds:

```json
{
  ... (all v0 fields),
  "openalex_id": "https://openalex.org/W2741809807",
  "denylist_tier": "seed" | "1hop_referenced" | "1hop_citedby"
}
```

`tier=seed` means it was in v0; `tier=1hop_*` means it was pulled in by reference closure of a v0 seed.

### `denylist/denylist_stats.json`
Summary counts; used by audits and the contamination report at release time. Shape is dict-of-dicts; see `tools/build_denylist.py` output.

---

## Stage 1 — SFT mixture (~80K examples)

All SFT examples share one envelope:

```json
{
  "instance_id": "sft_abc123",                          // unique, stable
  "task_type": "gene_card_extract",                     // one of the 8 below
  "domain": "biology",                                   // for stratified analysis
  "subfield": "Single-Cell Foundation Models",
  "prompt": "...",                                       // string the model is conditioned on
  "completion": "...",                                   // string the model should produce
  "metadata": {                                          // task-specific fields
    "source_paper_id": "openalex:W...",                  // origin paper (must be in Safety Pool)
    "teacher_model": "gpt-5.5",                          // who generated the completion
    "teacher_call_timestamp": "2026-05-19T12:34:56Z",
    "verification": {                                    // per-example quality signals
      "schema_valid": true,
      "evidence_citation_frac": 0.96,
      "round_trip_bertscore": 0.78,
      "rejection_sample_pass": true                      // base model was wrong on this; we kept it
    }
  },
  "split": "train" | "val"                               // 99/1 split
}
```

### The 8 SFT task types

| `task_type` | Share | `prompt` shape | `completion` shape |
|---|---|---|---|
| `gene_card_extract` | 32% | `<paper text/abstract>` | gene-card JSON (6 fields + evidence quotes) |
| `genome_diff_annotate` | 16% | `[parent card]\n[child card]` | `{fates: {...}, driver: "...", dynamics: "..."}` |
| `lineage_trace_reconstruct` | 8% | `[shuffled cards]` | `{ordering: [...], per_edge_dynamics: [...]}` |
| `lineage_verify` | 8% | `[proposed lineage with possible defect]` | `{valid: bool, failure_mode: "...", repair: "..."}` |
| `idea_generate` | 8% | `[trace] + [open question]` | `{problem, mechanism, expected_contribution, lineage_connection}` |
| `free_form_science_qa` | 15% | open science question | free-form answer |
| `general_reasoning` | 10% | math/code/STEM problem | long-CoT solution |
| `general_chat` | 3% | chat prompt | chat response (Tulu-3 minus SciRIFF) |

### Detailed task shapes

**`gene_card_extract` — `completion` schema**

```json
{
  "mechanism_genome": "Bidirectional transformer encoder with masked language modeling on DNA k-mers.",
  "niche_genome": "Genomic sequence representation; downstream tasks include promoter prediction.",
  "observation_genome": "Outperforms task-specific models by 1-5% AUC on TF binding.",
  "limitation_genome": "Limited to 512 tokens; k-mer tokenization loses single-nucleotide resolution.",
  "delta_genome": "First BERT-style pretraining for DNA sequences.",
  "claim_genome": "Pretrained genomic LMs enable transfer learning across DNA tasks.",
  "evidence_quotes": [                                   // each quote MUST appear verbatim in prompt
    {"field": "mechanism_genome", "quote": "BERT masked language modeling on human reference genome"},
    {"field": "observation_genome", "quote": "outperforms prior task-specific models on promoter prediction"}
  ]
}
```

**`genome_diff_annotate` — `completion` schema**

```json
{
  "fates": {
    "mechanism_genome": "INHERITED",
    "niche_genome": "MUTATED",
    "observation_genome": "NOVEL",
    "limitation_genome": "MUTATED",
    "delta_genome": "NOVEL",
    "claim_genome": "NOVEL"
  },
  "driver": "mechanism",                                  // ∈ {mechanism, niche, observation, limitation}
  "dynamics": "Adaptive Radiation",                       // ∈ DYNAMICS_LABELS
  "rationale": "Same encoder mechanism, applied to a new biological niche."
}
```

**`lineage_trace_reconstruct` — `completion` schema**

```json
{
  "ordering": [2, 1, 4, 5, 3],                            // 1-indexed positions
  "per_edge_dynamics": ["Mutation", "Adaptive Radiation", "Mutation", "Hybridization"],
  "rationale": "..."
}
```

**`lineage_verify` — `completion` schema**

```json
{
  "valid": false,
  "failure_mode": "wrong_step",                           // ∈ {intruder, wrong_step, missing_link, citation_conflict, valid}
  "specific_defect": "Position 3 (DETR) cannot inherit from position 2 (BERT) because mechanisms are disjoint.",
  "repair": "Replace position 3 with ViT, which inherits Transformer attention from position 2."
}
```

**`idea_generate` — `completion` schema** (mirrors GENE-Arena)

```json
{
  "name": "GenomicNeRF",
  "problem": "Continuous-coordinate representation of long-range chromatin contacts...",
  "mechanism": "Apply NeRF-style coordinate MLPs to 3D Hi-C contact maps...",
  "expected_contribution": "Methods + dataset; first NeRF for genome 3D structure.",
  "lineage_connection": {                                  // structured pointers — required for Arena Heredity
    "parents": ["NeRF (Mildenhall 2020)", "Hi-C foundation models (2023)"],
    "inherits": "coordinate MLP + positional encoding",
    "repairs_limitation": "voxel grids don't scale to genome-wide Hi-C",
    "dynamics": "Hybridization"
  },
  "evaluation_plan": "..."
}
```

---

## Stage 2 — GRPO prompts

Same envelope as SFT but **no `completion` field**; reward is computed at rollout time. Adds reward-side metadata:

```json
{
  "instance_id": "grpo_xyz789",
  "task_type": "T4-01_consistency_check",                 // T1-T4 exam types are first-class here
  "prompt": "...",
  "metadata": {
    "reward_head": "exam",                                 // ∈ {exam, math, arena}
    "gold_answer": {"label": "C", "verify": ["T","F","F","T"]},  // for exam/math heads
    "source_text_for_evidence_check": "...",               // for arena head's verifier
    "parent_card": null                                    // populated for lineage tasks
  }
}
```

Reward is one of three heads, summed with `w_exam=0.4 + w_math=0.2 + w_arena=0.4` in the trainer config:

- **exam head:** exact-match against `metadata.gold_answer` (0/1, all-or-nothing).
- **math head:** sympy/numeric for math; pytest for code (0/1).
- **arena head:** PES rubric scored by GPT-5.5 frozen judge (0-100, normalized to [0,1]); Heredity weight ×1.5.

---

## Stage 3 — evo-OPD inputs/outputs

### Per-rollout input (one record per prompt, prepared by trainer)

```json
{
  "prompt": "...",
  "task_type": "genome_diff_annotate",                    // routes to schema + reward
  "source_text": "...",                                    // for verifier's evidence_citation_frac (may be null)
  "gold_answer": {...},                                    // for verifier's exact_match (may be null)
  "parent_card": {                                         // for lineage c(y,p); null if not parent-child task
    "legacy_genome": {
      "mechanism_genome": "...",
      "niche_genome": "...",
      "observation_genome": "...",
      "limitation_genome": "...",
      "delta_genome": "...",
      "claim_genome": "..."
    }
  }
}
```

### Per-rollout output (one record per sample, logged for diagnostics)

```json
{
  "prompt_id": "evo_opd_step42_prompt7_sample3",
  "completion": "...",                                     // student rollout text
  "verifier": {
    "v": 0.65, "schema_valid": 1.0, "evidence_citation_frac": 0.8,
    "dynamics_consistency": 1.0, "exact_match": 0.0,
    "components_active": ["schema","evidence","dynamics","exact"]
  },
  "lineage": {                                             // null if no parent_card provided
    "c": 0.72, "schema_complete": 1.0, "fate_content_consistency": 0.83,
    "dynamics_from_fates": 1.0, "anti_copy": 1.0,
    "per_field_match": {...}
  },
  "kl_summary": {                                          // for diagnostics
    "kl_term_mean": -0.082,
    "verifier_term_mean": 0.015,
    "lineage_term_mean": 0.008,
    "alpha_distribution": {"boilerplate": 0.42, "content_field": 0.35,
                            "evidence_span": 0.05, "dynamics_label": 0.02,
                            "gold_answer": 0.08, "unknown": 0.08}
  },
  "step": 42,
  "timestamp": "2026-05-19T13:22:45Z"
}
```

### Per-token reward tensor (passed to trainer, not persisted)

`PerTokenReward` dataclass from `evo_opd/rewards.py`:

```python
rewards:  list[float]   # length = n_tokens; r_t = -α·kl + α·λ_v·v + α·λ_c·c
alphas:   list[float]   # for diagnostics
phi:      list[str]     # role per token: boilerplate/content_field/.../gold_answer
fields:   list[str|None] # which gene field this token is in, if any
verifier: VerifierScore  # one per rollout
lineage:  LineageScore | None
kl_term_mean / verifier_term_mean / lineage_term_mean: float
```

---

## Stage 4 — Eval logs

### Per-benchmark result

```json
{
  "benchmark": "gene_exam_main_challenge",                // or gpqa_diamond, ideabench, etc.
  "checkpoint": "idea-8b-evo-opd-step300",
  "n_instances": 1029,
  "macro_score": 0.273,                                    // exact accuracy (task-macro)
  "per_tier_score": {"T1": 0.31, "T2": 0.28, "T3": 0.26, "T4": 0.21},
  "per_task_score": {"T1-01_contribution_type": 0.32, ...},
  "config": {
    "vllm_version": "0.7.x",
    "max_output_tokens": 16384,
    "concurrency": 32,
    "temperature": 0.0,
    "seed": 42
  },
  "wallclock_seconds": 12340,
  "input_tokens_total": 1234567,
  "output_tokens_total": 234567
}
```

### Contamination probe result (gated for public release)

```json
{
  "checkpoint": "idea-8b-evo-opd-step300",
  "denylist_version": "v1",
  "probes": {
    "canary_guid_reproduction": {"n_probes": 100, "n_reproduced": 0, "pass": true},
    "min_k_prob": {"denylist_mean": -2.34, "control_mean": -2.31, "gap_abs": 0.03, "pass": true},
    "time_travel_completion": {"n_probes": 100, "n_exact_match": 0, "pass": true},
    "paraphrase_mirror": {"original_score": 0.275, "paraphrased_score": 0.268, "abs_gap": 0.007, "pass": true}
  },
  "overall_pass": true
}
```

---

## Directory layout for produced data

```
idea_train/data/
├── stage1_sft/
│   ├── train.jsonl                                       # ~80K examples
│   ├── val.jsonl                                          # ~800 examples (1%)
│   └── provenance.jsonl                                   # per-example: source_paper_id, teacher, timestamp, denylist_proof
├── stage2_grpo/
│   ├── prompts.jsonl                                      # ~10K prompts (no completions)
│   └── rollout_logs/step_{N}.jsonl                        # per-step rollout + reward logs
├── stage3_evo_opd/
│   ├── prompts.jsonl                                      # separate from Stage 2 prompts
│   └── rollout_logs/step_{N}.jsonl
└── eval/
    ├── gene_exam/*.json
    ├── gene_arena/*.json
    ├── generality_suite/*.json
    └── contamination_probes/*.json
```

---

## What's already implemented vs spec'd

| Component | Spec'd here | Code exists |
|---|---|---|
| denylist v0 schema | ✓ | ✓ `tools/build_denylist.py` |
| denylist v1 schema | ✓ | ✓ smoke test only (full run pending threading) |
| Stage 1 SFT envelope + 8 task types | ✓ | ✗ data not generated yet (Stage 1 work) |
| GRPO prompt envelope | ✓ | ✗ |
| evo-OPD input/output shapes | ✓ | ✓ `evo_opd/rewards.py:PerTokenReward` |
| eval log shapes | ✓ | ✗ |
| contamination probe log | ✓ | ✗ |
