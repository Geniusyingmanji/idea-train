# GeneTrace Data Format

> Concrete JSON schemas for the GeneTrace corpus. Drives both training-data
> generation (`tools/generate_sft_*.py`) and the open-source release
> (`paper_positioning.md` §4). Authoritative source for field semantics;
> `evo_opd/schemas.py` is the in-code mirror that must stay in sync.

GeneTrace has **three levels** of annotation, layered:

```
Level 1: GenomeCard(paper)        ←  one record per paper
Level 2: DynamicsEdge(p → q)      ←  one record per ordered paper pair on a lineage
Level 3: LineageChain([p₀,…,pₙ])  ←  one record per multi-paper chain
```

Plus a fourth artefact bundled with the release:
```
Level 4: VerifierBundle           ←  per-annotation automated scores
                                     + reasoning trace from the teacher
```

---

## 1. Level 1 — GenomeCard

One JSON object per paper. **Required:** all six gene fields; **optional:** evidence quotes per field.

```json
{
  "card_id": "card::<paper_id>",
  "paper_id": "paper:<slug>:<year>",          // canonical paper key
  "year": 2016,
  "title": "...",
  "domain": ["cs.LG"],                         // ArXiv subjects or equivalent
  "source_text": "<concatenated title + abstract + intro>",

  "genome": {
    "mechanism_genome":   "<one-paragraph claim of HOW the paper works>",
    "niche_genome":       "<one-paragraph claim of the PROBLEM addressed>",
    "observation_genome": "<one-paragraph claim of the EMPIRICAL FINDING>",
    "limitation_genome":  "<one-paragraph claim of what it CANNOT do>",
    "delta_genome":       "<what is NEW vs the closest predecessor; '' if no predecessor>",
    "claim_genome":       "<the headline claim the paper makes about its own contribution>"
  },

  "evidence": {
    "mechanism_genome":   [{"quote": "...", "char_offset": 1234}],
    "niche_genome":       [...],
    "observation_genome": [...],
    "limitation_genome":  [...],
    "delta_genome":       [...],
    "claim_genome":       [...]
  },

  "provenance": {
    "teacher_model":      "gpt-5.5",
    "teacher_api_version": "2024-12-01-preview",
    "teacher_input_tokens": 1840,
    "teacher_output_tokens": 412,
    "prompt_hash":        "sha256:abc123…",
    "generation_ts_utc":  "2026-05-17T19:02:14Z",
    "version":            "genetrace-v0.1"
  },

  "verifier": {                                // see Level 4
    "schema_valid":       1.0,
    "evidence_grounded":  0.85,
    "v_total":            0.93,
    "verifier_version":   "evo_opd.verifier@a1b2c3"
  },

  "safety": {
    "in_denylist":        false,               // hard constraint: must be false
    "denylist_version":   "v0",
    "pre_2017":           true                 // soft pre-filter
  }
}
```

Field-level conventions:
- `paper_id` is the slugified citation key from IdeaEvolving's `paper_db` plus a year suffix; matches `safe_pool` keys 1:1.
- All gene fields are *free-text claims*, not entity names. They are deliberately
  prose, not key-value, because the down-stream task (lineage reasoning) needs
  the rationale, not just the label.
- Empty string `""` in `delta_genome` is meaningful (no traceable predecessor),
  not a missing value. `null` is forbidden anywhere in the schema.
- `evidence[field]` is OPTIONAL but should be present for ≥80% of records in v0.1
  (we measure this in the dataset card).

---

## 2. Level 2 — DynamicsEdge

One JSON object per ordered paper pair `(p → q)` that we have annotated. **Sparse:**
only pairs with at least one teacher-annotated dynamics label are included;
unannotated pairs are absent (no false negatives).

```json
{
  "edge_id":   "edge::<p>::<q>",
  "p_paper_id": "paper:foo_2015",
  "q_paper_id": "paper:bar_2016",
  "p_card_id": "card::paper:foo_2015",
  "q_card_id": "card::paper:bar_2016",

  "dynamics":  "Mutation",                     // one of 5; see schemas.py
  "driver":    "limitation",                   // which of p's 6 fields drove the change
  "gene_fates": {                              // per-field transition tags
    "mechanism_genome":   "MUTATED",
    "niche_genome":       "INHERITED",
    "observation_genome": "NOVEL",
    "limitation_genome":  "INHERITED",
    "delta_genome":       "NOVEL",
    "claim_genome":       "MUTATED"
  },

  "evidence": {
    "p_limitation_quote": "...the model fails to scale beyond 1B parameters...",
    "q_mechanism_quote":  "...we propose a sparse-attention variant that scales to 13B..."
  },

  "reasoning_trace": "<GPT-5.5 chain-of-thought, ~200-500 tokens>",

  "provenance": { ... },                       // same shape as GenomeCard.provenance
  "verifier": {
    "schema_valid":     1.0,
    "dynamics_valid":   1.0,                   // dynamics ∈ enum
    "fate_consistent":  1.0,                   // fates respect dynamics constraints
    "evidence_grounded": 0.91,
    "v_total":          0.95,
    "verifier_version": "evo_opd.verifier@a1b2c3"
  },

  "safety": {
    "both_in_safe_pool": true,                 // hard constraint
    "denylist_version":  "v0"
  }
}
```

Constraints enforced by the verifier (see `evo_opd/verifier.py`):
- `dynamics ∈ {Mutation, Adaptive Radiation, Hybridization, Speciation, Niche Competition}`.
- `driver ∈ {mechanism, niche, observation, limitation}`.
- `gene_fates[field] ∈ {INHERITED, MUTATED, LOST, NOVEL, HYBRIDIZED}`.
- For `dynamics == "Hybridization"`: at least one fate must be `HYBRIDIZED`.
- For `dynamics == "Niche Competition"`: `niche_genome` fate must be `MUTATED` or `LOST`.
- For `dynamics == "Speciation"`: ≥3 fates must be `MUTATED` or `NOVEL`.

These constraints make the verifier non-trivial and let evo-OPD use the
verifier as a structured reward (not just a 0/1 exact-match flag).

---

## 3. Level 3 — LineageChain

One JSON object per chain of length 3+ (pairs are handled at Level 2).

```json
{
  "chain_id":   "chain::lin_042",
  "members":    ["paper:foo_2014", "paper:bar_2015", "paper:baz_2016", "paper:qux_2017"],
  "card_ids":   ["card::paper:foo_2014", ...],
  "edges":      ["edge::paper:foo_2014::paper:bar_2015", ...],
  "domain":     "cs.LG",                       // dominant domain across the chain

  "per_step_dynamics": ["Mutation", "Adaptive Radiation", "Hybridization"],

  "chain_summary": "<2-3 sentence narrative of the chain>",
  "chain_reasoning_trace": "<GPT-5.5 chain-of-thought explaining the chain>",

  "provenance": { ... },
  "verifier": {
    "all_pairs_present":   1.0,                // every consecutive pair has a Level-2 edge
    "dynamics_consistent": 1.0,                // chain narrative agrees with per-step labels
    "v_total":             0.95
  },

  "safety": {
    "all_in_safe_pool": true,
    "denylist_version": "v0"
  }
}
```

---

## 4. Level 4 — VerifierBundle (bundled with the release)

A separately distributable JSON-Lines file mirroring every annotation's
verifier subscores, for consumers who want to:
- re-score with a different verifier version,
- filter by score threshold (e.g. only `v_total ≥ 0.9`),
- use the scores as RL rewards directly.

One record per annotation:

```json
{"annotation_id": "card::paper:foo_2015", "level": 1, "scores": {...}}
{"annotation_id": "edge::paper:foo_2015::paper:bar_2016", "level": 2, "scores": {...}}
{"annotation_id": "chain::lin_042", "level": 3, "scores": {...}}
```

`scores` is identical to the inline `verifier` object on each Level 1/2/3 record;
we duplicate to give consumers a single-file fast path without parsing the full
corpus.

---

## 5. Release tiers (mirror of `paper_positioning.md` §4)

| Release | Cards | Edges | Chains | Reasoning traces | Status |
|---|---|---|---|---|---|
| **v0.1** (paper) | 5K | 2K | 200 | ~7K | exists as raw `data/stage1_sft/*.jsonl`, needs schema normalisation |
| **v0.2** (post-acceptance) | 20K | 8K | 1K | ~30K | OpenAlex expansion in progress |
| **v1.0** (production) | 50K+ | 25K+ | 5K+ | ~80K+ | future, multi-teacher consensus |

For each release we additionally ship:
- `verifier_bundle.jsonl` (Level 4)
- `denylist.jsonl` + `safe_pool.jsonl` (contamination guard)
- `min_k_report.json` (leakage diagnostic — measure Min-K%++ probability that GENE-Exam paper text was memorised; should be near baseline for our pre-2017 safe pool)
- `dataset_card.md` (HuggingFace format)

---

## 6. Contamination guard (the explicit promise to the community)

The release ships with three artefacts that let any consumer reproduce the
contamination check end-to-end:

1. **`denylist.jsonl`** — the 8,359 papers extracted from IdeaEvolving's
   `paper_db/`, plus the v1 OpenAlex 1-hop expansion when ready. Format:
   ```json
   {"paper_id": "...", "s2_id": "...", "openalex_id": "...", "source": "ideaevolving_db|openalex_expansion"}
   ```
2. **`safe_pool_construction.py`** — the exact script that produces the
   pre-2017 ∖ denylist intersection. Deterministic given the inputs.
3. **`min_k_report.json`** — per-domain Min-K%++ leakage scores measured on
   GENE-Exam paper text. We document that our released corpus is < threshold
   on this metric (target: Min-20%++ < 0.05 on the held-out exam papers).

The contract with downstream users:
> If you train on GeneTrace v0.1 and evaluate on GENE-Exam, your training
> data is *provably disjoint* from the evaluation papers under the published
> denylist + safe-pool methodology. We provide the scripts to verify this
> yourself.

---

## 7. How this schema maps to the existing code (today)

| Schema concept | Existing artefact | Action required |
|---|---|---|
| Level 1 GenomeCard | `data/stage1_sft/train.jsonl` (round-1 gene_card_extract) | Add `evidence` extraction step; normalise field names; add `verifier` subscores |
| Level 2 DynamicsEdge | implicit in `T3-09_relation_classify` SFT examples (300 records) | Promote to first-class JSON record; harvest `gene_fates` from verifier output |
| Level 3 LineageChain | not yet built | Construct via DFS on Level-2 edges; teacher-write the `chain_summary` |
| Level 4 VerifierBundle | partial — `metadata.verifier_score` per training row | Extract to a separate file at release time |
| Contamination guard | `denylist/denylist_v0.{csv,jsonl}` + `tools/build_safe_pool.py` | Add Min-K% diagnostic script |

Estimated effort to ship v0.1: 2–3 days of scripting + 1 day of teacher calls
for evidence extraction on the existing 856 cards.

---

## 8. Open questions (to resolve before v0.1 freeze)

- **Reasoning trace length cap.** Currently uncapped; GPT-5.5 traces can be 1K+
  tokens. For release, cap at 1024 tokens? Drives storage size.
- **Per-domain stratification.** Currently CS-only. v0.2 should add bio/physics;
  do we ship v0.1 with a single-domain note, or wait for cross-domain?
- **License of GPT-5.5 reasoning traces.** Azure OpenAI terms allow downstream
  use of model outputs. We must verify and cite the relevant clause in the
  dataset card. **Action: legal review before release.**
- **Versioning.** Pin `genetrace-v0.1` to a git tag; downstream consumers should
  cite by version. Use semver: `vMAJOR.MINOR.PATCH`.
