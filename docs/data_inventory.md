# Training Data Inventory

Generated 2026-05-22 after ~14h autonomous expansion (v5 → v29). Final corpus
state, sourced from `data/agentic_combined_v3to29/`, `data/dpo_combined/`, and
`data/rl_prompts_combined/`.

## Headline numbers

| Pool | Count | Path |
|---|---:|---|
| **SFT demos** | **5773** | `data/agentic_combined_v3to31/sft_demos.jsonl` |
| **DPO pairs** | **2593** | `data/dpo_combined/preferences.jsonl` |
| **RL prompts** | **2930** | `data/rl_prompts_combined/rl_prompts.jsonl` |

**Total: 11036 training items.** 65 unique disciplines. 1247 Chinese demos (23%).

## SFT — 17 source rounds

Length-tier distribution: **61% short (1-3 tools) / 15% med / 24% long**. Median 3 tools/demo.
v3 alone was 11/12/77, so the bias has been deliberately inverted.

| Round | n | What it covers | Why |
|---|---:|---|---|
| `agentic_v3` | 1032 | 10 research-workflow archetypes (W1-W10) | Foundation — diversity in research patterns |
| `agentic_v4` | 322 | 10 new disciplines × 7 prompt styles, explicit length tiers | Discipline + style diversity beyond v3 |
| `agentic_v5` | 487 | Short-focused (62% ≤3 tools), schema-aware (gene_genome / idea_plan / free_text) | Fix v3's over-tooling bias; first multi-schema training |
| `agentic_v6` | 310 | Bilingual (50/50 EN/ZH), 3 schemas balanced | Multi-language + multi-schema robustness |
| `agentic_v8` | 396 | Failure recovery (6 scenarios) + bench-specific (SGI strict idea_plan, ArenaRL travel, GENE-Arena cross-cut) | Resilient behavior + bench-specific schema rigor |
| `agentic_v9` | 262 | 2-turn refinement (v1) + ArenaRL travel-zh + cross-discipline pairs | First multi-turn dialogue, more bench coverage |
| `agentic_v11` | 471 | Direct-propose (60%) / single-search-confirm (40%) — mean 1.40 tools/demo | Target the 18% propose rate on v3 SFT |
| `agentic_v12` | 278 | Pure Chinese, 15 disciplines, 80% short, mixed schemas | Boost ZH from 8% → 16% of corpus |
| `agentic_v14` | 275 | 2-turn refinement v2 (clean structure, 100% yield) | Fix v9's 22% multi-turn yield via simpler format |
| `agentic_v15` | 385 | ZH (58%) + free_text_answer (42%) | Lift under-represented schema + lang |
| `agentic_v16` | 280 | Ambiguous-prompt clarification (6 ambiguity types) | Surface assumptions on under-specified prompts |
| `agentic_v18` | 173 | Self-correction (5 mid-trajectory pivot patterns) | Agent detects own error before propose |
| `agentic_v21` | 246 | ZH style-varied (工业实践/政策咨询/学者发问/同行评议/会议征稿) | Diversify ZH beyond research-question style |
| `agentic_v23` | 147 | Math + formal sciences (8 domains) | Fill the sparse formal-sciences gap |
| `agentic_v24` | 151 | Lineage-tool chains (forces extract_genome + genome_diff) | Raise rare-tool usage (was 21% / 15%) |
| `agentic_v26` | 79 | 10 fresh disciplines (quantum_info, synth_bio, climate, HCI, …) | SFT for v25's RL-only disciplines |
| `agentic_v29` | 219 | ZH for 10 fresh disciplines | Bilingual coverage for the new disciplines |
| `agentic_v30` | 120 | Cross-lingual: EN prompt → ZH rationale + ZH gene_genome | Trains cross-lingual research reasoning |
| `agentic_v31` | 140 | Reverse cross-lingual: ZH prompt → EN rationale + EN gene_genome | Complements v30 for bidirectional transfer |

## DPO pairs — 6 rounds, 7 corruption modes

| Round | n | Approach |
|---|---:|---|
| `agentic_v7` | 91 | Tournament-style, 6 rejection modes (chosen/rejected as independent calls; 34% yield) |
| `agentic_v10` | 98 | Focused 2-mode (wrong_schema, premature_propose; 41% yield) |
| `agentic_v13` | 961 | **Corruption approach: take existing SFT demo as chosen, corrupt it as rejected. 95% yield.** |
| `agentic_v19` | 571 | v13 with fresh seed (2033) |
| `agentic_v22` | 575 | v13 with fresh seed (2037) |
| `agentic_v27` | 297 | v13 with seed 2039, focused 3-mode |

**Rejection-mode distribution (combined):**

| Mode | n |
|---|---:|
| wrong_schema | 552 |
| premature_propose | 532 |
| no_evidence | 481 |
| schema_collapse | 383 |
| truncated | 370 |
| verbose_padding | 272 |
| made_up_papers | 3 |

## RL prompts — 4 rounds

| Round | n | What it covers |
|---|---:|---|
| `agentic_v10/rl_prompts.jsonl` | 1190 | 20 disc × 3 schemas × 2 langs |
| `agentic_v20/rl_prompts.jsonl` | 800 | 20 disc × style combos (peer-review, CFP, academic, learner, industry, policy) |
| `agentic_v25/rl_prompts.jsonl` | 540 | 10 fresh disciplines (quantum_info, synth_bio, climate, HCI, …) × 3 styles |
| `agentic_v28/rl_prompts.jsonl` | 400 | ZH for the 10 fresh disciplines |

## Key empirical finding (motivation)

v3 SFT model's GENE-Arena rollouts (150 ideas) had only **18% propose rate** (27/150) — the model over-tooled and never reached propose. Rounds v5/v11/v16/v18/v29 directly target this by biasing short trajectories, training direct-propose, and teaching self-correction.

## Tool-coverage of SFT corpus

In `agentic_combined_v3to29/sft_demos.jsonl`:

| Tool | demos using | % |
|---|---:|---:|
| search | most | ~95 |
| read | most | ~80 |
| propose | all | 100 |
| extract_genome | ~1200 | 22 |
| genome_diff | ~900 | 16 |
| novelty_check | sparse | ~10 |

v24 (lineage chains) raised extract_genome / genome_diff floor; v8-recovery `novelty_says_not_novel_revise` is the main novelty_check source.

## Excluded / historical

- `agentic_v1` (246 simple SFT demos, early), `agentic_v2` (287 web-search demos), `agentic_v3_v4_combined` (1354, superseded by v3to29), `agentic_v25` (RL prompts only — included in `rl_prompts_combined`).
- v7/v10/v13/v19/v22/v27 are NOT in `agentic_combined_v3to29` (those are DPO pairs, not SFT demos) but are in `dpo_combined`.

## Compatibility with existing trainer

`tools/sft_agentic_v3.py` reads `{"full_prompt", "completion"}`. The combined file has both keys for every row, plus discipline / lang / archetype metadata for filtering.
