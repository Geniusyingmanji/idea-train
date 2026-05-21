# v3 Research-Workflow Demo Dataset — Summary

**Generated**: 2026-05-20  |  **Total demos**: 1032  |  **Source**: GPT-5.5 + real OpenAlex candidates

## Archetype distribution

| Archetype | Count | % |
|---|---|---|
| W10_vague_to_concrete | 151 | 14.6% |
| W1_pure_discovery | 169 | 16.4% |
| W2_single_paper_extension | 109 | 10.6% |
| W3_multi_paper_synthesis | 91 | 8.8% |
| W4_literature_review | 142 | 13.8% |
| W5_critical_analysis | 75 | 7.3% |
| W6_cross_domain_bridge | 74 | 7.2% |
| W7_hypothesis_refinement | 90 | 8.7% |
| W8_reproduction_doubt | 63 | 6.1% |
| W9_tool_heavy_comparison | 68 | 6.6% |

## Discipline distribution

| Discipline | Count |
|---|---|
| chemistry | 112 |
| astronomy | 107 |
| biology | 102 |
| energy | 93 |
| physics | 91 |
| materials | 86 |
| mathematics | 79 |
| computer_science | 78 |
| neuroscience | 75 |
| earth_science | 74 |
| cs | 27 |
| earth | 20 |
| information | 18 |
| life | 14 |
| material | 12 |
| agriculture | 7 |
| astro | 7 |
| math | 7 |
| medicine | 7 |
| neuro | 6 |
| climate | 5 |
| ecology | 5 |

## Skill coverage

| Skill | # demos demonstrating |
|---|---|
| problem_clarification | 715 (69%) |
| lineage_construction | 541 (52%) |
| gap_identification | 1032 (100%) |
| method_transfer | 486 (47%) |
| critical_evaluation | 731 (71%) |
| multi_source_synthesis | 941 (91%) |
| hypothesis_specification | 581 (56%) |
| reflection_revision | 322 (31%) |

## Tool usage

**Avg tools per demo**: 7.78  (v1 = 3.74, v2 = 3.74)

| Tool | Total calls | Per-demo avg | % of all calls |
|---|---|---|---|
| read | 2523 | 2.44 | 31.4% |
| search | 2312 | 2.24 | 28.8% |
| propose | 1128 | 1.09 | 14.1% |
| extract_genome | 835 | 0.81 | 10.4% |
| genome_diff | 634 | 0.61 | 7.9% |
| novelty_check | 594 | 0.58 | 7.4% |

## Completion lengths

- min: 6524 chars
- median: 10360 chars (~2590 tokens)
- mean: 10546
- max: 47202

## Archetype × Skill matrix

(# demos where archetype's prompted skill OR text-detected skill present)

| Archetype | problem_clarification | lineage_construction | gap_identification | method_transfer | critical_evaluation | multi_source_synthesis | hypothesis_specification | reflection_revision |
|---|---|---|---|---|---|---|---|---|
| W10_vague_to_concrete | 151 | 151 | 151 | 54 | 102 | 149 | 106 | 14 |
| W1_pure_discovery | 169 | 169 | 169 | 73 | 91 | 169 | 87 | 13 |
| W2_single_paper_extension | 41 | 109 | 109 | 42 | 91 | 97 | 26 | 109 |
| W3_multi_paper_synthesis | 25 | 10 | 91 | 91 | 67 | 91 | 39 | 20 |
| W4_literature_review | 142 | 57 | 142 | 67 | 97 | 142 | 44 | 23 |
| W5_critical_analysis | 50 | 10 | 75 | 22 | 75 | 47 | 75 | 13 |
| W6_cross_domain_bridge | 9 | 8 | 74 | 74 | 44 | 74 | 31 | 15 |
| W7_hypothesis_refinement | 69 | 12 | 90 | 22 | 33 | 69 | 90 | 90 |
| W8_reproduction_doubt | 40 | 4 | 63 | 19 | 63 | 35 | 63 | 9 |
| W9_tool_heavy_comparison | 19 | 11 | 68 | 22 | 68 | 68 | 20 | 16 |

## What this dataset teaches

Each demo is a fully-played-out research trajectory in one of 10 workflow archetypes (see `research_workflows.md`). The dataset jointly trains the 8 research skills above. The agent learns:

1. **When to call which tool** — archetypes have characteristic patterns
2. **How to think between tool calls** — the rationale lines model expert reasoning
3. **How to ground proposals in real papers** — all `oa:W...` IDs are real OpenAlex works
4. **How to handle vague vs concrete prompts** — W1, W4, W10 give vague topics; W2, W5 give precise ones
5. **How to use structural reasoning tools** (extract_genome, genome_diff, novelty_check) for self-correction

## Training implications

- Use `max_len ≥ 3072` (median completion ≈ 2.5K tokens, max ~12K)
- Start from `qwen3-8b-sft-v3/final` (clean base) or `qwen3-8b-agentic-v2-sft/final` (already has agentic format)
- Use `LR=3e-5` (lower than v1 5e-5 due to longer context)
- 2 epochs should be enough; consider 1 if loss plateaus
- After SFT, evaluate on GENE-Arena + SGI-Bench + ArenaRL (when ready)
