# Data

Single source of truth for: corpus state, schema spec, design rationale.

## 1. Current corpus

| Pool | Count | Path |
|---|---:|---|
| **SFT demos** | **6506** | `data/agentic_combined_v3to42/sft_demos.jsonl` |
| **DPO pairs** | **3124** | `data/dpo_combined/preferences.jsonl` (15 rejection modes) |
| **RL prompts** | **3050** | `data/rl_prompts_combined/rl_prompts.jsonl` |

**Total: 11476 training items.** 73 unique disciplines. 1247 Chinese demos (21%). Median 3 tools/demo. 64% short trajectories (1-3 tools) — was 11% in v3 alone.

## 2. SFT — 25 source rounds

Length-tier: **64% short / 14% med / 22% long**. (Inverted from v3's 11/12/77.)

| Round | n | What it covers |
|---|---:|---|
| `agentic_v3` | 1032 | 10 research-workflow archetypes (W1-W10) — foundation |
| `agentic_v4` | 322 | 10 new disciplines × 7 prompt styles, length tiers |
| `agentic_v5` | 487 | Short-focused (62% ≤3 tools), schema-aware (gene_genome / idea_plan / free_text) |
| `agentic_v6` | 310 | Bilingual EN/ZH 50/50 × 3 schemas |
| `agentic_v8` | 396 | Failure recovery (6 scenarios) + bench-specific (SGI strict, ArenaRL travel, GENE-Arena cross-cut) |
| `agentic_v9` | 262 | 2-turn refinement (v1) + ArenaRL travel-zh + cross-discipline pairs |
| `agentic_v11` | 471 | Direct-propose / single-search-confirm (mean 1.40 tools/demo) |
| `agentic_v12` | 278 | Pure Chinese, 15 disciplines, 80% short |
| `agentic_v14` | 275 | 2-turn refinement v2 (clean structure, 100% yield) |
| `agentic_v15` | 385 | ZH + free_text_answer focus |
| `agentic_v16` | 280 | Ambiguous-prompt clarification (6 ambiguity types) |
| `agentic_v18` | 173 | Self-correction (5 mid-trajectory pivot patterns) |
| `agentic_v21` | 246 | ZH style-varied (工业/政策/学者/同行评议/会议征稿) |
| `agentic_v23` | 147 | Math + formal sciences (8 domains) |
| `agentic_v24` | 151 | Lineage-tool chains (forces extract_genome + genome_diff) |
| `agentic_v26` | 79 | 10 fresh disciplines (quantum_info, synth_bio, climate, HCI, …) |
| `agentic_v29` | 219 | ZH for 10 fresh disciplines |
| `agentic_v30` | 120 | Cross-lingual: EN prompt → ZH rationale + ZH gene_genome |
| `agentic_v31` | 140 | Reverse cross-lingual: ZH prompt → EN response |
| `agentic_v32` | 60 | Long-context dense prompts (4-7 sentences w/ prior-work context) |
| `agentic_v33` | 120 | Soft-sciences (legal, journalism, music_audio, sports, architecture, ...) |
| `agentic_v34` | 77 | Evidence-grounded — gene_genome explicitly cites paper_id from search results in 2+ fields |
| `agentic_v35` | 135 | ZH evidence-grounded (Chinese complement to v34) |
| `agentic_v36` | 62 | Multi-turn + evidence-grounded combined (2 propose calls, 2nd cites paper_id) |
| `agentic_v38` | 105 | ZH multi-turn refinement (Chinese complement to v14) |
| `agentic_v41` | 60 | Direct-only propose for fresh disciplines (strict 1-action, no search) |
| `agentic_v42` | 114 | novelty_check chain — search + novelty_check + propose (lifts rare tool usage) |

## 3. DPO — 6 rounds, 7 corruption modes

Combined yield: **3124 pairs**. `v13 corruption-style` had 95% pair yield vs tournament-style's 34-41% (see `experiment.md` §3).

7 rounds total. v37 adds 4 new corruption modes complementing the original 7. **Mode distribution:** wrong_schema 552, premature_propose 532, no_evidence 481, schema_collapse 383, truncated 370, verbose_padding 272, mode_collapse 75, fake_citation 75, contradictory_fields 74, shallow_search 74, made_up_papers 3.

| Round | n | Approach |
|---|---:|---|
| `agentic_v7` | 91 | Tournament-style, 6 rejection modes (34% yield) |
| `agentic_v10` | 98 | Focused 2-mode (wrong_schema, premature_propose, 41% yield) |
| `agentic_v13` | 961 | **Corruption: chosen=existing SFT demo, rejected=GPT corrupts it** (95% yield) |
| `agentic_v19` | 571 | v13 with fresh seed (2033) |
| `agentic_v22` | 575 | v13 with fresh seed (2037) |
| `agentic_v27` | 297 | v13 with seed 2039, focused 3-mode |
| `agentic_v37` | 298 | **4 new corruption modes**: shallow_search, contradictory_fields, fake_citation, mode_collapse |
| `agentic_v40` | 233 | **4 more new corruption modes**: over_hedging, jargon_overload, mismatch_discipline, truncated_propose |

**Rejection-mode distribution:** wrong_schema 552, premature_propose 532, no_evidence 481, schema_collapse 383, truncated 370, verbose_padding 272, made_up_papers 3.

## 4. RL prompts — 4 rounds

| Source | n | What it covers |
|---|---:|---|
| `agentic_v10/rl_prompts.jsonl` | 1190 | 20 disc × 3 schemas × 2 langs |
| `agentic_v20/rl_prompts.jsonl` | 800 | 20 disc × style combos (peer-review/CFP/academic/learner/industry/policy) |
| `agentic_v25/rl_prompts.jsonl` | 540 | 10 fresh disciplines × 3 styles |
| `agentic_v28/rl_prompts.jsonl` | 400 | ZH for the 10 fresh disciplines |
| `agentic_v39/rl_prompts.jsonl` | 120 | Long-context (4-6 sentence prompts with prior-work citations) |

## 5. Schema spec

### SFT demo record

```json
{
  "prompt_id": "<round>::<disc>::<idx>::<tier>::<schema>",
  "source": "synthetic_v<N>",
  "discipline": "computer_science",
  "lang": "en" | "zh" | "en_prompt_zh_resp" | "zh_prompt_en_resp",
  "length_tier": "very_short" | "short" | "medium" | "long",
  "archetype": "<round-specific>",
  "topic": "[<round>/<axis>/<disc>] <prompt preview>",
  "full_prompt": "<user prompt, what the SFT trainer feeds as user message>",
  "completion": "<agent trajectory: rationale + ```action JSON``` + [result] ... ending with propose>",
  "candidates": [...],                    // OpenAlex prefetched candidates
  "input_tokens": int, "output_tokens": int, "latency_ms": int,
  "_source_round": "agentic_v<N>"         // added by merge script
}
```

Trainer expects `{full_prompt, completion}`. Both keys are present in every row. `tools/sft_agentic_v3.py` reads it directly.

### Trajectory format inside `completion`

```
<rationale: 1-2 sentences>

```action
{"tool": "search", "query": "...", "year_min": 2018, "year_max": 2025, "k": 5}
```

[result]: <1-3 sentence simulated tool result>

... more steps ...

```action
{"tool": "propose", "gene_genome": {<6 fields>}}
```
```

### Tool schemas

```jsonc
search:        {"tool":"search","query":"...","year_min":2018,"year_max":2025,"k":5}
read:          {"tool":"read","paper_id":"oa:Wxxxxx"}
extract_genome:{"tool":"extract_genome","paper_id":"oa:Wxxxxx"}
genome_diff:   {"tool":"genome_diff","parent_id":"oa:Wxxxxx","proposed_genome":{...}}
novelty_check: {"tool":"novelty_check","mechanism":"...","year_min":...,"year_max":...}
propose:       see below
```

### Three `propose` schemas

**gene_genome** (GENE-Arena style, ~60% of demos)

```json
{"tool":"propose","gene_genome":{
  "mechanism_genome":"...","niche_genome":"...","observation_genome":"...",
  "limitation_genome":"...","delta_genome":"...","claim_genome":"..."
}}
```

**idea_plan** (SGI-Bench task_2 style, ~30%)

```json
{"tool":"propose","idea_plan":{
  "Idea":"...","ImplementationSteps":{"1":"...","2":"..."},
  "ImplementationOrder":["1-2","2-3"], "Dataset":"...",
  "EvaluationMetrics":{"metric":"desc"}, "ExpectedOutcome":"..."
}}
```

**free_text_answer** (ArenaRL style, ~10%)

```json
{"tool":"propose","answer":"<3-5 paragraph natural-language proposal>"}
```

### DPO record

```json
{
  "prompt_id": "...", "source_round": "v7"|"v10"|"v13"|"v19"|"v22"|"v27",
  "discipline": "...", "lang": "en"|"zh",
  "rejection_mode": "wrong_schema"|"premature_propose"|"no_evidence"|"schema_collapse"|"truncated"|"verbose_padding"|"made_up_papers",
  "full_prompt": "...", "candidates": [...],
  "chosen": "<trajectory>", "rejected": "<trajectory>"
}
```

### Tool corpus

- `data/genetrace_v0_1/` — 855 GenomeCards + 300 DynamicsEdges (lineage corpus). Used by `extract_genome`, `genome_diff`, `novelty_check` as the local KB. Format: see archived `history/genetrace_format.md`.
- `data/openalex_cache/` — disk cache of OpenAlex search + works queries (excluded from git).

## 6. Design rationale (why the corpus looks this way)

**The motivating empirical finding:** Qwen3-8B agentic-v3-sft has only **18% propose rate** on 150 GENE-Arena prompts (27/150). The v3 SFT data had median 10 tools/demo, 77% long trajectories — the model learned "research = many tool calls" and ran out of `max_turns` before reaching propose. (Detail: `experiment.md` §2.)

**Targeted fixes in v5-v33:**

1. **Length-bias inversion** (v5/v11/v12/v18): forced short trajectories, hit 64% short / 14% med / 22% long in combined corpus.
2. **Schema multiplicity** (v6/v15): trained on all three propose schemas, not just gene_genome.
3. **Chinese coverage** (v6/v9/v12/v15/v21/v28/v29 + cross-lingual v30/v31): from 8% → 21%. Unlocks ArenaRL Open-Travel + Open-DeepResearch.
4. **Failure resilience** (v8 recovery / v16 ambiguous / v18 self-correction): teaches the agent to handle realistic messy inputs.
5. **Bench-specific** (v8 SGI strict + arena_travel + gene_arena_crosscut, v23 math/formal, v24 lineage chains): explicit coverage of the 3 target benchmarks' schemas.
6. **Discipline breadth** (v4/v23/v26/v33 fresh disciplines): 73 unique disciplines now (was 33 in v3 alone).

Older design docs (10 archetypes W1-W10, 8 core skills) are archived in `history/research_workflows.md`. The current corpus is broader than that framework but consistent with it.

## 7. Reproduce / re-merge

```bash
# Merge SFT (rebuild combined file from individual rounds):
python tools/merge_sft_v3_to_v8.py

# Merge DPO:
python tools/merge_dpo_all.py
```

Each individual round was built by `tools/build_v<N>_<name>.py`. Many are similar — the canonical "build a round" template is `tools/build_v5_short_focused.py` (3-phase: synthesize prompts → prefetch OpenAlex → generate demos via GPT-5.5 with strict ```action``` format).
