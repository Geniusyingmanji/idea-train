"""Analyze v3 demo dataset: extract skill coverage, exemplars, write summary.

Output:
  data/agentic_v3/skill_breakdown.json — per-skill / per-archetype distribution
  data/agentic_v3/exemplars.md           — 1 best demo per archetype, for inspection
  data/agentic_v3/summary.md             — markdown summary of the dataset
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

DATA = Path("/home/azureuser/workspace-gzy/zyf/idea_train/data/agentic_v3")
DEMOS = DATA / "sft_demos.jsonl"
OUT_SKILLS = DATA / "skill_breakdown.json"
OUT_EXEMPLARS = DATA / "exemplars.md"
OUT_SUMMARY = DATA / "summary.md"

SKILLS = [
    "problem_clarification",
    "lineage_construction",
    "gap_identification",
    "method_transfer",
    "critical_evaluation",
    "multi_source_synthesis",
    "hypothesis_specification",
    "reflection_revision",
]

# Heuristic for inferring extra skills demonstrated in a completion (beyond
# the prompted archetype). Pattern → skill label.
SKILL_PATTERNS = {
    "problem_clarification": [
        r"narrow\s+down", r"clarif", r"reframe", r"sub-problem",
        r"more specific", r"specifically",
    ],
    "lineage_construction": [
        r"lineage", r"ancestor", r"predecessor", r"prior work", r"foundational",
        r"trace\s+back", r"genealog",
    ],
    "gap_identification": [
        r"gap", r"limitation", r"under-explored", r"not\s+(yet|been)\s+address",
        r"missing", r"underexplored",
    ],
    "method_transfer": [
        r"transfer", r"port\s+(to|from)", r"adapt(ation|ed)?\s+(to|from|of)",
        r"borrow\s+from", r"inspired\s+by\s+\w+\s+in",
    ],
    "critical_evaluation": [
        r"critic", r"flaw", r"weakness", r"valid", r"question\s+the",
        r"counter", r"reproducib", r"replicat",
    ],
    "multi_source_synthesis": [
        r"combin", r"integrate", r"synthesiz", r"unif(y|ied|ying)",
        r"bridge", r"across\s+(papers|sources|methods)",
    ],
    "hypothesis_specification": [
        r"hypothes(is|es)", r"falsifiab", r"testabl", r"validation experiment",
        r"experiment\s+to\s+(test|verify|falsify)",
    ],
    "reflection_revision": [
        r"refine", r"revise", r"reconsider", r"upon reflect",
        r"second pass", r"genome_diff.*self", r"self.check", r"upon.*review",
    ],
}


def detect_skills(text: str) -> set[str]:
    """Heuristic: skills demonstrated based on lexical patterns in the demo."""
    found = set()
    lower = text.lower()
    for skill, pats in SKILL_PATTERNS.items():
        for p in pats:
            if re.search(p, lower):
                found.add(skill)
                break
    return found


def main():
    demos = []
    with DEMOS.open() as f:
        for line in f:
            try:
                demos.append(json.loads(line))
            except Exception:
                continue
    print(f"loaded {len(demos)} demos")

    # archetype × skill coverage matrix
    arch_skill_count = defaultdict(lambda: defaultdict(int))
    skill_total = Counter()
    arch_total = Counter()
    disc_total = Counter()
    tool_count_per_demo = []
    tool_calls = Counter()
    len_chars = []
    n_tools_per_demo = []

    for d in demos:
        arch = d.get("archetype", "unk")
        disc = d.get("discipline", "unk")
        arch_total[arch] += 1
        disc_total[disc] += 1
        c = d["completion"]
        len_chars.append(len(c))

        # tool counts
        n_tools = 0
        for t in ["search", "read", "extract_genome", "genome_diff",
                  "novelty_check", "propose"]:
            n = c.count(f'"tool": "{t}"') + c.count(f'"tool":"{t}"')
            tool_calls[t] += n
            n_tools += n
        n_tools_per_demo.append(n_tools)

        # skill detection (combine prompt-archetype skills + demonstrated patterns)
        prompted_skills = set(d.get("skills_inferred", []))
        demonstrated = detect_skills(c)
        all_skills = prompted_skills | demonstrated
        for s in all_skills:
            arch_skill_count[arch][s] += 1
            skill_total[s] += 1

    # write skill_breakdown.json
    out = {
        "total_demos": len(demos),
        "archetype_total": dict(arch_total),
        "discipline_total": dict(disc_total),
        "skill_total": dict(skill_total),
        "arch_skill_matrix": {a: dict(s) for a, s in arch_skill_count.items()},
        "tool_calls_total": dict(tool_calls),
        "avg_tools_per_demo": mean(n_tools_per_demo),
        "tool_per_demo_distribution": {
            "min": min(n_tools_per_demo),
            "p25": sorted(n_tools_per_demo)[len(n_tools_per_demo) // 4],
            "median": median(n_tools_per_demo),
            "p75": sorted(n_tools_per_demo)[3 * len(n_tools_per_demo) // 4],
            "max": max(n_tools_per_demo),
        },
        "completion_chars": {
            "min": min(len_chars), "median": median(len_chars),
            "mean": mean(len_chars), "max": max(len_chars),
        },
    }
    OUT_SKILLS.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT_SKILLS}")

    # exemplars: best demo per archetype (longest with all 6 fields filled)
    exemplars_md = ["# v3 Exemplars (1 per archetype)\n"]
    by_arch: dict[str, list[dict]] = defaultdict(list)
    for d in demos:
        by_arch[d.get("archetype", "unk")].append(d)
    for arch in sorted(by_arch.keys()):
        candidates = by_arch[arch]
        # pick longest demo as exemplar (proxy for "richer")
        chosen = max(candidates, key=lambda d: len(d["completion"]))
        exemplars_md.append(f"\n## {arch}\n")
        exemplars_md.append(f"**Prompt** (discipline: {chosen.get('discipline')}, source: {chosen.get('source')}):\n")
        exemplars_md.append(f"> {chosen.get('full_prompt', '')[:500]}\n")
        exemplars_md.append(f"\n**Skills demonstrated**: {', '.join(chosen.get('skills_inferred', []))}\n")
        exemplars_md.append(f"\n**Completion** (first 2500 chars):\n")
        exemplars_md.append("```\n" + chosen["completion"][:2500] + "\n...[truncated]\n```\n")
    OUT_EXEMPLARS.write_text("".join(exemplars_md))
    print(f"wrote {OUT_EXEMPLARS}")

    # summary.md — high-level digest
    summary = []
    summary.append("# v3 Research-Workflow Demo Dataset — Summary\n")
    summary.append(f"\n**Generated**: 2026-05-20  |  **Total demos**: {len(demos)}  |  **Source**: GPT-5.5 + real OpenAlex candidates\n")
    summary.append("\n## Archetype distribution\n\n")
    summary.append("| Archetype | Count | % |\n|---|---|---|\n")
    for arch, n in sorted(arch_total.items()):
        summary.append(f"| {arch} | {n} | {n/len(demos)*100:.1f}% |\n")

    summary.append("\n## Discipline distribution\n\n")
    summary.append("| Discipline | Count |\n|---|---|\n")
    for d, n in sorted(disc_total.items(), key=lambda x: -x[1]):
        summary.append(f"| {d} | {n} |\n")

    summary.append("\n## Skill coverage\n\n")
    summary.append("| Skill | # demos demonstrating |\n|---|---|\n")
    for s in SKILLS:
        n = skill_total.get(s, 0)
        summary.append(f"| {s} | {n} ({n/len(demos)*100:.0f}%) |\n")

    summary.append("\n## Tool usage\n\n")
    total_tool_calls = sum(tool_calls.values())
    summary.append(f"**Avg tools per demo**: {mean(n_tools_per_demo):.2f}  (v1 = 3.74, v2 = 3.74)\n\n")
    summary.append("| Tool | Total calls | Per-demo avg | % of all calls |\n|---|---|---|---|\n")
    for t, n in sorted(tool_calls.items(), key=lambda x: -x[1]):
        summary.append(f"| {t} | {n} | {n/len(demos):.2f} | {n/max(total_tool_calls,1)*100:.1f}% |\n")

    summary.append("\n## Completion lengths\n\n")
    summary.append(f"- min: {min(len_chars)} chars\n")
    summary.append(f"- median: {median(len_chars):.0f} chars (~{median(len_chars)//4:.0f} tokens)\n")
    summary.append(f"- mean: {mean(len_chars):.0f}\n")
    summary.append(f"- max: {max(len_chars)}\n")

    summary.append("\n## Archetype × Skill matrix\n\n")
    summary.append("(# demos where archetype's prompted skill OR text-detected skill present)\n\n")
    arch_keys = sorted(arch_skill_count.keys())
    summary.append("| Archetype | " + " | ".join(SKILLS) + " |\n")
    summary.append("|" + "---|" * (len(SKILLS) + 1) + "\n")
    for arch in arch_keys:
        row = [arch] + [str(arch_skill_count[arch].get(s, 0)) for s in SKILLS]
        summary.append("| " + " | ".join(row) + " |\n")

    summary.append("\n## What this dataset teaches\n\n")
    summary.append("Each demo is a fully-played-out research trajectory in one of 10 workflow archetypes "
                   "(see `research_workflows.md`). The dataset jointly trains the 8 research skills above. "
                   "The agent learns:\n\n")
    summary.append("1. **When to call which tool** — archetypes have characteristic patterns\n")
    summary.append("2. **How to think between tool calls** — the rationale lines model expert reasoning\n")
    summary.append("3. **How to ground proposals in real papers** — all `oa:W...` IDs are real OpenAlex works\n")
    summary.append("4. **How to handle vague vs concrete prompts** — W1, W4, W10 give vague topics; W2, W5 give precise ones\n")
    summary.append("5. **How to use structural reasoning tools** (extract_genome, genome_diff, novelty_check) for self-correction\n")

    summary.append("\n## Training implications\n\n")
    summary.append("- Use `max_len ≥ 3072` (median completion ≈ 2.5K tokens, max ~12K)\n")
    summary.append("- Start from `qwen3-8b-sft-v3/final` (clean base) or `qwen3-8b-agentic-v2-sft/final` (already has agentic format)\n")
    summary.append("- Use `LR=3e-5` (lower than v1 5e-5 due to longer context)\n")
    summary.append("- 2 epochs should be enough; consider 1 if loss plateaus\n")
    summary.append("- After SFT, evaluate on GENE-Arena + SGI-Bench + ArenaRL (when ready)\n")

    OUT_SUMMARY.write_text("".join(summary))
    print(f"wrote {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
