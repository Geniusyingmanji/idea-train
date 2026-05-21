"""GPT-5.5 teacher prompt templates for Stage 1 SFT data generation.

Five canonical templates, each producing a strictly-JSON answer that our
parser/verifier can adjudicate. All templates instruct the teacher to:
  1. Quote evidence verbatim from the source paper.
  2. Emit one fenced ```json``` block at the end.
"""
from __future__ import annotations

from textwrap import dedent

SYSTEM_PROMPT = dedent("""\
    You are a meticulous scientific lineage analyst.

    Output rules:
      1. Quote any evidence verbatim from the source text (no paraphrasing).
      2. Always end your response with exactly ONE ```json``` code fence containing the structured answer.
      3. The JSON must match the schema in the user prompt EXACTLY — no extra keys, no missing keys.
      4. If a field is genuinely unknown from the source, set it to an empty string "" or [].
""")


# ---------------------------------------------------------------------------
# 1. gene_card_extract  — paper text → 6-field gene card with evidence
# ---------------------------------------------------------------------------
GENE_CARD_EXTRACT = dedent("""\
    Read the following paper text and extract a 6-field IdeaGenome card.

    PAPER TEXT:
    ---
    {paper_text}
    ---

    Required JSON schema (no other keys allowed):
    {{
      "mechanism_genome":   "the core technical approach (≤ 40 words). The primary heritable gene.",
      "niche_genome":       "the precise problem/task/domain targeted — WHAT, not HOW (≤ 40 words).",
      "observation_genome": "key empirical finding or insight (≤ 40 words).",
      "limitation_genome":  "known weakness or future-work pressure (≤ 40 words).",
      "delta_genome":       "what specifically changed vs the closest predecessor (≤ 40 words).",
      "claim_genome":       "the primary falsifiable assertion in ONE sentence.",
      "evidence_quotes": [
        {{"field": "mechanism_genome", "quote": "VERBATIM span from the paper text supporting this field"}},
        {{"field": "observation_genome", "quote": "VERBATIM span from the paper text"}}
      ]
    }}

    Provide at least 2 evidence quotes; every quote MUST appear character-for-character in the paper text above.
""")


# ---------------------------------------------------------------------------
# 2. genome_diff_annotate — (parent, child) → fates + driver + dynamics
# ---------------------------------------------------------------------------
GENOME_DIFF_ANNOTATE = dedent("""\
    Compare a PARENT paper genome to a CHILD paper genome.

    PARENT GENE-CARD:
    {parent_card_json}

    CHILD GENE-CARD:
    {child_card_json}

    Tasks:
    (a) For each of the 6 gene fields, label the FATE of the parent's content as the child treats it:
        - INHERITED  : child's field is essentially the same as parent's
        - MUTATED    : child's field is recognizably derived from parent's but altered
        - LOST       : parent had content; child does not address this field
        - NOVEL      : parent did not have content; child introduces new content
        - HYBRIDIZED : child's field combines content from parent AND another external lineage
    (b) Identify the PRIMARY DRIVER of this transition: one of {{mechanism, niche, observation, limitation}}.
    (c) Classify the evolutionary DYNAMICS via the decision tree (top-to-bottom, first match wins):
        - Hybridization        : ≥2 genes are HYBRIDIZED (from a distinct external lineage)
        - Speciation           : all mechanism genes LOST + NOVEL replacements + niche INHERITED/MUTATED
        - Adaptive Radiation   : mechanism INHERITED/MUTATED + niche MUTATED/LOST/NOVEL (domain shift)
        - Niche Competition    : zero genes INHERITED or MUTATED (parallel competition)
        - Mutation             : default (most genes inherited/mutated, same niche)

    Required JSON schema:
    {{
      "fates": {{
        "mechanism_genome": "INHERITED | MUTATED | LOST | NOVEL | HYBRIDIZED",
        "niche_genome":     "...",
        "observation_genome": "...",
        "limitation_genome": "...",
        "delta_genome":     "...",
        "claim_genome":     "..."
      }},
      "driver":    "mechanism | niche | observation | limitation",
      "dynamics":  "Mutation | Adaptive Radiation | Hybridization | Speciation | Niche Competition",
      "rationale": "ONE sentence justifying driver + dynamics."
    }}
""")


# ---------------------------------------------------------------------------
# 3. lineage_trace_reconstruct — shuffled cards → ordering + per-edge dynamics
# ---------------------------------------------------------------------------
LINEAGE_TRACE_RECONSTRUCT = dedent("""\
    Below are N gene-cards from a single research lineage, presented in RANDOM order.
    Reconstruct the chronological lineage (oldest predecessor first → newest descendant last)
    and annotate each parent→child edge with the dominant evolutionary dynamics.

    SHUFFLED CARDS (labeled 1..N):
    {shuffled_cards_text}

    Required JSON schema:
    {{
      "ordering": [<1-indexed card numbers in chronological order>],
      "per_edge_dynamics": [
        "Mutation | Adaptive Radiation | Hybridization | Speciation | Niche Competition",
        ... (length = len(ordering) - 1)
      ],
      "rationale": "ONE short paragraph explaining the lineage logic."
    }}
""")


# ---------------------------------------------------------------------------
# 4. lineage_verify — proposed lineage → valid? if not, what's wrong + repair
# ---------------------------------------------------------------------------
LINEAGE_VERIFY = dedent("""\
    A user has proposed the following research lineage. Verify whether it is internally coherent
    AND scientifically grounded.

    PROPOSED LINEAGE:
    {proposed_lineage_text}

    Possible failure modes (pick one if the lineage is invalid):
      - intruder           : one paper does not belong in this lineage
      - wrong_step         : two adjacent papers cannot be in a parent→child relation
      - missing_link       : an obvious intermediate predecessor is missing
      - citation_conflict  : the cited dynamics label contradicts the gene-fate evidence
      - valid              : the lineage is correct as-is

    Required JSON schema:
    {{
      "valid":            true | false,
      "failure_mode":     "intruder | wrong_step | missing_link | citation_conflict | valid",
      "specific_defect":  "ONE sentence pointing to the exact paper(s) or edge that fails, OR empty string if valid",
      "repair":           "ONE sentence describing how to fix the lineage, OR empty string if valid"
    }}
""")


# ---------------------------------------------------------------------------
# 5. idea_generate — trace + open question → structured proposal
# ---------------------------------------------------------------------------
IDEA_GENERATE = dedent("""\
    Given an established research lineage and a frontier OPEN QUESTION, propose a NEW research idea
    that would coherently extend the lineage.

    LINEAGE (chronological gene-cards):
    {lineage_text}

    OPEN QUESTION:
    {open_question}

    Your proposal must (a) inherit at least one mechanism/niche from a named parent paper above,
    (b) repair one declared limitation, and (c) make a falsifiable claim.

    Required JSON schema:
    {{
      "name": "short memorable name (3-5 words)",
      "problem": "what specific frontier problem this addresses (≤ 60 words)",
      "mechanism": "the proposed approach (≤ 80 words)",
      "expected_contribution": "method | dataset | theory | benchmark | system",
      "lineage_connection": {{
        "parents": ["paper title 1", "paper title 2"],
        "inherits": "what mechanism/insight is carried over (≤ 30 words)",
        "repairs_limitation": "which parent's limitation is repaired and how (≤ 30 words)",
        "dynamics": "Mutation | Adaptive Radiation | Hybridization | Speciation"
      }},
      "evaluation_plan": "ONE paragraph describing how to test the claim experimentally."
    }}
""")


TEMPLATES = {
    "gene_card_extract":         GENE_CARD_EXTRACT,
    "genome_diff_annotate":      GENOME_DIFF_ANNOTATE,
    "lineage_trace_reconstruct": LINEAGE_TRACE_RECONSTRUCT,
    "lineage_verify":            LINEAGE_VERIFY,
    "idea_generate":             IDEA_GENERATE,
}


def build_messages(task_type: str, **kwargs) -> list[dict]:
    """Build a [system, user] messages list for the given task type."""
    if task_type not in TEMPLATES:
        raise KeyError(f"Unknown task_type {task_type!r}; known: {list(TEMPLATES)}")
    user = TEMPLATES[task_type].format(**kwargs)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
