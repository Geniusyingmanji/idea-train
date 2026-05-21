"""Pairwise PES judge for evo-OPD v6 arena-rank reward.

For each (prompt, candidate_A, candidate_B), GPT-5.5 picks a winner on the 5
creative PES sub-dims:
  - originality
  - balanced_novelty
  - mechanism_concreteness
  - limitation_repair
  - expected_impact

Winner per dim ∈ {"A", "B", "tie"}; the overall winner is whoever wins more dims
(ties broken in caller by accumulated per-dim avg sub-score). Implemented async
via gpt55_client.batch_call so a whole tournament's pairs fire in parallel.

Why pairwise (not pointwise): ArenaRL (arXiv 2601.06487) shows pointwise scalar
scoring is unstable on open-ended tasks — judge ratings drift, scales are
arbitrary, policy collapses to safe high-mean mode. Pairwise judgements
("which is more original?") are robust to drift since absolute scale is
irrelevant.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Sequence

from ..teachers.gpt55_client import TeacherCall, build_client, call_one

JUDGE_DIMS = [
    "originality",
    "balanced_novelty",
    "mechanism_concreteness",
    "limitation_repair",
    "expected_impact",
]

DIM_DESCRIPTIONS = {
    "originality": "deviates from obvious extensions of the parent paper",
    "balanced_novelty": "novelty is measured — not gratuitous, not incremental",
    "mechanism_concreteness": "proposed mechanism is specified well enough to implement",
    "limitation_repair": "addresses the parent paper's stated limitation",
    "expected_impact": "if successful, plausibly advances the field",
}

_JUDGE_SYSTEM = (
    "You are an expert reviewer for scientific research proposals. "
    "Given a prompt and two candidate proposals (A and B), you must pick which "
    "is better on each of 5 evaluation dimensions. Output ONLY valid JSON inside "
    "```json ... ``` fences. No commentary outside the fences. "
    "Use 'tie' sparingly — only when truly indistinguishable on that dim."
)


def _build_judge_prompt(prompt: str, idea_a: str, idea_b: str) -> str:
    """Construct the user-message text the judge will see."""
    dims_blob = "\n".join(
        f'  - "{d}": {DIM_DESCRIPTIONS[d]}' for d in JUDGE_DIMS
    )
    return f"""You see one prompt and two candidate research proposals (A and B). For each of 5 dimensions, pick which is better. Then give the overall winner.

DIMENSIONS:
{dims_blob}

PROMPT:
{prompt[:3000]}

CANDIDATE A:
{idea_a[:3500]}

CANDIDATE B:
{idea_b[:3500]}

Output JSON with this exact schema:
```json
{{
  "originality": "A" | "B" | "tie",
  "balanced_novelty": "A" | "B" | "tie",
  "mechanism_concreteness": "A" | "B" | "tie",
  "limitation_repair": "A" | "B" | "tie",
  "expected_impact": "A" | "B" | "tie",
  "overall": "A" | "B" | "tie",
  "brief_reason": "<1-2 sentences>"
}}
```"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_judge_output(text: str) -> dict | None:
    """Extract JSON from the judge's reply; return None on failure."""
    if not text:
        return None
    m = _FENCE_RE.search(text)
    blob = m.group(1) if m else text
    blob = blob.strip()
    # try raw JSON
    try:
        return json.loads(blob)
    except Exception:
        pass
    # try to extract the largest {...} block
    start = blob.find("{")
    end = blob.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(blob[start:end + 1])
        except Exception:
            return None
    return None


@dataclass
class PairwiseResult:
    pair_id: str
    prompt_id: str
    idx_a: int
    idx_b: int
    per_dim: dict[str, str] = field(default_factory=dict)  # dim -> "A"/"B"/"tie"
    overall: str = "tie"                                     # "A" / "B" / "tie"
    a_wins_count: int = 0                                    # count of dims where A wins
    b_wins_count: int = 0
    n_ties: int = 0
    raw_reason: str = ""
    latency_ms: float = 0.0
    error: str | None = None


def _result_from_judge_json(
    pair_id: str, prompt_id: str, idx_a: int, idx_b: int,
    judge_json: dict | None, latency_ms: float, error: str | None,
) -> PairwiseResult:
    if judge_json is None:
        # default to tie on parse failure
        return PairwiseResult(
            pair_id=pair_id, prompt_id=prompt_id, idx_a=idx_a, idx_b=idx_b,
            overall="tie", n_ties=len(JUDGE_DIMS),
            latency_ms=latency_ms,
            error=error or "judge_output_unparseable",
        )
    per_dim = {}
    a_w = b_w = ties = 0
    for d in JUDGE_DIMS:
        v = str(judge_json.get(d, "tie")).strip().lower()
        if v in ("a",):
            per_dim[d] = "A"
            a_w += 1
        elif v in ("b",):
            per_dim[d] = "B"
            b_w += 1
        else:
            per_dim[d] = "tie"
            ties += 1
    overall = str(judge_json.get("overall", "tie")).strip().upper()
    if overall not in ("A", "B", "TIE"):
        # derive from per-dim majority
        overall = "A" if a_w > b_w else ("B" if b_w > a_w else "TIE")
    return PairwiseResult(
        pair_id=pair_id, prompt_id=prompt_id, idx_a=idx_a, idx_b=idx_b,
        per_dim=per_dim, overall=overall.lower() if overall == "TIE" else overall,
        a_wins_count=a_w, b_wins_count=b_w, n_ties=ties,
        raw_reason=str(judge_json.get("brief_reason", ""))[:300],
        latency_ms=latency_ms,
        error=error,
    )


# convert "tie" canonical lowercase
def _norm_overall(s: str) -> str:
    s = s.strip().upper()
    return s if s in ("A", "B") else "tie"


def judge_one_pair(
    prompt: str, idea_a: str, idea_b: str,
    *, pair_id: str, prompt_id: str, idx_a: int, idx_b: int,
    client=None, max_tokens: int = 600, retries: int = 2,
) -> PairwiseResult:
    """Single pairwise judge call. Thread-safe given the underlying client is."""
    if client is None:
        client = build_client()
    user_text = _build_judge_prompt(prompt, idea_a, idea_b)
    call = TeacherCall(
        prompt_id=pair_id,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    res = call_one(client, call, retries=retries)
    parsed = _parse_judge_output(res.content) if res.error is None else None
    out = _result_from_judge_json(
        pair_id=pair_id, prompt_id=prompt_id,
        idx_a=idx_a, idx_b=idx_b,
        judge_json=parsed, latency_ms=res.latency_ms, error=res.error,
    )
    # normalize the canonical case for overall
    if out.overall.upper() in ("A", "B"):
        out.overall = out.overall.upper()
    return out


def judge_pairs_parallel(
    pairs: Sequence[tuple],
    *, client=None, workers: int = 8, retries: int = 2,
    max_tokens: int = 600,
) -> list[PairwiseResult]:
    """Run a batch of pairwise judgements concurrently.

    `pairs` is a sequence of dicts/tuples with keys:
      pair_id, prompt_id, idx_a, idx_b, prompt, idea_a, idea_b
    """
    if client is None:
        client = build_client()

    def _to_kwargs(p):
        if isinstance(p, dict):
            return p
        # tuple: (pair_id, prompt_id, idx_a, idx_b, prompt, idea_a, idea_b)
        keys = ("pair_id", "prompt_id", "idx_a", "idx_b", "prompt", "idea_a", "idea_b")
        return dict(zip(keys, p))

    results: list[PairwiseResult] = [None] * len(pairs)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_i = {}
        for i, p in enumerate(pairs):
            kw = _to_kwargs(p)
            fut = ex.submit(
                judge_one_pair,
                kw["prompt"], kw["idea_a"], kw["idea_b"],
                pair_id=kw["pair_id"], prompt_id=kw["prompt_id"],
                idx_a=kw["idx_a"], idx_b=kw["idx_b"],
                client=client, max_tokens=max_tokens, retries=retries,
            )
            fut_to_i[fut] = i
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                results[i] = fut.result()
            except Exception as e:  # broad
                kw = _to_kwargs(pairs[i])
                results[i] = PairwiseResult(
                    pair_id=kw["pair_id"], prompt_id=kw["prompt_id"],
                    idx_a=kw["idx_a"], idx_b=kw["idx_b"],
                    overall="tie", n_ties=len(JUDGE_DIMS),
                    error=f"future_failed: {type(e).__name__}: {e}",
                )
    return results


if __name__ == "__main__":
    # smoke test: a clearly-better-A pair
    prompt = (
        "Parent paper: 'A simple diffusion model for molecule generation.' "
        "Its stated limitation: no physical validity constraint. "
        "Propose a follow-up paper."
    )
    idea_a = (
        '{"idea": "Physics-Aware Diffusion for Drug Discovery", '
        '"core_method": "Add a physical-validity scoring head to the diffusion '
        'reverse pass that uses a learned energy function from MD trajectories, '
        'rejecting denoising steps that produce non-physical conformations. '
        'Train end-to-end on QM9 + GEOM-Drugs.", '
        '"limitation_addressed": "lack of physical validity in baseline diffusion."}'
    )
    idea_b = (
        '{"idea": "Another Diffusion Model", '
        '"core_method": "Use diffusion to generate molecules.", '
        '"limitation_addressed": "improving molecule generation."}'
    )

    print("=== smoke: 1 pair (A should win clearly) ===")
    res = judge_one_pair(prompt, idea_a, idea_b,
                         pair_id="smoke-0", prompt_id="smoke", idx_a=0, idx_b=1)
    print(f"  per_dim:  {res.per_dim}")
    print(f"  overall:  {res.overall}  (a={res.a_wins_count} b={res.b_wins_count} ties={res.n_ties})")
    print(f"  reason:   {res.raw_reason}")
    print(f"  latency:  {res.latency_ms:.0f}ms  err={res.error}")

    print("\n=== smoke: 5 parallel ===")
    pairs = [
        dict(pair_id=f"s-{i}", prompt_id="smoke", idx_a=0, idx_b=1,
             prompt=prompt, idea_a=idea_a, idea_b=idea_b)
        for i in range(5)
    ]
    import time as _t
    t0 = _t.time()
    out = judge_pairs_parallel(pairs, workers=5)
    print(f"  5 pairs done in {_t.time() - t0:.1f}s")
    for r in out:
        print(f"  {r.pair_id}: overall={r.overall} a={r.a_wins_count} b={r.b_wins_count}")
