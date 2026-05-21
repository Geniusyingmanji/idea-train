"""Pointwise PES judge — the v5-style absolute rating, used ONLY for the
v6-pointwise ablation to isolate "tournament structure" from "judge identity".

For each (prompt, idea), GPT-5.5 rates the idea 1-5 on each of the 5 creative
PES sub-dims, then averages. Output normalized to [0, 1] for compatibility with
the GRPO reward composition.

This is the *baseline* against which v6's tournament-rank is compared. If
v6-pointwise still beats v3 SFT on PES, then the tournament structure isn't
essential — judge quality is. If only v6 (with tournament) beats v3, then
ArenaRL's central thesis ports to scientific idea generation, and the
tournament IS load-bearing.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Sequence

from ..teachers.gpt55_client import TeacherCall, build_client, call_one
from .pairwise_pes import JUDGE_DIMS, DIM_DESCRIPTIONS, _FENCE_RE

_JUDGE_SYSTEM = (
    "You are an expert reviewer for scientific research proposals. "
    "Given a prompt and one candidate proposal, rate the proposal on 5 dimensions "
    "(1 = very poor, 5 = excellent). Output ONLY valid JSON inside ```json ... ``` "
    "fences. No commentary outside the fences. Use the full 1-5 range — don't "
    "default to 3."
)


def _build_judge_prompt(prompt: str, idea: str) -> str:
    dims_blob = "\n".join(
        f'  - "{d}" (1-5): {DIM_DESCRIPTIONS[d]}' for d in JUDGE_DIMS
    )
    return f"""You see one prompt and one candidate research proposal. Rate the proposal on 5 dimensions, each 1-5 (use the full range).

DIMENSIONS:
{dims_blob}

PROMPT:
{prompt[:3000]}

CANDIDATE:
{idea[:3500]}

Output JSON with this exact schema:
```json
{{
  "originality": 1 | 2 | 3 | 4 | 5,
  "balanced_novelty": 1 | 2 | 3 | 4 | 5,
  "mechanism_concreteness": 1 | 2 | 3 | 4 | 5,
  "limitation_repair": 1 | 2 | 3 | 4 | 5,
  "expected_impact": 1 | 2 | 3 | 4 | 5,
  "brief_reason": "<1-2 sentences>"
}}
```"""


def _parse_judge_output(text: str) -> dict | None:
    if not text:
        return None
    m = _FENCE_RE.search(text)
    blob = m.group(1) if m else text
    blob = blob.strip()
    try:
        return json.loads(blob)
    except Exception:
        pass
    start = blob.find("{")
    end = blob.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(blob[start:end + 1])
        except Exception:
            return None
    return None


@dataclass
class PointwiseResult:
    judge_id: str
    prompt_id: str
    idx: int
    scores: dict[str, float] = field(default_factory=dict)  # dim -> 1..5
    mean_score: float = 3.0     # avg of the 5 dims
    normalized: float = 0.5     # (mean_score - 1) / 4 ∈ [0, 1]
    raw_reason: str = ""
    latency_ms: float = 0.0
    error: str | None = None


def judge_one(
    prompt: str, idea: str,
    *, judge_id: str, prompt_id: str, idx: int,
    client=None, max_tokens: int = 400, retries: int = 2,
) -> PointwiseResult:
    if client is None:
        client = build_client()
    user_text = _build_judge_prompt(prompt, idea)
    call = TeacherCall(
        prompt_id=judge_id,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_text},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    res = call_one(client, call, retries=retries)
    parsed = _parse_judge_output(res.content) if res.error is None else None
    if parsed is None:
        return PointwiseResult(
            judge_id=judge_id, prompt_id=prompt_id, idx=idx,
            scores={d: 3.0 for d in JUDGE_DIMS},
            mean_score=3.0, normalized=0.5,
            latency_ms=res.latency_ms,
            error=res.error or "judge_output_unparseable",
        )
    scores = {}
    vals = []
    for d in JUDGE_DIMS:
        v = parsed.get(d, 3)
        try:
            v = float(v)
        except Exception:
            v = 3.0
        v = max(1.0, min(5.0, v))
        scores[d] = v
        vals.append(v)
    mean = sum(vals) / len(vals)
    return PointwiseResult(
        judge_id=judge_id, prompt_id=prompt_id, idx=idx,
        scores=scores, mean_score=mean,
        normalized=(mean - 1.0) / 4.0,
        raw_reason=str(parsed.get("brief_reason", ""))[:300],
        latency_ms=res.latency_ms,
    )


def judge_batch_parallel(
    items: Sequence[dict],
    *, client=None, workers: int = 8, retries: int = 2,
    max_tokens: int = 400,
) -> list[PointwiseResult]:
    """`items` is a sequence of dicts with keys: judge_id, prompt_id, idx, prompt, idea."""
    if client is None:
        client = build_client()
    results: list[PointwiseResult] = [None] * len(items)  # type: ignore
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_to_i = {}
        for i, kw in enumerate(items):
            fut = ex.submit(
                judge_one, kw["prompt"], kw["idea"],
                judge_id=kw["judge_id"], prompt_id=kw["prompt_id"], idx=kw["idx"],
                client=client, max_tokens=max_tokens, retries=retries,
            )
            fut_to_i[fut] = i
        for fut in as_completed(fut_to_i):
            i = fut_to_i[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                kw = items[i]
                results[i] = PointwiseResult(
                    judge_id=kw["judge_id"], prompt_id=kw["prompt_id"], idx=kw["idx"],
                    scores={d: 3.0 for d in JUDGE_DIMS},
                    mean_score=3.0, normalized=0.5,
                    error=f"future_failed: {type(e).__name__}: {e}",
                )
    return results


def score_group_pointwise(
    prompt: str, candidates: Sequence[str],
    *, prompt_id: str, client=None, workers: int = 8,
) -> dict:
    """Score K candidates pointwise; return same shape as tournament output for
    drop-in use in the reward composition.

    Returns: dict with keys
      - scores: list[float] each in [0,1] (normalized)
      - z_advantage: z-normalized within group (same as tournament path)
      - n_judge_calls: K (one per candidate; cheaper than tournament's 2(K-1))
    """
    K = len(candidates)
    items = [
        dict(judge_id=f"{prompt_id}::pointwise::{i}",
             prompt_id=prompt_id, idx=i,
             prompt=prompt, idea=c)
        for i, c in enumerate(candidates)
    ]
    results = judge_batch_parallel(items, client=client, workers=workers)
    scores = [r.normalized for r in results]
    mu = sum(scores) / K if K > 0 else 0.0
    var = sum((s - mu) ** 2 for s in scores) / max(K, 1)
    sigma = max(var ** 0.5, 1e-6)
    z = [(s - mu) / sigma for s in scores]
    return {
        "scores": scores,
        "z_advantage": z,
        "n_judge_calls": K,
        "raw_results": results,
    }


if __name__ == "__main__":
    import time
    prompt = (
        "Parent paper: 'A simple diffusion model for molecule generation.' "
        "Limitation: no physical validity. Propose a follow-up."
    )
    candidates = [
        '{"idea":"Physics-Aware Diffusion","core_method":"learned MD energy head","limitation_addressed":"physical validity"}',
        '{"idea":"Another molecule generator","core_method":"use diffusion","limitation_addressed":"generation"}',
    ]
    t0 = time.time()
    out = score_group_pointwise(prompt, candidates, prompt_id="smoke-pointwise", workers=2)
    print(f"K={len(candidates)} done in {time.time() - t0:.1f}s")
    for i, (s, z, r) in enumerate(zip(out["scores"], out["z_advantage"], out["raw_results"])):
        print(f"  [{i}] norm={s:.3f}  z={z:+.2f}  scores={r.scores}")
