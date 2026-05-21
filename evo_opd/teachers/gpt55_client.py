"""GPT-5.5 teacher client for Stage 1 SFT data generation.

Wraps IdeaEvolving's UnifiedLLMClient with:
  - concurrent calling (ThreadPoolExecutor)
  - retry on transient failures
  - per-call logging to JSONL
  - simple rate-limit awareness (semaphore)

Reads Azure keyless config from environment / config defaults.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

# Add IdeaEvolving to path so we can import the unified client
sys.path.insert(0, "/home/azureuser/workspace-gzy/zyf/IdeaEvolving")
os.environ.setdefault("no_proxy", "*")
os.environ.setdefault("NO_PROXY", "*")

from agent.llm_client import LLMClientConfig, UnifiedLLMClient  # type: ignore

DEFAULT_CONFIG = dict(
    provider="azure",
    model="gpt-5.5",
    azure_endpoint="https://t2vgoaigpt4o3.openai.azure.com/",
    azure_auth_mode="azure_cli",
    api_version="2024-12-01-preview",
)


@dataclass
class TeacherCall:
    prompt_id: str               # caller-supplied stable id
    messages: list[dict]
    max_tokens: int = 4096
    temperature: float | None = None
    metadata: dict | None = None  # extras to carry through (paper id, task type)


@dataclass
class TeacherResult:
    prompt_id: str
    content: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    finish_reason: str | None
    error: str | None = None
    metadata: dict | None = None


def build_client(**overrides) -> UnifiedLLMClient:
    cfg_kwargs = {**DEFAULT_CONFIG, **overrides}
    return UnifiedLLMClient(LLMClientConfig(**cfg_kwargs))


def call_one(client: UnifiedLLMClient, call: TeacherCall, retries: int = 2) -> TeacherResult:
    last_err: str | None = None
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            res = client.chat(
                call.messages,
                max_tokens=call.max_tokens,
                temperature=call.temperature,
            )
            return TeacherResult(
                prompt_id=call.prompt_id,
                content=res.content,
                input_tokens=res.input_tokens,
                output_tokens=res.output_tokens,
                latency_ms=res.latency_ms,
                finish_reason=res.finish_reason,
                metadata=call.metadata,
            )
        except Exception as e:  # broad: capture all transient + permanent
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < retries:
                time.sleep(2 ** attempt + 1)
            else:
                return TeacherResult(
                    prompt_id=call.prompt_id,
                    content="",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=(time.time() - t0) * 1000 if 't0' in dir() else 0,
                    finish_reason="error",
                    error=last_err,
                    metadata=call.metadata,
                )
    return TeacherResult(prompt_id=call.prompt_id, content="", input_tokens=0, output_tokens=0,
                          latency_ms=0, finish_reason="error", error=last_err)


_LOG_LOCK = threading.Lock()


def batch_call(
    calls: list[TeacherCall],
    workers: int = 16,
    log_path: Path | str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    client: UnifiedLLMClient | None = None,
) -> list[TeacherResult]:
    """Run a batch concurrently. Streams results to log_path if given."""
    if client is None:
        client = build_client()
    log_fp = None
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fp = log_path.open("a")
    results: list[TeacherResult] = []
    n_total = len(calls)
    n_done = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(call_one, client, c): c for c in calls}
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                n_done += 1
                if log_fp is not None:
                    with _LOG_LOCK:
                        log_fp.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                        log_fp.flush()
                if on_progress is not None and n_done % 25 == 0:
                    on_progress(n_done, n_total)
    finally:
        if log_fp is not None:
            log_fp.close()
    return results


if __name__ == "__main__":
    # smoke
    client = build_client()
    calls = [TeacherCall(prompt_id=f"smoke-{i}",
                          messages=[{"role": "user", "content": f"Reply only with: ok-{i}"}],
                          max_tokens=32)
             for i in range(5)]
    t0 = time.time()
    results = batch_call(calls, workers=5)
    el = time.time() - t0
    print(f"=== batch_call smoke: 5 prompts in {el:.1f}s ===")
    for r in results:
        print(f"  {r.prompt_id}: latency={r.latency_ms:.0f}ms in={r.input_tokens} out={r.output_tokens} "
              f"content={r.content[:30]!r} err={r.error}")
