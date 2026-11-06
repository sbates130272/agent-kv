"""Async request execution against an OpenAI-compatible completions endpoint."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

from kvsweep.metrics import RequestResult


@dataclass
class RequestSpec:
    book_id: int
    title: str
    chunk_idx: int
    question_idx: int
    prompt: str


async def _fire(
    client: AsyncOpenAI,
    model: str,
    spec: RequestSpec,
    max_tokens: int,
    sem: asyncio.Semaphore,
    round_idx: int,
    round_label: str,
) -> RequestResult:
    async with sem:
        start = time.monotonic()
        first_token_t: float | None = None
        response_text = ""
        error: str | None = None

        try:
            stream = await client.completions.create(
                model=model,
                prompt=spec.prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                stream=True,
            )
            async for chunk in stream:
                if first_token_t is None and chunk.choices:
                    first_token_t = time.monotonic()
                if chunk.choices:
                    response_text += chunk.choices[0].text or ""
        except Exception as exc:
            error = str(exc)
            if first_token_t is None:
                first_token_t = time.monotonic()

        end = time.monotonic()
        ttft = (first_token_t - start) if first_token_t else (end - start)
        e2e = end - start

        prompt_tokens = len(spec.prompt.split())
        output_tokens = len(response_text.split())
        tpot = (e2e - ttft) / (output_tokens - 1) if output_tokens >= 2 else None

        return RequestResult(
            round_idx=round_idx,
            round_label=round_label,
            book_id=spec.book_id,
            title=spec.title,
            chunk_idx=spec.chunk_idx,
            question_idx=spec.question_idx,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            ttft_s=ttft,
            e2e_s=e2e,
            tpot_s=tpot,
            error=error,
        )


async def run_round(
    client: AsyncOpenAI,
    model: str,
    specs: list[RequestSpec],
    max_tokens: int,
    max_inflight: int,
    round_idx: int,
    round_label: str,
    progress_callback=None,
) -> list[RequestResult]:
    """Fire all specs concurrently up to max_inflight and collect results."""
    sem = asyncio.Semaphore(max_inflight)

    async def _tracked(spec: RequestSpec) -> RequestResult:
        result = await _fire(client, model, spec, max_tokens, sem, round_idx, round_label)
        if progress_callback:
            progress_callback(result)
        return result

    return list(await asyncio.gather(*[_tracked(s) for s in specs]))
