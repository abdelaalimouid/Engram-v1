"""Qwen Cloud client — PROOF OF ALIBABA CLOUD DEPLOYMENT.

Every model call in Engram goes through this module, which targets the
Alibaba Cloud Model Studio (DashScope) international endpoint:

    https://dashscope-intl.aliyuncs.com/compatible-mode/v1

Models used:
  - qwen3.7-plus        agent reasoning and replies
  - qwen-flash          importance scoring, fact extraction, contradiction
                        adjudication, eval judging (cheap, high volume)
  - text-embedding-v4   memory embeddings (512-dim)
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import numpy as np
from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        if not config.QWEN_API_KEY:
            raise RuntimeError(
                "No Qwen Cloud API key found. Set DASHSCOPE_API_KEY (or put "
                "CLOUD_API_KEY=... in an `env` file at the repo root)."
            )
        _client = OpenAI(api_key=config.QWEN_API_KEY, base_url=config.QWEN_BASE_URL)
    return _client


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Single chat completion against Qwen on Alibaba Cloud."""
    response = client().chat.completions.create(
        model=model or config.CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def chat_json(
    messages: list[dict[str, str]],
    model: str | None = None,
    max_tokens: int = 1024,
    retries: int = 2,
) -> dict[str, Any]:
    """Chat completion that must return a JSON object. Retries on parse failure."""
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        text = chat(
            messages,
            model=model or config.FAST_MODEL,
            temperature=0.0 if attempt == 0 else 0.3,
            max_tokens=max_tokens,
        )
        try:
            return _extract_json(text)
        except (json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            time.sleep(0.5)
    raise ValueError(f"Model did not return valid JSON: {last_error}")


def _extract_json(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply, tolerating code fences and prose."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts with text-embedding-v4 on Alibaba Cloud. Returns unit vectors
    of shape (len(texts), EMBED_DIM), so cosine similarity is a dot product.
    """
    if not texts:
        return np.zeros((0, config.EMBED_DIM), dtype=np.float32)
    # DashScope embedding batches are capped; chunk defensively.
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 10):
        batch = texts[i : i + 10]
        response = client().embeddings.create(
            model=config.EMBED_MODEL, input=batch, dimensions=config.EMBED_DIM
        )
        vectors.extend(item.embedding for item in response.data)
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
