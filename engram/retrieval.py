"""Recall: hybrid scoring + token-budget packing.

Score for each candidate memory:

    relevance = 0.60 * cosine_similarity
              + 0.20 * importance
              + 0.20 * recency_prior
    score     = relevance * (0.15 + 0.85 * retention)

Retention gates the score multiplicatively rather than additively: a faded
memory can still surface on a very strong semantic match (recognition beats
free recall, as in humans), but weak matches on faded memories drop out.

Selected memories are then packed under a hard token budget with a greedy
density heuristic (score per token), the direct answer to the track brief's
"recalling critical memories within limited context windows".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import forgetting
from .store import Belief, Episode, MemoryStore

W_SIMILARITY = 0.60
W_IMPORTANCE = 0.20
W_RECENCY = 0.20
RETENTION_FLOOR = 0.15  # a fully-faded memory keeps 15% of its relevance


@dataclass
class RecalledMemory:
    kind: str            # "belief" | "episode" | "summary"
    id: str
    text: str
    score: float
    similarity: float
    retention: float
    importance: float
    tokens: int
    provenance: str | None = None
    stability_before: float = 0.0
    stability_after: float = 0.0


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token), good enough for budgeting."""
    return max(1, len(text) // 4)


def _score(similarity: float, importance: float, recency: float, retention: float) -> float:
    relevance = W_SIMILARITY * similarity + W_IMPORTANCE * importance + W_RECENCY * recency
    return relevance * (RETENTION_FLOOR + (1.0 - RETENTION_FLOOR) * retention)


def score_episodes(
    query_vec: np.ndarray, episodes: list[Episode], now: float
) -> list[RecalledMemory]:
    candidates = [e for e in episodes if e.embedding is not None]
    if not candidates:
        return []
    matrix = np.stack([e.embedding for e in candidates])
    similarities = matrix @ query_vec
    recalled = []
    for episode, similarity in zip(candidates, similarities):
        retention = forgetting.retention(now, episode.last_access, episode.stability)
        recency = forgetting.recency_score(now, episode.created_at)
        text = (
            f"[{episode.role}] {episode.content}"
            if episode.kind == "episode"
            else f"[memory summary] {episode.content}"
        )
        recalled.append(
            RecalledMemory(
                kind="summary" if episode.kind == "summary" else "episode",
                id=episode.id,
                text=text,
                score=_score(float(similarity), episode.importance, recency, retention),
                similarity=float(similarity),
                retention=retention,
                importance=episode.importance,
                tokens=approx_tokens(text),
                stability_before=episode.stability,
            )
        )
    return recalled


def score_beliefs(
    query_vec: np.ndarray, beliefs: list[Belief], now: float
) -> list[RecalledMemory]:
    candidates = [b for b in beliefs if b.embedding is not None]
    if not candidates:
        return []
    matrix = np.stack([b.embedding for b in candidates])
    similarities = matrix @ query_vec
    recalled = []
    for belief, similarity in zip(candidates, similarities):
        retention = forgetting.retention(now, belief.last_access, belief.stability)
        recency = forgetting.recency_score(now, belief.created_at)
        text = belief.statement()
        recalled.append(
            RecalledMemory(
                kind="belief",
                id=belief.id,
                text=text,
                # Beliefs are distilled knowledge: confidence stands in for importance.
                score=_score(float(similarity), belief.confidence, recency, retention),
                similarity=float(similarity),
                retention=retention,
                importance=belief.confidence,
                tokens=approx_tokens(text),
                provenance=belief.source_episode,
                stability_before=belief.stability,
            )
        )
    return recalled


def pack_budget(
    candidates: list[RecalledMemory],
    token_budget: int,
    min_score: float = 0.08,
) -> list[RecalledMemory]:
    """Greedy knapsack by score density (score per token). If nothing fits the
    budget outright, fall back to the single best memory, truncated to fit;
    the agent should never go in blind when a strong memory exists."""
    viable = sorted(
        (c for c in candidates if c.score >= min_score),
        key=lambda c: c.score / c.tokens,
        reverse=True,
    )
    if not viable:
        return []

    packed: list[RecalledMemory] = []
    used = 0
    for candidate in viable:
        if used + candidate.tokens > token_budget:
            continue
        packed.append(candidate)
        used += candidate.tokens

    if not packed:
        best = max(viable, key=lambda c: c.score)
        best.text = best.text[: token_budget * 4]
        best.tokens = approx_tokens(best.text)
        packed = [best]

    packed.sort(key=lambda c: c.score, reverse=True)
    return packed


def recall(
    store: MemoryStore,
    query_vec: np.ndarray,
    token_budget: int,
    exclude_session: str | None = None,
) -> list[RecalledMemory]:
    """Full recall pass: score beliefs + episodes, pack under budget, and
    reinforce everything that made the cut (recall strengthens memory)."""
    now = store.now()
    episodes = [
        e
        for e in store.episodes(status="active")
        if exclude_session is None or e.session_id != exclude_session or e.kind == "summary"
    ]
    candidates = score_beliefs(query_vec, store.beliefs(), now)
    candidates += score_episodes(query_vec, episodes, now)

    packed = pack_budget(candidates, token_budget)

    for memory in packed:
        new_stability = forgetting.reinforce(memory.stability_before)
        memory.stability_after = new_stability
        if memory.kind == "belief":
            store.reinforce_belief(memory.id, new_stability)
        else:
            store.reinforce_episode(memory.id, new_stability)

    return packed
