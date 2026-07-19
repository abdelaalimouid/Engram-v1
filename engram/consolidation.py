"""Consolidation: turning experience into knowledge, and forgetting on purpose.

Two mechanisms:

1. Perception (per turn, cheap):  a single qwen-flash call scores importance,
   extracts entities, and distils candidate facts from a user turn. Facts are
   integrated into the belief ledger with contradiction handling. When a new
   fact collides with a currently-held belief on the same (subject, predicate),
   qwen-flash adjudicates: supersede, coexist, or discard. Superseding is
   bi-temporal: the old belief keeps its validity interval and provenance.

2. Sleep cycle (periodic):  episodes whose retention has decayed below the
   compression threshold are grouped by session, summarised by qwen-flash into
   one dense summary memory, and archived. The gist survives at a fraction of
   the token cost; verbatim detail is let go. This is the "timely forgetting"
   the track brief asks for.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np

from . import config, forgetting, qwen_cloud
from .store import Episode, MemoryStore

PERCEIVE_PROMPT = """You are the perception module of a memory system. Analyze ONE user message \
and return strict JSON:
{
  "importance": <float 0..1, how worth remembering long-term this message is>,
  "entities": [<distinct people/places/projects/things mentioned>],
  "facts": [
    {"subject": "...", "predicate": "...", "object": "...", "confidence": <float 0..1>}
  ]
}
Rules:
- Facts must be durable knowledge about the user or their world (identity, preferences, \
hobbies, skills, relationships, situations, decisions), NOT transient chit-chat.
- Use "user" as subject for facts about the speaker.
- If the speaker introduces themselves, ALWAYS record {"subject": "user", "predicate": \
"is named", "object": "<name>"}; identity facts are the most durable of all.
- Keep predicates short and canonical (e.g. "lives in", "is allergic to", \
"works on", "prefers", "dislikes", "has deadline").
- Never guess. If you do not know the object of a fact, omit that fact entirely; \
do NOT emit placeholders like "unknown", "unspecified", "n/a", or "none".
- Return {"importance": 0.1, "entities": [], "facts": []} for small talk.
- JSON only, no commentary."""

ADJUDICATE_PROMPT = """You are the belief-revision module of a memory system. An agent holds a \
current belief about a subject and has just learned a new fact about the same subject. They may \
concern the same attribute or closely related ones.

Current belief: "{old}"  (learned {age_days:.1f} days ago)
New fact:       "{new}"

Decide and return strict JSON: {{"action": "supersede" | "coexist" | "discard_new", "reason": "<short>"}}
- "supersede": the new fact updates or replaces the old one (e.g. the subject's location, job, or \
project name changed over time, or the new fact corrects the old).
- "coexist": both can be true at the same time (e.g. two different hobbies, or unrelated facts).
- "discard_new": the new fact is redundant or clearly less reliable than the old.
JSON only."""

SUMMARIZE_PROMPT = """You are the consolidation module of a memory system. Compress these fading \
conversation memories into ONE dense third-person summary (max 80 words) that preserves durable, \
useful information about the user and drops filler. Return only the summary text.

Memories:
{block}"""


# Degenerate values the extractor sometimes emits when it has nothing concrete;
# a fact containing any of these is dropped rather than stored.
PLACEHOLDER_VALUES = {
    "", "unknown", "unclear", "unspecified", "undefined", "n/a", "na", "none",
    "null", "tbd", "not specified", "not sure", "unsure", "?", "something",
    "someone", "somewhere", "nothing",
}


def perceive(text: str) -> dict[str, Any]:
    """One cheap qwen-flash call: importance + entities + candidate facts."""
    try:
        result = qwen_cloud.chat_json(
            [
                {"role": "system", "content": PERCEIVE_PROMPT},
                {"role": "user", "content": text},
            ],
            max_tokens=600,
        )
    except Exception:
        return {"importance": 0.3, "entities": [], "facts": []}
    result.setdefault("importance", 0.3)
    result.setdefault("entities", [])
    result.setdefault("facts", [])
    result["importance"] = min(max(float(result["importance"]), 0.0), 1.0)
    return result


def integrate_fact(
    store: MemoryStore,
    fact: dict[str, Any],
    source_episode: str | None,
) -> dict[str, Any]:
    """Insert a fact into the belief ledger with contradiction handling.
    Returns an event dict describing what happened (for the UI trace)."""
    subject = str(fact.get("subject", "")).strip()
    predicate = str(fact.get("predicate", "")).strip()
    obj = str(fact.get("object", "")).strip()
    confidence = min(max(float(fact.get("confidence", 0.7)), 0.0), 1.0)
    if not subject or not predicate or not obj:
        return {"action": "skipped", "reason": "incomplete triple"}
    if (
        subject.lower() in PLACEHOLDER_VALUES
        or predicate.lower() in PLACEHOLDER_VALUES
        or obj.lower() in PLACEHOLDER_VALUES
    ):
        return {"action": "skipped", "reason": "placeholder value"}

    same_pred = store.current_beliefs_for(subject, predicate)
    statement = f"{subject} {predicate} {obj}"

    # Exact duplicate: reinforce instead of duplicating.
    for belief in same_pred:
        if belief.object.strip().lower() == obj.lower():
            store.boost_belief_confidence(belief.id, max(belief.confidence, confidence) + 0.05)
            store.log_event("belief_reinforced", {"belief": statement, "id": belief.id})
            return {"action": "reinforced", "belief_id": belief.id, "statement": statement}

    embedding = qwen_cloud.embed_one(statement)

    # Candidates for contradiction handling: beliefs on the same (subject,
    # predicate), plus semantically-close beliefs about the same subject under a
    # different predicate, so "moved to Berlin" can supersede "lives in Lisbon",
    # not just an exact-predicate rewrite. The model makes the final call below.
    candidates: dict[str, Any] = {b.id: b for b in same_pred}
    for belief in _related_beliefs(store, subject, predicate, embedding):
        candidates.setdefault(belief.id, belief)

    new_belief = store.add_belief(
        subject, predicate, obj, confidence, source_episode,
        stability=forgetting.initial_stability(confidence),
        embedding=embedding,
    )

    outcomes = []
    for belief in candidates.values():
        verdict = _adjudicate(store, belief.statement(), statement, belief.created_at)
        action = verdict["action"]
        if action == "supersede":
            store.supersede_belief(belief.id, new_belief.id)
            store.log_event(
                "belief_superseded",
                {"old": belief.statement(), "new": statement, "reason": verdict.get("reason", "")},
            )
        elif action == "discard_new":
            store.supersede_belief(new_belief.id, belief.id)
            store.log_event(
                "belief_discarded",
                {"discarded": statement, "kept": belief.statement(), "reason": verdict.get("reason", "")},
            )
            outcomes.append(action)
            break  # the new belief is now invalidated; stop adjudicating.
        else:
            store.log_event("belief_coexists", {"a": belief.statement(), "b": statement})
        outcomes.append(action)

    if not candidates:
        store.log_event("belief_learned", {"belief": statement, "id": new_belief.id})
    return {"action": outcomes[0] if outcomes else "learned", "belief_id": new_belief.id, "statement": statement}


def _related_beliefs(
    store: MemoryStore, subject: str, predicate: str, embedding: np.ndarray
) -> list[Any]:
    """Current beliefs about the same subject under a *different* predicate whose
    embedding is close enough to the incoming statement to be a possible
    contradiction. Same-predicate beliefs are handled separately."""
    related = []
    for belief in store.current_beliefs_for_subject(subject):
        if belief.predicate.lower() == predicate.lower() or belief.embedding is None:
            continue
        if float(np.dot(belief.embedding, embedding)) >= config.BELIEF_CONTRADICTION_SIM:
            related.append(belief)
    return related


def _adjudicate(store: MemoryStore, old: str, new: str, old_created: float) -> dict[str, Any]:
    age_days = max(0.0, store.now() - old_created) / 86400.0
    try:
        verdict = qwen_cloud.chat_json(
            [{"role": "user", "content": ADJUDICATE_PROMPT.format(old=old, new=new, age_days=age_days)}],
            max_tokens=150,
        )
        if verdict.get("action") in ("supersede", "coexist", "discard_new"):
            return verdict
    except Exception:
        pass
    # Default: newer information wins (recency bias, like human belief updating).
    return {"action": "supersede", "reason": "adjudication unavailable; newest wins"}


def process_turn(store: MemoryStore, episode: Episode) -> dict[str, Any]:
    """Full perception pass for one stored user episode (runs in background)."""
    result = perceive(episode.content)
    integrations = [
        integrate_fact(store, fact, episode.id) for fact in result["facts"][:6]
    ]
    return {"importance": result["importance"], "entities": result["entities"], "integrations": integrations}


def sleep_cycle(store: MemoryStore) -> dict[str, Any]:
    """Compress faded episodes into summaries; archive the originals."""
    started = time.time()
    now = store.now()
    faded = [
        e
        for e in store.episodes(status="active")
        if e.kind == "episode"
        and forgetting.retention(now, e.last_access, e.stability)
        < config.COMPRESSION_RETENTION_THRESHOLD
    ]

    by_session: dict[str, list[Episode]] = defaultdict(list)
    for episode in faded:
        by_session[episode.session_id].append(episode)

    summaries_created = 0
    episodes_archived = 0
    for session_id, group in by_session.items():
        if len(group) < 2:
            continue  # not worth compressing a single stray line
        group.sort(key=lambda e: e.created_at)
        block = "\n".join(f"- [{e.role}] {e.content}" for e in group)
        try:
            summary_text = qwen_cloud.chat(
                [{"role": "user", "content": SUMMARIZE_PROMPT.format(block=block)}],
                model=config.FAST_MODEL,
                temperature=0.2,
                max_tokens=200,
            ).strip()
        except Exception:
            continue
        if not summary_text:
            continue
        embedding = qwen_cloud.embed_one(summary_text)
        mean_importance = sum(e.importance for e in group) / len(group)
        store.add_episode(
            session_id=session_id,
            role="system",
            content=summary_text,
            importance=min(1.0, mean_importance + 0.2),  # gist outranks any single line
            stability=forgetting.initial_stability(min(1.0, mean_importance + 0.2)) * 2,
            embedding=embedding,
            kind="summary",
            source_ids=[e.id for e in group],
        )
        store.archive_episodes([e.id for e in group])
        summaries_created += 1
        episodes_archived += len(group)
        store.log_event(
            "sleep_compression",
            {"session": session_id, "archived": len(group), "summary": summary_text[:120]},
        )

    report = {
        "faded_candidates": len(faded),
        "summaries_created": summaries_created,
        "episodes_archived": episodes_archived,
        "duration_s": round(time.time() - started, 2),
    }
    store.log_event("sleep_cycle", report)
    return report
