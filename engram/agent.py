"""The agent loop.

Per turn:
  1. Embed the user message (text-embedding-v4 on Qwen Cloud).
  2. Recall: score every belief and episode, pack the best under the token
     budget, reinforce what was recalled.
  3. Answer with qwen3.7-plus, grounding on the packed memory block plus a
     short working-memory window of the current session.
  4. Store the turn as episodic memory.
  5. (Background) Perceive the turn: importance, entities, fact extraction,
     belief integration with contradiction handling.

The recall trace is returned alongside the reply so the UI can show exactly
which memories fired, how strong they were, and how they were reinforced.
"""

from __future__ import annotations

import json
from typing import Any

from . import config, consolidation, forgetting, qwen_cloud, retrieval
from .store import MemoryStore

SYSTEM_PROMPT = """You are Engram, an assistant with a persistent long-term memory that spans \
every past session with this user.

Below is your MEMORY RECALL for this turn — beliefs and episodes retrieved from long-term \
memory, ordered by recall strength. Treat them as your own remembered knowledge:
- Use them naturally; never say "according to my memory database".
- Prefer beliefs (distilled facts) over raw episodes when they conflict.
- If memory contradicts what the user just said, trust the user and note the update gracefully.
- If you genuinely don't remember something, say so honestly rather than inventing.

MEMORY RECALL:
{memory_block}"""

NO_MEMORY_LINE = "(no relevant long-term memories surfaced for this turn)"


class EngramAgent:
    def __init__(self, store: MemoryStore | None = None):
        self.store = store or MemoryStore()
        self._turns_since_sleep = 0

    def chat(self, session_id: str, message: str) -> dict[str, Any]:
        query_vec = qwen_cloud.embed_one(message)

        recalled = retrieval.recall(
            self.store,
            query_vec,
            token_budget=config.MEMORY_TOKEN_BUDGET,
            exclude_session=session_id,
        )
        memory_block = (
            "\n".join(
                f"- ({m.kind}, strength {m.score:.2f}, retention {m.retention:.2f}) {m.text}"
                for m in recalled
            )
            or NO_MEMORY_LINE
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(memory_block=memory_block)}
        ]
        messages += self._working_memory(session_id)
        messages.append({"role": "user", "content": message})

        reply = qwen_cloud.chat(messages, model=config.CHAT_MODEL, max_tokens=1024)

        user_episode = self.store.add_episode(
            session_id=session_id,
            role="user",
            content=message,
            importance=0.3,  # provisional; perception refines it in background
            stability=forgetting.initial_stability(0.3),
            embedding=query_vec,
        )
        self.store.add_episode(
            session_id=session_id,
            role="assistant",
            content=reply,
            importance=0.2,
            stability=forgetting.initial_stability(0.2),
            embedding=qwen_cloud.embed_one(reply),
        )
        self.store.log_event(
            "turn",
            {
                "session": session_id,
                "recalled": len(recalled),
                "memory_tokens": sum(m.tokens for m in recalled),
            },
        )

        self._turns_since_sleep += 1
        return {
            "reply": reply,
            "user_episode_id": user_episode.id,
            "recall_trace": [self._trace(m) for m in recalled],
            "memory_tokens_used": sum(m.tokens for m in recalled),
            "memory_token_budget": config.MEMORY_TOKEN_BUDGET,
        }

    def perceive_turn(self, episode_id: str) -> dict[str, Any]:
        """Background perception for a stored user episode: refine importance,
        extract entities/facts, integrate beliefs."""
        row = self.store.conn.execute(
            "SELECT * FROM episodes WHERE id=?", (episode_id,)
        ).fetchone()
        if row is None:
            return {"error": "episode not found"}
        episode = self.store._row_to_episode(row)
        result = consolidation.process_turn(self.store, episode)
        self.store.conn.execute(
            "UPDATE episodes SET importance=?, stability=?, entities=? WHERE id=?",
            (
                result["importance"],
                forgetting.initial_stability(result["importance"]),
                json.dumps(result["entities"]),
                episode_id,
            ),
        )
        self.store.conn.commit()
        return result

    def maybe_sleep(self) -> dict[str, Any] | None:
        if self._turns_since_sleep >= config.CONSOLIDATE_EVERY_N_TURNS:
            self._turns_since_sleep = 0
            return consolidation.sleep_cycle(self.store)
        return None

    def _working_memory(self, session_id: str) -> list[dict[str, str]]:
        rows = self.store.conn.execute(
            "SELECT role, content FROM episodes "
            "WHERE session_id=? AND kind='episode' AND role IN ('user','assistant') "
            "ORDER BY created_at DESC LIMIT ?",
            (session_id, config.WORKING_MEMORY_TURNS),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    @staticmethod
    def _trace(memory: retrieval.RecalledMemory) -> dict[str, Any]:
        return {
            "kind": memory.kind,
            "id": memory.id,
            "text": memory.text,
            "score": round(memory.score, 3),
            "similarity": round(memory.similarity, 3),
            "retention": round(memory.retention, 3),
            "importance": round(memory.importance, 3),
            "tokens": memory.tokens,
            "stability_before_h": round(memory.stability_before, 1),
            "stability_after_h": round(memory.stability_after, 1),
            "provenance": memory.provenance,
        }
