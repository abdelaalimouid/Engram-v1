"""Persistent memory store (SQLite).

Two memory systems, mirroring the episodic/semantic split in biological memory:

  episodes  Raw interaction events plus consolidation summaries. Subject to
            forgetting: status walks active -> archived once a summary
            (kind='summary') absorbs them.

  beliefs   Semantic facts as (subject, predicate, object) triples with
            *bi-temporal* validity: beliefs are never deleted, they are
            superseded. `valid_to IS NULL` means currently held. Every belief
            keeps provenance to the episode it was learned from.

A `clock` with a persisted offset supports "timewarp": advancing simulated
time so forgetting and consolidation can be demonstrated live.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id            TEXT PRIMARY KEY,
    session_id    TEXT NOT NULL,
    role          TEXT NOT NULL,             -- user | assistant | system
    kind          TEXT NOT NULL DEFAULT 'episode',  -- episode | summary
    content       TEXT NOT NULL,
    created_at    REAL NOT NULL,
    importance    REAL NOT NULL DEFAULT 0.3,
    stability     REAL NOT NULL,             -- forgetting-curve stability, hours
    last_access   REAL NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'active',   -- active | archived
    source_ids    TEXT,                      -- JSON list, for summaries
    entities      TEXT,                      -- JSON list of entity strings
    embedding     BLOB
);

CREATE TABLE IF NOT EXISTS beliefs (
    id             TEXT PRIMARY KEY,
    subject        TEXT NOT NULL,
    predicate      TEXT NOT NULL,
    object         TEXT NOT NULL,
    confidence     REAL NOT NULL DEFAULT 0.8,
    created_at     REAL NOT NULL,
    valid_from     REAL NOT NULL,
    valid_to       REAL,                     -- NULL = currently held
    superseded_by  TEXT,                     -- belief id that replaced this one
    source_episode TEXT,                     -- provenance
    stability      REAL NOT NULL,
    last_access    REAL NOT NULL,
    access_count   INTEGER NOT NULL DEFAULT 0,
    embedding      BLOB
);

CREATE TABLE IF NOT EXISTS events (
    id         TEXT PRIMARY KEY,
    ts         REAL NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_status ON episodes(status);
CREATE INDEX IF NOT EXISTS idx_beliefs_current ON beliefs(subject, predicate)
    WHERE valid_to IS NULL;
"""


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _pack(vector: np.ndarray | None) -> bytes | None:
    return None if vector is None else np.asarray(vector, dtype=np.float32).tobytes()


def _unpack(blob: bytes | None) -> np.ndarray | None:
    return None if blob is None else np.frombuffer(blob, dtype=np.float32)


@dataclass
class Episode:
    id: str
    session_id: str
    role: str
    kind: str
    content: str
    created_at: float
    importance: float
    stability: float
    last_access: float
    access_count: int
    status: str
    source_ids: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    embedding: np.ndarray | None = None


@dataclass
class Belief:
    id: str
    subject: str
    predicate: str
    object: str
    confidence: float
    created_at: float
    valid_from: float
    valid_to: float | None
    superseded_by: str | None
    source_episode: str | None
    stability: float
    last_access: float
    access_count: int
    embedding: np.ndarray | None = None

    def statement(self) -> str:
        return f"{self.subject} {self.predicate} {self.object}"


class MemoryStore:
    def __init__(self, db_path: str | None = None):
        path = db_path or config.DB_PATH
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------- clock ---
    def now(self) -> float:
        """Simulated time = wall clock + persisted timewarp offset."""
        return time.time() + self._clock_offset()

    def _clock_offset(self) -> float:
        row = self.conn.execute("SELECT value FROM meta WHERE key='clock_offset'").fetchone()
        return float(row["value"]) if row else 0.0

    def timewarp(self, hours: float) -> float:
        offset = self._clock_offset() + hours * 3600.0
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('clock_offset', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(offset),),
        )
        self.conn.commit()
        self.log_event("timewarp", {"hours": hours, "total_offset_hours": offset / 3600.0})
        return offset / 3600.0

    # ---------------------------------------------------------- episodes ---
    def add_episode(
        self,
        session_id: str,
        role: str,
        content: str,
        importance: float,
        stability: float,
        embedding: np.ndarray | None,
        kind: str = "episode",
        source_ids: list[str] | None = None,
        entities: list[str] | None = None,
    ) -> Episode:
        now = self.now()
        episode = Episode(
            id=_new_id("ep"),
            session_id=session_id,
            role=role,
            kind=kind,
            content=content,
            created_at=now,
            importance=importance,
            stability=stability,
            last_access=now,
            access_count=0,
            status="active",
            source_ids=source_ids or [],
            entities=entities or [],
            embedding=embedding,
        )
        self.conn.execute(
            "INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                episode.id, episode.session_id, episode.role, episode.kind,
                episode.content, episode.created_at, episode.importance,
                episode.stability, episode.last_access, episode.access_count,
                episode.status, json.dumps(episode.source_ids),
                json.dumps(episode.entities), _pack(episode.embedding),
            ),
        )
        self.conn.commit()
        return episode

    def episodes(self, status: str | None = "active", limit: int = 2000) -> list[Episode]:
        query = "SELECT * FROM episodes"
        params: tuple[Any, ...] = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = self.conn.execute(query, params + (limit,)).fetchall()
        return [self._row_to_episode(row) for row in rows]

    def _row_to_episode(self, row: sqlite3.Row) -> Episode:
        return Episode(
            id=row["id"], session_id=row["session_id"], role=row["role"],
            kind=row["kind"], content=row["content"], created_at=row["created_at"],
            importance=row["importance"], stability=row["stability"],
            last_access=row["last_access"], access_count=row["access_count"],
            status=row["status"],
            source_ids=json.loads(row["source_ids"] or "[]"),
            entities=json.loads(row["entities"] or "[]"),
            embedding=_unpack(row["embedding"]),
        )

    def reinforce_episode(self, episode_id: str, new_stability: float) -> None:
        self.conn.execute(
            "UPDATE episodes SET stability=?, last_access=?, access_count=access_count+1 "
            "WHERE id=?",
            (new_stability, self.now(), episode_id),
        )
        self.conn.commit()

    def archive_episodes(self, ids: list[str]) -> None:
        self.conn.executemany(
            "UPDATE episodes SET status='archived' WHERE id=?", [(i,) for i in ids]
        )
        self.conn.commit()

    # ----------------------------------------------------------- beliefs ---
    def add_belief(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float,
        source_episode: str | None,
        stability: float,
        embedding: np.ndarray | None,
    ) -> Belief:
        now = self.now()
        belief = Belief(
            id=_new_id("bl"), subject=subject, predicate=predicate, object=obj,
            confidence=confidence, created_at=now, valid_from=now, valid_to=None,
            superseded_by=None, source_episode=source_episode,
            stability=stability, last_access=now, access_count=0,
            embedding=embedding,
        )
        self.conn.execute(
            "INSERT INTO beliefs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                belief.id, belief.subject, belief.predicate, belief.object,
                belief.confidence, belief.created_at, belief.valid_from,
                belief.valid_to, belief.superseded_by, belief.source_episode,
                belief.stability, belief.last_access, belief.access_count,
                _pack(belief.embedding),
            ),
        )
        self.conn.commit()
        return belief

    def beliefs(self, include_superseded: bool = False, limit: int = 2000) -> list[Belief]:
        query = "SELECT * FROM beliefs"
        if not include_superseded:
            query += " WHERE valid_to IS NULL"
        query += " ORDER BY created_at DESC LIMIT ?"
        rows = self.conn.execute(query, (limit,)).fetchall()
        return [self._row_to_belief(row) for row in rows]

    def current_beliefs_for(self, subject: str, predicate: str) -> list[Belief]:
        rows = self.conn.execute(
            "SELECT * FROM beliefs WHERE valid_to IS NULL "
            "AND lower(subject)=lower(?) AND lower(predicate)=lower(?)",
            (subject, predicate),
        ).fetchall()
        return [self._row_to_belief(row) for row in rows]

    def _row_to_belief(self, row: sqlite3.Row) -> Belief:
        return Belief(
            id=row["id"], subject=row["subject"], predicate=row["predicate"],
            object=row["object"], confidence=row["confidence"],
            created_at=row["created_at"], valid_from=row["valid_from"],
            valid_to=row["valid_to"], superseded_by=row["superseded_by"],
            source_episode=row["source_episode"], stability=row["stability"],
            last_access=row["last_access"], access_count=row["access_count"],
            embedding=_unpack(row["embedding"]),
        )

    def supersede_belief(self, old_id: str, new_id: str) -> None:
        """Bi-temporal invalidation: the old belief keeps its history but stops
        being current. Nothing is ever deleted."""
        self.conn.execute(
            "UPDATE beliefs SET valid_to=?, superseded_by=? WHERE id=?",
            (self.now(), new_id, old_id),
        )
        self.conn.commit()

    def boost_belief_confidence(self, belief_id: str, confidence: float) -> None:
        self.conn.execute(
            "UPDATE beliefs SET confidence=?, last_access=?, access_count=access_count+1 "
            "WHERE id=?",
            (min(confidence, 0.99), self.now(), belief_id),
        )
        self.conn.commit()

    def reinforce_belief(self, belief_id: str, new_stability: float) -> None:
        self.conn.execute(
            "UPDATE beliefs SET stability=?, last_access=?, access_count=access_count+1 "
            "WHERE id=?",
            (new_stability, self.now(), belief_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------ events ---
    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events VALUES (?,?,?,?)",
            (_new_id("ev"), self.now(), event_type, json.dumps(payload)),
        )
        self.conn.commit()

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"id": r["id"], "ts": r["ts"], "type": r["type"], "payload": json.loads(r["payload"])}
            for r in rows
        ]

    # ------------------------------------------------------------- stats ---
    def stats(self) -> dict[str, Any]:
        one = lambda q: self.conn.execute(q).fetchone()[0]  # noqa: E731
        return {
            "episodes_active": one("SELECT COUNT(*) FROM episodes WHERE status='active' AND kind='episode'"),
            "summaries": one("SELECT COUNT(*) FROM episodes WHERE kind='summary' AND status='active'"),
            "episodes_archived": one("SELECT COUNT(*) FROM episodes WHERE status='archived'"),
            "beliefs_current": one("SELECT COUNT(*) FROM beliefs WHERE valid_to IS NULL"),
            "beliefs_superseded": one("SELECT COUNT(*) FROM beliefs WHERE valid_to IS NOT NULL"),
            "clock_offset_hours": self._clock_offset() / 3600.0,
            "simulated_now": self.now(),
        }
