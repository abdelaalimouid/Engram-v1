"""FastAPI server exposing the Engram agent and its memory internals.

Run:  uvicorn engram.server:app --host 0.0.0.0 --port 8000
UI:   http://localhost:8000
"""

from __future__ import annotations

from pathlib import Path

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, consolidation, forgetting
from .agent import EngramAgent

app = FastAPI(title="Engram", version="1.0.0")
agent = EngramAgent()

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    message: str = Field(min_length=1, max_length=8000)


class TimewarpRequest(BaseModel):
    hours: float = Field(gt=0, le=24 * 365)


@app.post("/api/chat")
def chat(request: ChatRequest, background: BackgroundTasks):
    result = agent.chat(request.session_id, request.message)
    background.add_task(agent.perceive_turn, result["user_episode_id"])
    sleep_report = agent.maybe_sleep()
    if sleep_report:
        result["sleep_report"] = sleep_report
    return result


@app.get("/api/memories")
def memories(status: str = "active"):
    store = agent.store
    now = store.now()
    return [
        {
            "id": e.id,
            "session_id": e.session_id,
            "role": e.role,
            "kind": e.kind,
            "content": e.content,
            "created_at": e.created_at,
            "importance": round(e.importance, 2),
            "stability_h": round(e.stability, 1),
            "retention": round(forgetting.retention(now, e.last_access, e.stability), 3),
            "access_count": e.access_count,
            "status": e.status,
            "source_ids": e.source_ids,
        }
        for e in store.episodes(status=status if status != "all" else None)
    ]


@app.get("/api/beliefs")
def beliefs(include_superseded: bool = True):
    store = agent.store
    now = store.now()
    return [
        {
            "id": b.id,
            "subject": b.subject,
            "predicate": b.predicate,
            "object": b.object,
            "statement": b.statement(),
            "confidence": round(b.confidence, 2),
            "current": b.valid_to is None,
            "valid_from": b.valid_from,
            "valid_to": b.valid_to,
            "superseded_by": b.superseded_by,
            "source_episode": b.source_episode,
            "retention": round(forgetting.retention(now, b.last_access, b.stability), 3),
            "access_count": b.access_count,
        }
        for b in store.beliefs(include_superseded=include_superseded)
    ]


@app.get("/api/events")
def events(limit: int = 60):
    return agent.store.recent_events(limit=min(limit, 200))


@app.get("/api/stats")
def stats():
    payload = agent.store.stats()
    payload["memory_token_budget"] = config.MEMORY_TOKEN_BUDGET
    payload["models"] = {
        "chat": config.CHAT_MODEL,
        "fast": config.FAST_MODEL,
        "embedding": config.EMBED_MODEL,
    }
    return payload


@app.post("/api/consolidate")
def consolidate():
    return consolidation.sleep_cycle(agent.store)


@app.post("/api/timewarp")
def timewarp(request: TimewarpRequest):
    total = agent.store.timewarp(request.hours)
    return {"advanced_hours": request.hours, "total_offset_hours": total}


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
