"""Configuration. Reads the Qwen Cloud API key from the environment or a local
`env` / `.env` file at the repository root (KEY=VALUE lines).
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_env_file() -> None:
    for name in ("env", ".env"):
        path = REPO_ROOT / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file()

# --- Qwen Cloud (Alibaba Cloud Model Studio / DashScope international) ---
QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
QWEN_API_KEY = (
    os.environ.get("DASHSCOPE_API_KEY")
    or os.environ.get("CLOUD_API_KEY")
    or os.environ.get("QWEN_API_KEY")
    or ""
)

# Reasoning model for the agent's replies; fast model for extraction/judging.
CHAT_MODEL = os.environ.get("QWEN_CHAT_MODEL", "qwen3.7-plus")
FAST_MODEL = os.environ.get("QWEN_FAST_MODEL", "qwen-flash")
EMBED_MODEL = os.environ.get("QWEN_EMBED_MODEL", "text-embedding-v4")
EMBED_DIM = int(os.environ.get("QWEN_EMBED_DIM", "512"))

# --- Memory engine ---
DB_PATH = os.environ.get("ENGRAM_DB", str(REPO_ROOT / "data" / "engram.db"))
MEMORY_TOKEN_BUDGET = int(os.environ.get("MEMORY_TOKEN_BUDGET", "1200"))
WORKING_MEMORY_TURNS = int(os.environ.get("WORKING_MEMORY_TURNS", "12"))
CONSOLIDATE_EVERY_N_TURNS = int(os.environ.get("CONSOLIDATE_EVERY_N_TURNS", "16"))

# Cosine-similarity threshold for spotting a contradiction between an incoming
# fact and an existing belief about the same subject under a *different*
# predicate (e.g. "moved to Berlin" vs "lives in Lisbon"). The model still makes
# the final supersede/coexist call; this only decides which beliefs to check.
BELIEF_CONTRADICTION_SIM = float(os.environ.get("BELIEF_CONTRADICTION_SIM", "0.5"))

# Retention below this marks an episode as a compression candidate.
COMPRESSION_RETENTION_THRESHOLD = float(
    os.environ.get("COMPRESSION_RETENTION_THRESHOLD", "0.25")
)
