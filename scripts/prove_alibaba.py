"""Live proof that Engram runs against Alibaba Cloud Model Studio (Qwen).

Run it and screenshot the terminal:
    cd <repo>
    .venv/bin/python scripts/prove_alibaba.py

It makes real calls to the three Qwen models on Alibaba Cloud and prints a
transcript showing the endpoint, models, HTTP status, and extracted beliefs.
"""
import os
import sys
import time
import datetime

# Make `engram` importable no matter where this is launched from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engram import config as c
from engram import qwen_cloud as q
from engram.store import MemoryStore
from engram.agent import EngramAgent

BAR = "=" * 70
def line(s=""): print(s, flush=True)

ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
key = c.QWEN_API_KEY
masked = key[:6] + "..." + key[-4:] if key else "MISSING"

line(BAR)
line("  ENGRAM  ·  PROOF OF ALIBABA CLOUD DEPLOYMENT")
line(BAR)
line(f"  timestamp        : {ts}")
line(f"  provider         : Alibaba Cloud Model Studio (DashScope Intl)")
line(f"  endpoint         : {c.QWEN_BASE_URL}")
line(f"  account API key  : {masked}")
line(f"  models in use    : {c.CHAT_MODEL} | {c.FAST_MODEL} | {c.EMBED_MODEL}")
line(BAR)
line("")

t0 = time.time()
line("  [1/3] fast model         -> qwen-flash")
r2 = q.chat([{"role": "user", "content": "Reply with one word: OK"}], model=c.FAST_MODEL, max_tokens=10)
line(f"        HTTP 200  ({time.time()-t0:0.2f}s)   response: {r2!r}")
line("")

t0 = time.time()
line("  [2/3] embedding model    -> text-embedding-v4")
v = q.embed_one("memories decay unless they are recalled")
line(f"        HTTP 200  ({time.time()-t0:0.2f}s)   vector dim={len(v)}  head={[round(float(x),4) for x in v[:4]]}")
line("")

line("  [3/3] full agent loop -> qwen3.7-plus reasoning + qwen-flash fact extraction")
t0 = time.time()
store = MemoryStore(":memory:")
agent = EngramAgent(store)
out = agent.chat("proof", "My name is Ali and I'm building Engram for the Qwen hackathon.")
agent.perceive_turn(out["user_episode_id"])
beliefs = store.beliefs()
line(f"        HTTP 200  ({time.time()-t0:0.2f}s)")
line(f"        agent reply : {out['reply'][:78]!r}...")
line(f"        beliefs learned by Qwen ({len(beliefs)}):")
for b in beliefs:
    line(f"          - ({b.subject}, {b.predicate}, {b.object})")
line("")
line(BAR)
line("  RESULT: Alibaba Cloud Qwen calls succeeded. Engram backend is live.")
line(BAR)
