# Engram — Architecture

## Design thesis

The MemoryAgent track asks for three hard properties: efficient storage and retrieval, **timely forgetting of outdated information**, and recalling critical memories **within limited context windows**. Standard RAG-over-history fails all three as it scales: the index grows monotonically, stale facts retrieve as confidently as fresh ones, and "top-k" retrieval has no notion of a token budget.

Engram's answer is to model memory the way cognitive science does — as a *dynamical system*, not a database.

## The two memory systems

```
 experience ──► EPISODIC STORE ──(sleep cycle)──► summaries ──► archive
                     │
                (perception, qwen-flash)
                     ▼
                BELIEF LEDGER  ◄─ contradiction adjudication, supersession
```

### Episodic store (`episodes` table)

Raw interaction events. Each carries:

- `importance` ∈ [0,1] — scored by qwen-flash perception (provisionally 0.3 at write time, refined in a background task so the chat path stays fast).
- `stability` S (hours) — the forgetting-curve time constant. Initialized as `S₀ = 24·(1 + 4·importance)`: trivia starts with ~1 day of stability, critical facts with ~5 days.
- `last_access` / `access_count` — recall telemetry.

### Belief ledger (`beliefs` table) — bi-temporal SPO triples

Perception distils durable facts into `(subject, predicate, object)` triples: `(user, is allergic to, shellfish)`. The ledger is **bi-temporal**:

- `valid_from` / `valid_to` — when the fact held in the world. `valid_to IS NULL` means currently held.
- Nothing is ever deleted. When the user says "I moved to Ponta Delgada", the old belief `(user, lives in, Lisbon)` gets `valid_to = now` and `superseded_by = <new id>`. The agent can answer both "where do I live?" and, in principle, "where did I use to live?".
- `source_episode` provides provenance — every belief can cite the exact conversational moment it was learned.

**Contradiction handling:** when a new fact shares `(subject, predicate)` with a current belief but differs in object, qwen-flash adjudicates one of three verdicts: `supersede` (state changed / correction), `coexist` (both true — hobbies, multiple projects), `discard_new` (redundant or less reliable). If adjudication fails, the newest information wins — the same recency bias humans use for belief updating.

## Forgetting (`engram/forgetting.py`)

Retention follows the Ebbinghaus curve:

```
R(t) = exp(−Δt / S)          Δt = hours since last access
```

Recall triggers **spaced-repetition reinforcement**:

```
S′ = min(S · 1.6 + 12, S_max)
```

Memories the agent actually uses become effectively permanent within a handful of recalls; memories it never touches decay toward the compression threshold. This solves "timely forgetting" *without ever making an explicit delete decision* — relevance is revealed by usage, not predicted upfront.

## Retrieval under a token budget (`engram/retrieval.py`)

Each candidate (belief or episode) is scored:

```
relevance = 0.60·cosine_sim + 0.20·importance + 0.20·recency_prior
score     = relevance · (0.15 + 0.85·retention)
```

Retention gates the score **multiplicatively with a floor**: a faded memory can still surface on a very strong semantic match (recognition beats free recall, as in humans), but weak matches on faded memories drop out entirely.

Selected candidates are then packed under a hard budget (default 1,200 tokens) by **greedy knapsack on score density** (score per token) — a dense one-line belief beats a rambling episode with a slightly higher raw score. If nothing fits, the single best memory is truncated to fit; the agent never goes in blind. This is the direct answer to "recall within limited context windows": memory context is *O(budget)*, not *O(history)*.

Everything that made the cut is reinforced — retrieval and forgetting close the loop.

## Consolidation sleep cycle (`engram/consolidation.py`)

Periodically (every N turns, or on demand):

1. Find active episodes with `retention < 0.25`.
2. Group by session, summarise each group with qwen-flash into one dense gist (≤80 words).
3. Store the summary as a new high-stability memory (its importance = mean of sources + 0.2; its stability doubled — gist is more durable than verbatim).
4. Archive the originals (kept for audit, excluded from recall).

Long-term memory therefore *asymptotes* instead of growing linearly: verbatim recent memory, compressed remote memory — the same gradient biological consolidation produces.

## The timewarp clock

`store.now()` = wall clock + a persisted offset. `POST /api/timewarp` advances simulated time so decay, consolidation, and supersession — processes that naturally take weeks — can be demonstrated in a 3-minute video. The offset applies uniformly to every timestamped computation, so accelerated behavior is *identical* to real elapsed time.

## Qwen Cloud model routing

| Model | Role | Why |
|---|---|---|
| `qwen3.7-plus` | Agent replies | Reasoning model; grounds answers on the packed memory block, handles graceful belief-conflict responses |
| `qwen-flash` | Perception, adjudication, summarisation, eval judging | High call volume, low unit cost; all calls are single-purpose JSON tasks |
| `text-embedding-v4` | Memory embeddings (512-dim, unit-normalized) | Cosine similarity = dot product; 512 dims keeps the store compact |

All calls route through `engram/qwen_cloud.py` → `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (Alibaba Cloud Model Studio, international).

**Cost profile per turn:** 1 embedding + 1 qwen3.7-plus reply + (background) 1 qwen-flash perception + occasional adjudications. The heavy reasoning model is never used for bookkeeping.

## Failure handling

- Perception/adjudication failures degrade gracefully: the turn still completes with provisional importance; adjudication falls back to newest-wins.
- JSON parsing tolerates code fences and prose wrappers, with temperature-bumped retries.
- The background perception task never blocks the chat path.
- SQLite with WAL-free simple transactions — one file, trivially portable; the store interface is narrow enough to swap for RDS/PolarDB in production.

## Scaling notes

At hackathon scale, scoring is a dense matrix–vector product over all candidates (< 10k rows, sub-millisecond). The upgrade path is mechanical: swap the in-process scorer for an ANN index (e.g. Alibaba Cloud AnalyticDB vector search or OpenSearch) *for the similarity term only* — importance, recency, and retention remain cheap SQL columns, and the knapsack packer is unchanged.
