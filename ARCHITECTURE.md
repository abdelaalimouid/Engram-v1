# Engram architecture

## Design goals

The MemoryAgent track asks for three properties: efficient storage and retrieval, timely forgetting of outdated information, and recall of critical memories within limited context windows. Plain RAG over chat history fails all three as it scales. The index grows monotonically, stale facts are retrieved as confidently as fresh ones, and top-k retrieval has no concept of a token budget.

Engram treats memory as a dynamical system instead of a database. Memories have state that changes over time, and retrieval feeds back into that state.

## Two memory systems

```
 experience --> EPISODIC STORE --(sleep cycle)--> summaries --> archive
                     |
               (fact extraction, qwen-flash)
                     v
                BELIEF LEDGER  <-- contradiction adjudication, supersession
```

### Episodic store (`episodes` table)

Raw interaction events. Each carries:

- `importance` in [0,1], scored by a qwen-flash pass over the message. The score is provisional (0.3) at write time and refined in a background task so the chat path stays fast.
- `stability` S in hours, the forgetting-curve time constant. Initialised as `S0 = 24 * (1 + 4 * importance)`: small talk starts with about a day of stability, critical facts with about five days.
- `last_access` and `access_count` for recall telemetry.

### Belief ledger (`beliefs` table)

Durable facts distilled into (subject, predicate, object) triples, for example `(user, is allergic to, shellfish)`. The ledger is bi-temporal:

- `valid_from` / `valid_to` record when the fact held. `valid_to IS NULL` means currently held.
- Beliefs are never deleted. When the user says "I moved to Ponta Delgada", the old belief `(user, lives in, Lisbon)` gets `valid_to = now` and `superseded_by = <new id>`. The agent can answer both "where do I live?" and "where did I use to live?".
- `source_episode` links each belief to the exact conversational moment it was learned, so any answer can be traced back to its origin.

When a new fact shares (subject, predicate) with a current belief but differs in object, qwen-flash picks one of three verdicts: `supersede` (the state changed, or this is a correction), `coexist` (both can hold, e.g. two hobbies), or `discard_new` (redundant or less reliable). If the adjudication call fails, the newest information wins.

## Forgetting (`engram/forgetting.py`)

Retention follows an exponential forgetting curve:

```
R(t) = exp(-dt / S)        dt = hours since last access
```

Recall triggers reinforcement, the same update spaced-repetition systems use:

```
S' = min(S * 1.6 + 12, S_max)
```

A memory recalled a handful of times becomes effectively permanent. A memory never recalled decays toward the compression threshold. The system never has to make an explicit "delete this" decision; usage reveals what matters.

## Retrieval under a token budget (`engram/retrieval.py`)

Each candidate belief or episode is scored:

```
relevance = 0.60 * cosine_sim + 0.20 * importance + 0.20 * recency_prior
score     = relevance * (0.15 + 0.85 * retention)
```

Retention gates the score multiplicatively, with a floor. A faded memory can still surface on a very strong semantic match (recognition is easier than free recall, as in humans), but weak matches on faded memories drop out.

Selected candidates are then packed under a hard budget (default 1,200 tokens) by a greedy knapsack on score per token. A one-line belief with a slightly lower score beats a rambling episode with a slightly higher one. If nothing fits, the single best memory is truncated to fit, so the agent never answers blind when a strong memory exists. Memory context is bounded by the budget, not by the length of the history.

Everything that makes the cut gets reinforced. Retrieval and forgetting close the loop.

## Consolidation sleep cycle (`engram/consolidation.py`)

Every N turns, or on demand:

1. Find active episodes with retention below 0.25.
2. Group them by session and summarise each group with qwen-flash into one gist of at most 80 words.
3. Store the summary as a new memory with higher importance and doubled stability. The gist is more durable than the verbatim lines it replaces.
4. Archive the originals. They are kept for audit but excluded from recall.

Long-term storage levels off instead of growing linearly: verbatim recent memory, compressed remote memory.

## The simulated clock

`store.now()` returns wall time plus a persisted offset. `POST /api/timewarp` advances the offset so that decay, consolidation, and supersession, which naturally play out over weeks, can be observed in minutes. The offset applies uniformly to every timestamped computation, so accelerated behaviour is identical to real elapsed time.

## Model routing

| Model | Role | Reason |
|---|---|---|
| `qwen3.7-plus` | Agent replies | Reasoning model; grounds answers on the packed memory block and handles belief conflicts gracefully |
| `qwen-flash` | Fact extraction, adjudication, summarisation, benchmark judging | High call volume, low unit cost; every call is a single-purpose JSON task |
| `text-embedding-v4` | Memory embeddings, 512-dim, unit-normalised | Cosine similarity reduces to a dot product; 512 dims keeps the store small |

All calls route through `engram/qwen_cloud.py` to `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (Alibaba Cloud Model Studio, international). Per turn the cost is one embedding, one qwen3.7-plus reply, and one background qwen-flash extraction, plus occasional adjudications. The reasoning model never does bookkeeping.

## Failure handling

- Extraction or adjudication failures degrade gracefully: the turn completes with the provisional importance score, and adjudication falls back to newest-wins.
- JSON parsing tolerates code fences and surrounding prose, with a temperature-bumped retry.
- Background extraction never blocks the chat path.
- Storage is a single SQLite file behind a narrow store interface, so swapping in ApsaraDB RDS for multi-instance deployments is a contained change.

## Scaling

At current scale, scoring is a dense matrix-vector product over all candidates (under 10k rows, sub-millisecond). The upgrade path is to move only the similarity term to an ANN index such as AnalyticDB vector search. Importance, recency, and retention stay as cheap SQL columns, and the knapsack packer is unchanged.
