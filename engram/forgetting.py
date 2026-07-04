"""The forgetting model.

Every memory carries a *stability* S (hours). Its retention follows an
Ebbinghaus-style exponential forgetting curve:

    R(t) = exp(-Δt / S)        Δt = hours since last access

Recalling a memory reinforces it (spaced repetition): stability grows
multiplicatively, so frequently-used memories become nearly permanent while
untouched ones fade toward the compression threshold, where the consolidation
sleep cycle summarises and archives them.

Importance sets the starting stability: a throwaway remark starts with ~a day
of stability; a critical fact starts with ~five days.
"""

from __future__ import annotations

import math

HOUR = 3600.0

# Stability bounds (hours): between half a day and ten years.
MIN_STABILITY = 12.0
MAX_STABILITY = 24.0 * 365 * 10

# Reinforcement: S' = S * GROWTH + BONUS
REINFORCE_GROWTH = 1.6
REINFORCE_BONUS = 12.0


def initial_stability(importance: float) -> float:
    """Map importance in [0, 1] to starting stability in hours (24h..120h)."""
    importance = min(max(importance, 0.0), 1.0)
    return 24.0 * (1.0 + 4.0 * importance)


def retention(now: float, last_access: float, stability: float) -> float:
    """Current retention in [0, 1] given a unix `now` and `last_access`."""
    if stability <= 0:
        return 0.0
    hours_elapsed = max(0.0, now - last_access) / HOUR
    return math.exp(-hours_elapsed / stability)


def reinforce(stability: float) -> float:
    """Strengthen a memory after successful recall."""
    return min(MAX_STABILITY, max(MIN_STABILITY, stability * REINFORCE_GROWTH + REINFORCE_BONUS))


def recency_score(now: float, created_at: float, half_life_hours: float = 168.0) -> float:
    """Soft recency prior with a one-week half-life, independent of retention."""
    hours_elapsed = max(0.0, now - created_at) / HOUR
    return math.exp(-math.log(2) * hours_elapsed / half_life_hours)
