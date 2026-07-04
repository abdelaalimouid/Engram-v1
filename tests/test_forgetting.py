import math

from engram import forgetting


def test_retention_is_one_at_access_time():
    now = 1_000_000.0
    assert forgetting.retention(now, now, stability=48.0) == 1.0


def test_retention_decays_exponentially():
    now = 1_000_000.0
    stability_hours = 48.0
    one_stability_later = now + stability_hours * 3600
    retention = forgetting.retention(one_stability_later, now, stability_hours)
    assert math.isclose(retention, math.exp(-1), rel_tol=1e-9)


def test_retention_never_negative_or_above_one():
    now = 1_000_000.0
    assert 0.0 <= forgetting.retention(now + 1e9, now, 12.0) <= 1.0
    # Access "in the future" (clock skew) clamps to full retention.
    assert forgetting.retention(now, now + 500, 12.0) == 1.0


def test_reinforce_grows_and_caps():
    stability = 24.0
    for _ in range(200):
        new = forgetting.reinforce(stability)
        assert new > stability or new == forgetting.MAX_STABILITY
        stability = new
    assert stability == forgetting.MAX_STABILITY


def test_initial_stability_scales_with_importance():
    assert forgetting.initial_stability(0.0) == 24.0
    assert forgetting.initial_stability(1.0) == 120.0
    assert forgetting.initial_stability(2.0) == 120.0  # clamped


def test_recency_half_life():
    now = 1_000_000.0
    week_ago = now - 168 * 3600
    assert math.isclose(forgetting.recency_score(now, week_ago), 0.5, rel_tol=1e-9)
