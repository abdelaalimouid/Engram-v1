import numpy as np
import pytest

from engram.store import MemoryStore


@pytest.fixture
def store():
    return MemoryStore(":memory:")


def vec():
    v = np.random.rand(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_episode_roundtrip(store):
    episode = store.add_episode("s1", "user", "hello world", 0.5, 48.0, vec())
    loaded = store.episodes()[0]
    assert loaded.id == episode.id
    assert loaded.content == "hello world"
    assert loaded.embedding is not None and loaded.embedding.shape == (512,)


def test_belief_supersession_is_bitemporal(store):
    old = store.add_belief("user", "lives in", "Lisbon", 0.9, None, 48.0, vec())
    new = store.add_belief("user", "lives in", "Ponta Delgada", 0.9, None, 48.0, vec())
    store.supersede_belief(old.id, new.id)

    current = store.beliefs(include_superseded=False)
    assert [b.object for b in current] == ["Ponta Delgada"]

    everything = store.beliefs(include_superseded=True)
    superseded = next(b for b in everything if b.id == old.id)
    assert superseded.valid_to is not None
    assert superseded.superseded_by == new.id  # history preserved, not deleted


def test_current_beliefs_lookup_case_insensitive(store):
    store.add_belief("User", "Prefers", "tea", 0.8, None, 48.0, vec())
    assert len(store.current_beliefs_for("user", "prefers")) == 1


def test_timewarp_moves_clock(store):
    before = store.now()
    store.timewarp(24.0)
    assert store.now() - before >= 24 * 3600 - 1


def test_archive_and_stats(store):
    e1 = store.add_episode("s1", "user", "a", 0.3, 24.0, vec())
    store.add_episode("s1", "user", "b", 0.3, 24.0, vec())
    store.archive_episodes([e1.id])
    stats = store.stats()
    assert stats["episodes_active"] == 1
    assert stats["episodes_archived"] == 1
