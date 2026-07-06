from engram.retrieval import RecalledMemory, approx_tokens, pack_budget


def memory(id_, score, tokens):
    return RecalledMemory(
        kind="episode", id=id_, text="x" * tokens * 4, score=score,
        similarity=score, retention=1.0, importance=0.5, tokens=tokens,
    )


def test_pack_respects_budget():
    candidates = [memory(f"m{i}", 0.9 - i * 0.01, 100) for i in range(20)]
    packed = pack_budget(candidates, token_budget=350)
    assert sum(m.tokens for m in packed) <= 350
    assert 1 <= len(packed) <= 3


def test_pack_prefers_score_density():
    dense = memory("dense", 0.5, 10)     # 0.05 score/token
    bulky = memory("bulky", 0.6, 600)    # 0.001 score/token
    packed = pack_budget([dense, bulky], token_budget=100)
    ids = {m.id for m in packed}
    assert "dense" in ids and "bulky" not in ids


def test_best_memory_always_included_even_if_large():
    only = memory("huge", 0.9, 5000)
    packed = pack_budget([only], token_budget=100)
    assert [m.id for m in packed] == ["huge"]


def test_low_scores_filtered_out():
    packed = pack_budget([memory("weak", 0.05, 10)], token_budget=1000)
    assert packed == []


def test_approx_tokens_floor():
    assert approx_tokens("") == 1
    assert approx_tokens("a" * 400) == 100
