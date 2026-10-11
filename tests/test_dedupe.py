import pytest

from concord.dedupe import HashingEmbedder, cosine, duplicate_key, normalize


def test_normalize_ignores_case_punctuation_and_spacing():
    assert normalize("  Find sources,  on Solar-Power! ") == "find sources on solar power"


def test_duplicate_key_matches_trivially_reworded_descriptions():
    assert duplicate_key("Find sources.", ["a"]) == duplicate_key("find  sources", ["a"])


def test_duplicate_key_ignores_dependency_order():
    assert duplicate_key("merge", ["a", "b"]) == duplicate_key("merge", ["b", "a"])


def test_duplicate_key_differs_when_dependencies_differ():
    assert duplicate_key("summarize", ["a"]) != duplicate_key("summarize", ["b"])


def test_cosine_of_identical_and_unrelated_vectors():
    assert cosine([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


async def test_hashing_embedder_scores_shared_words():
    embedder = HashingEmbedder(dimensions=4096)  # large enough that these words do not collide
    same, close, far = await embedder.embed(
        [
            "power solar on sources recent find",
            "find recent sources about solar power",
            "write the final answer",
        ]
    )
    base = (await embedder.embed(["find recent sources on solar power"]))[0]

    assert cosine(base, same) == pytest.approx(1.0)  # word order does not matter
    assert cosine(base, close) == pytest.approx(5 / 6)  # five of six words shared
    assert cosine(base, far) == 0.0
