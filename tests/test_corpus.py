from concord.corpus import Corpus, Passage


def test_files_are_split_into_one_passage_per_section(tmp_path):
    doc = tmp_path / "guide.md"
    doc.write_text(
        "Intro line.\n\n# Leases\nA lease expires.\n\n## Heartbeats\nSent often.\n\n# Empty\n"
    )

    corpus = Corpus.from_files([doc])

    assert len(corpus) == 3
    assert corpus.search("lease")[0] == Passage("guide.md § Leases", "A lease expires.")
    assert corpus.search("intro")[0].source == "guide.md"


def test_search_ranks_by_relevance_and_limits_results():
    corpus = Corpus(
        [
            Passage("a", "The supervisor requeues a subgoal when its lease expires."),
            Passage("b", "Leases use the Redis clock. A lease is renewed by heartbeats."),
            Passage("c", "Spans are exported to Jaeger."),
        ]
    )

    assert [p.source for p in corpus.search("lease heartbeats")] == ["b", "a"]
    assert [p.source for p in corpus.search("lease heartbeats", k=1)] == ["b"]
    assert corpus.search("kubernetes") == []


def test_rare_words_count_for_more_than_common_ones():
    corpus = Corpus(
        [
            Passage("common", "worker worker worker worker"),
            Passage("rare", "worker jaeger"),
            Passage("other", "worker"),
        ]
    )

    assert corpus.search("worker jaeger")[0].source == "rare"


def test_empty_corpus_finds_nothing():
    assert Corpus([]).search("anything") == []
