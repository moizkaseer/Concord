import pytest

from concord.graph import GraphError, validate_graph
from concord.models import SubgoalSpec


def spec(id: str, *deps: str) -> SubgoalSpec:
    return SubgoalSpec(id=id, description=f"do {id}", depends_on=list(deps))


def test_returns_depth_of_longest_chain():
    graph = [spec("a"), spec("b"), spec("c", "a", "b"), spec("d", "c")]

    assert validate_graph(graph) == 3


def test_rejects_empty_graph():
    with pytest.raises(GraphError, match="no subgoals"):
        validate_graph([])


def test_rejects_duplicate_ids():
    with pytest.raises(GraphError, match="duplicate"):
        validate_graph([spec("a"), spec("a")])


def test_rejects_unknown_dependency():
    with pytest.raises(GraphError, match="unknown subgoal 'x'"):
        validate_graph([spec("a", "x")])


def test_rejects_self_dependency():
    with pytest.raises(GraphError, match="depends on itself"):
        validate_graph([spec("a", "a")])


def test_rejects_repeated_dependency():
    with pytest.raises(GraphError, match="twice"):
        validate_graph([spec("a"), spec("b", "a", "a")])


def test_rejects_cycle_and_names_its_members():
    graph = [spec("a"), spec("b", "a", "d"), spec("c", "b"), spec("d", "c")]

    with pytest.raises(GraphError, match=r"cycle.*\['b', 'c', 'd'\]"):
        validate_graph(graph)


def test_rejects_graph_over_size_limit():
    with pytest.raises(GraphError, match="limit is 2"):
        validate_graph([spec("a"), spec("b"), spec("c")], max_subgoals=2)


def test_rejects_graph_over_depth_limit():
    with pytest.raises(GraphError, match="depth is 3"):
        validate_graph([spec("a"), spec("b", "a"), spec("c", "b")], max_depth=2)
