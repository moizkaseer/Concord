"""Validation of a submitted subgoal graph."""

from collections import deque
from collections.abc import Sequence

from concord.models import SubgoalSpec

DEFAULT_MAX_SUBGOALS = 20
DEFAULT_MAX_DEPTH = 6


class GraphError(ValueError):
    """The submitted graph breaks a structural rule."""


def validate_graph(
    specs: Sequence[SubgoalSpec],
    *,
    max_subgoals: int = DEFAULT_MAX_SUBGOALS,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> int:
    """Check the graph is a bounded DAG and return its depth (longest dependency chain)."""
    if not specs:
        raise GraphError("graph has no subgoals")
    if len(specs) > max_subgoals:
        raise GraphError(f"graph has {len(specs)} subgoals, limit is {max_subgoals}")

    ids = [s.id for s in specs]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise GraphError(f"duplicate subgoal ids: {duplicates}")

    known = set(ids)
    dependents: dict[str, list[str]] = {i: [] for i in ids}
    unmet: dict[str, int] = {}
    for spec in specs:
        if len(set(spec.depends_on)) != len(spec.depends_on):
            raise GraphError(f"subgoal {spec.id!r} lists a dependency twice")
        for dep in spec.depends_on:
            if dep == spec.id:
                raise GraphError(f"subgoal {spec.id!r} depends on itself")
            if dep not in known:
                raise GraphError(f"subgoal {spec.id!r} depends on unknown subgoal {dep!r}")
            dependents[dep].append(spec.id)
        unmet[spec.id] = len(spec.depends_on)

    # Kahn's algorithm: anything never reached sits on a cycle.
    depth = {i: 1 for i in ids if unmet[i] == 0}
    queue = deque(depth)
    while queue:
        current = queue.popleft()
        for child in dependents[current]:
            depth[child] = max(depth.get(child, 0), depth[current] + 1)
            unmet[child] -= 1
            if unmet[child] == 0:
                queue.append(child)

    stuck = sorted(i for i in ids if unmet[i] > 0)
    if stuck:
        raise GraphError(f"dependency cycle among subgoals: {stuck}")

    graph_depth = max(depth.values())
    if graph_depth > max_depth:
        raise GraphError(f"graph depth is {graph_depth}, limit is {max_depth}")
    return graph_depth
