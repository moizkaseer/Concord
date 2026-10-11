import pytest

from concord.blackboard import RunExistsError
from concord.graph import GraphError
from concord.models import RunState, State, SubgoalSpec


def spec(id: str, *deps: str) -> SubgoalSpec:
    return SubgoalSpec(id=id, description=f"do {id}", depends_on=list(deps))


async def test_submit_marks_only_root_subgoals_ready(board):
    await board.submit([spec("a"), spec("b", "a")])

    assert (await board.get("a")).state == State.READY
    assert (await board.get("b")).state == State.PENDING
    assert (await board.status()).state == RunState.RUNNING


async def test_submit_rejects_invalid_graph_and_stores_nothing(board):
    with pytest.raises(GraphError):
        await board.submit([spec("a", "missing")])

    with pytest.raises(KeyError):
        await board.status()


async def test_submit_twice_is_rejected(board):
    await board.submit([spec("a")])

    with pytest.raises(RunExistsError):
        await board.submit([spec("a")])


async def test_lease_hands_each_subgoal_to_one_worker(board):
    await board.submit([spec("a")])

    first = await board.lease("w1", ttl=30)
    second = await board.lease("w2", ttl=30)

    assert first.id == "a"
    assert first.state == State.LEASED
    assert first.lease_owner == "w1"
    assert first.attempts == 1
    assert second is None


async def test_heartbeat_extends_lease_for_owner_only(board):
    await board.submit([spec("a")])
    leased = await board.lease("w1", ttl=1)

    assert await board.heartbeat("a", "w1", ttl=60)
    assert not await board.heartbeat("a", "w2", ttl=60)
    assert (await board.get("a")).lease_expires_at > leased.lease_expires_at


async def test_complete_is_rejected_for_a_worker_without_the_lease(board):
    await board.submit([spec("a")])
    await board.lease("w1", ttl=30)

    assert not await board.complete("a", "w2", "stolen")
    assert (await board.get("a")).state == State.LEASED


async def test_dependent_becomes_ready_only_when_all_dependencies_are_done(board):
    await board.submit([spec("a"), spec("b"), spec("c", "a", "b")])
    first = await board.lease("w1", ttl=30)
    second = await board.lease("w1", ttl=30)

    await board.complete(first.id, "w1", "out")
    assert (await board.get("c")).state == State.PENDING

    await board.complete(second.id, "w1", "out")
    assert (await board.get("c")).state == State.READY


async def test_run_completes_when_every_subgoal_is_done(board):
    await board.submit([spec("a"), spec("b", "a")])
    for _ in range(2):
        leased = await board.lease("w1", ttl=30)
        await board.complete(leased.id, "w1", f"output of {leased.id}")

    status = await board.status()
    assert status.state == RunState.COMPLETED
    assert status.done == 2
    assert await board.outputs(["a", "b"]) == {"a": "output of a", "b": "output of b"}


async def test_fail_marks_subgoal_and_run_failed(board):
    await board.submit([spec("a")])
    await board.lease("w1", ttl=30)

    assert await board.fail("a", "w1", "boom")

    failed = await board.get("a")
    assert failed.state == State.FAILED
    assert failed.error == "boom"
    assert (await board.status()).state == RunState.FAILED


async def test_every_state_change_is_logged_in_order(board):
    await board.submit([spec("a"), spec("b", "a")])
    for _ in range(2):
        leased = await board.lease("w1", ttl=30)
        await board.complete(leased.id, "w1", "out")

    events = [(e["type"], e.get("id")) for e in await board.events()]

    assert events == [
        ("run.submitted", None),
        ("subgoal.ready", "a"),
        ("subgoal.leased", "a"),
        ("subgoal.done", "a"),
        ("subgoal.ready", "b"),
        ("subgoal.leased", "b"),
        ("subgoal.done", "b"),
        ("run.completed", None),
    ]
