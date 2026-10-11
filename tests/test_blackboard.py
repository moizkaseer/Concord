import asyncio

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


async def test_failed_attempt_is_retried_while_attempts_remain(board):
    await board.submit([spec("a")], max_attempts=2)
    await board.lease("w1", ttl=30)

    assert await board.fail("a", "w1", "boom")

    assert (await board.get("a")).state == State.READY
    assert (await board.status()).state == RunState.RUNNING
    retried = [e for e in await board.events() if e["type"] == "subgoal.retried"]
    assert retried[0]["reason"] == "boom"


async def test_subgoal_out_of_attempts_fails_with_everything_that_depends_on_it(board):
    await board.submit([spec("a"), spec("b", "a"), spec("c", "b"), spec("x")], max_attempts=1)
    first = await board.lease("w1", ttl=30)
    second = await board.lease("w1", ttl=30)
    assert {first.id, second.id} == {"a", "x"}

    await board.fail("a", "w1", "boom")
    assert (await board.status()).state == RunState.RUNNING  # x can still finish
    await board.complete("x", "w1", "out")

    assert (await board.get("a")).error == "boom"
    assert (await board.get("b")).error == "dependency a failed"
    assert (await board.get("c")).error == "dependency b failed"
    status = await board.status()
    assert status.state == RunState.FAILED
    assert (status.done, status.failed) == (1, 3)


async def test_expired_lease_is_reclaimed(board):
    await board.submit([spec("a")])
    await board.lease("w1", ttl=0.05)
    assert await board.reclaim_expired() == []  # not expired yet

    await asyncio.sleep(0.1)

    reclaimed = await board.reclaim_expired()
    assert [(r.id, r.worker, r.attempt, r.requeued) for r in reclaimed] == [("a", "w1", 1, True)]
    assert (await board.get("a")).state == State.READY
    assert not await board.complete("a", "w1", "late")  # the old worker lost the lease


async def test_supervised_run_holds_subgoals_until_dispatched(board):
    await board.submit([spec("a")], supervised=True)

    assert (await board.get("a")).state == State.CHECKING
    assert await board.lease("w1", ttl=30) is None

    assert await board.dispatch("a")
    assert not await board.dispatch("a")  # a second dispatch does nothing
    assert (await board.lease("w1", ttl=30)).id == "a"


async def test_reuse_finishes_a_waiting_subgoal_with_another_ones_output(board):
    await board.submit([spec("a"), spec("b"), spec("c", "b")], supervised=True)
    assert not await board.reuse("b", "a", "exact match")  # a has no output yet
    await board.dispatch("a")
    await board.lease("w1", ttl=30)
    await board.complete("a", "w1", "output of a")

    assert await board.reuse("b", "a", "exact match")

    reused = await board.get("b")
    assert (reused.state, reused.output, reused.reused_from) == (State.DONE, "output of a", "a")
    assert (await board.get("c")).state == State.CHECKING  # b's dependent is released
    assert (await board.interventions())[0]["type"] == "supervisor.reused"


async def test_complete_adds_tokens_to_the_run(board):
    await board.submit([spec("a")])
    await board.lease("w1", ttl=30)

    await board.complete("a", "w1", "out", tokens=42)

    assert (await board.status()).tokens == 42


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
