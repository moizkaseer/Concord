import asyncio

from concord.llm import MockModelClient
from concord.models import RunState, State, SubgoalSpec
from concord.worker import WorkerPool, model_handler


def spec(id: str, *deps: str) -> SubgoalSpec:
    return SubgoalSpec(id=id, description=f"do {id}", depends_on=list(deps))


# Three independent roots feeding two middle layers and one final subgoal.
EIGHT_SUBGOALS = [
    spec("a"),
    spec("b"),
    spec("c"),
    spec("d", "a", "b"),
    spec("e", "b", "c"),
    spec("f", "d"),
    spec("g", "d", "e"),
    spec("h", "f", "g"),
]


async def test_eight_subgoal_graph_completes_in_order_and_in_parallel(board):
    """Phase 1 exit test."""
    client = MockModelClient({f"Subgoal: do {s.id}": f"result-{s.id}" for s in EIGHT_SUBGOALS})
    respond = model_handler(client)
    running = 0
    peak = 0

    async def handler(subgoal, inputs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await asyncio.sleep(0.05)  # hold the slot long enough for workers to overlap
        running -= 1
        return await respond(subgoal, inputs)

    await board.submit(EIGHT_SUBGOALS)
    status = await WorkerPool(board, handler, size=4, poll_interval=0.01).run(timeout=10)

    assert status.state == RunState.COMPLETED
    assert status.done == 8
    for s in EIGHT_SUBGOALS:
        subgoal = await board.get(s.id)
        assert subgoal.state == State.DONE
        assert subgoal.output == f"result-{s.id}"
        assert subgoal.attempts == 1

    # Dependency order: a subgoal is leased only after all its dependencies are done.
    events = await board.events()
    position = {(e["type"], e.get("id")): n for n, e in enumerate(events)}
    for s in EIGHT_SUBGOALS:
        for dep in s.depends_on:
            assert position[("subgoal.done", dep)] < position[("subgoal.leased", s.id)]

    # Parallelism: the three roots have no dependencies, so they should overlap.
    assert peak >= 3


async def test_handler_receives_outputs_of_its_dependencies(board):
    client = MockModelClient({"Subgoal: do a": "alpha", "Input from a: alpha": "saw alpha"})

    await board.submit([spec("a"), spec("b", "a")])
    await WorkerPool(board, model_handler(client), size=2, poll_interval=0.01).run(timeout=10)

    assert (await board.get("b")).output == "saw alpha"


async def test_handler_error_fails_the_run_and_stops_the_pool(board):
    async def handler(subgoal, inputs):
        raise ValueError("bad input")

    await board.submit([spec("a"), spec("b", "a")])
    status = await WorkerPool(board, handler, size=2, poll_interval=0.01).run(timeout=10)

    assert status.state == RunState.FAILED
    assert (await board.get("a")).error == "ValueError: bad input"
    assert (await board.get("b")).state == State.PENDING


async def test_long_handler_keeps_its_lease_through_heartbeats(board):
    async def handler(subgoal, inputs):
        await asyncio.sleep(0.5)  # longer than the lease, so only heartbeats keep it
        return "done"

    await board.submit([spec("a")])
    pool = WorkerPool(board, handler, size=1, lease_ttl=0.3, poll_interval=0.01)
    status = await pool.run(timeout=10)

    assert status.state == RunState.COMPLETED
    assert (await board.get("a")).attempts == 1
