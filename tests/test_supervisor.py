import asyncio
import contextlib

from concord.dedupe import HashingEmbedder
from concord.models import RunState, State, SubgoalSpec
from concord.supervisor import GROUP, Supervisor
from concord.worker import WorkerPool, WorkResult


def spec(id: str, description: str | None = None, *deps: str) -> SubgoalSpec:
    return SubgoalSpec(id=id, description=description or f"do {id}", depends_on=list(deps))


class Recorder:
    """A handler that records which subgoals reached a worker."""

    def __init__(self, tokens: int = 0) -> None:
        self.calls: list[str] = []
        self._tokens = tokens

    async def __call__(self, subgoal, inputs) -> WorkResult:
        self.calls.append(subgoal.id)
        await asyncio.sleep(0.02)
        return WorkResult(f"result-{subgoal.id}", self._tokens)


async def run_supervised(board, handler, **supervisor_options):
    pool = WorkerPool(board, handler, size=4, poll_interval=0.01)
    supervisor = Supervisor(board, tick=0.05, **supervisor_options)
    async with asyncio.timeout(15):
        status, _ = await asyncio.gather(pool.run(), supervisor.run())
    return status


async def test_run_completes_after_workers_die_mid_subgoal(board):
    """Phase 2 exit test, part one."""
    graph = [
        spec("a"),
        spec("b"),
        spec("c"),
        spec("d", None, "a", "b"),
        spec("e", None, "b", "c"),
        spec("f", None, "d"),
        spec("g", None, "d", "e"),
        spec("h", None, "f", "g"),
    ]
    reached_d = asyncio.Event()

    async def doomed(subgoal, inputs):
        if subgoal.id == "d":
            reached_d.set()
            await asyncio.sleep(3600)
        return WorkResult(f"result-{subgoal.id}")

    await board.submit(graph, supervised=True)
    supervisor = asyncio.create_task(Supervisor(board, tick=0.05).run())
    first_pool = asyncio.create_task(
        WorkerPool(board, doomed, size=4, lease_ttl=0.3, poll_interval=0.01).run()
    )
    async with asyncio.timeout(5):
        await reached_d.wait()

    # Kill the whole pool while it holds d: no completion, no failure report, and the
    # heartbeats stop, exactly as if the process had died.
    first_pool.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await first_pool

    survivors = WorkerPool(board, Recorder(), size=4, lease_ttl=0.3, poll_interval=0.01)
    status = await survivors.run(timeout=10)
    async with asyncio.timeout(5):
        await supervisor

    assert status.state == RunState.COMPLETED
    assert status.done == 8
    assert (await board.get("d")).attempts == 2
    requeued = [e for e in await board.interventions() if e["type"] == "supervisor.requeued"]
    assert {"id": "d", "reason": "lease expired"}.items() <= requeued[0].items()


async def test_duplicated_subgoal_is_served_from_the_existing_output(board):
    """Phase 2 exit test, part two."""
    handler = Recorder()
    await board.submit(
        [
            spec("first", "Find sources on solar power"),
            spec("second", "find sources on solar power."),
            spec("answer", "Write the answer", "first", "second"),
        ],
        supervised=True,
    )

    status = await run_supervised(board, handler)

    assert status.state == RunState.COMPLETED
    assert sorted(handler.calls) == ["answer", "first"]
    second = await board.get("second")
    assert second.reused_from == "first"
    assert second.output == "result-first"
    assert second.attempts == 0
    kinds = [e["type"] for e in await board.interventions() if e.get("id") == "second"]
    assert kinds == ["supervisor.held", "supervisor.reused"]


async def test_every_duplicate_of_a_subgoal_is_reused(board):
    handler = Recorder()
    await board.submit(
        [
            spec("first", "List the key facts"),
            spec("second", "list the key facts"),
            spec("third", "LIST THE KEY FACTS!"),
            spec("check", "Check the facts", "first"),
        ],
        supervised=True,
    )

    status = await run_supervised(board, handler)

    assert status.state == RunState.COMPLETED
    assert sorted(handler.calls) == ["check", "first"]
    reused = [e for e in await board.interventions() if e["type"] == "supervisor.reused"]
    assert {e["id"] for e in reused} == {"second", "third"}
    assert all(e["source"] == "first" and e["reason"] == "exact match" for e in reused)


async def test_work_that_depends_on_duplicates_is_also_recognised_as_duplicate(board):
    handler = Recorder()
    await board.submit(
        [
            spec("search", "Find sources"),
            spec("search-2", "find sources"),
            spec("summary", "Summarize the sources", "search"),
            spec("summary-2", "Summarize the sources", "search-2"),
        ],
        supervised=True,
    )

    status = await run_supervised(board, handler)

    assert status.state == RunState.COMPLETED
    assert sorted(handler.calls) == ["search", "summary"]
    assert (await board.get("summary-2")).reused_from == "summary"


async def test_same_instruction_on_different_inputs_is_not_a_duplicate(board):
    handler = Recorder()
    await board.submit(
        [
            spec("a"),
            spec("b"),
            spec("sum-a", "Summarize the input", "a"),
            spec("sum-b", "Summarize the input", "b"),
        ],
        supervised=True,
    )

    status = await run_supervised(board, handler)

    assert status.state == RunState.COMPLETED
    assert sorted(handler.calls) == ["a", "b", "sum-a", "sum-b"]
    assert await board.interventions() == []


async def test_reworded_duplicate_is_reused_when_similar_enough(board):
    handler = Recorder()
    await board.submit(
        [
            spec("first", "Find recent sources on solar power"),
            spec("second", "Find recent sources about solar power"),
            spec("other", "Write the final answer"),
        ],
        supervised=True,
    )

    status = await run_supervised(
        board, handler, embedder=HashingEmbedder(), similarity_threshold=0.8
    )

    assert status.state == RunState.COMPLETED
    assert sorted(handler.calls) == ["first", "other"]
    reused = [e for e in await board.interventions() if e["type"] == "supervisor.reused"]
    assert reused[0]["id"] == "second"
    assert reused[0]["reason"] == "similarity 0.83"


async def test_reworded_subgoal_below_the_threshold_runs_normally(board):
    handler = Recorder()
    await board.submit(
        [
            spec("first", "Find recent sources on solar power"),
            spec("second", "Find recent sources about solar power"),
        ],
        supervised=True,
    )

    await run_supervised(board, handler, embedder=HashingEmbedder(), similarity_threshold=0.9)

    assert sorted(handler.calls) == ["first", "second"]


async def test_duplicate_runs_itself_when_the_original_fails(board):
    calls: list[str] = []

    async def handler(subgoal, inputs):
        calls.append(subgoal.id)
        await asyncio.sleep(0.02)
        if subgoal.id == "first":
            raise ValueError("no luck")
        return WorkResult("ok")

    await board.submit(
        [spec("first", "Find sources"), spec("second", "find sources")],
        supervised=True,
        max_attempts=1,
    )

    status = await run_supervised(board, handler)

    assert status.state == RunState.FAILED
    assert (await board.get("first")).state == State.FAILED
    assert (await board.get("second")).state == State.DONE
    assert calls == ["first", "second"]


async def test_run_is_halted_when_it_goes_over_its_token_budget(board):
    handler = Recorder(tokens=10)
    await board.submit([spec("a"), spec("b", None, "a"), spec("c", None, "b")], supervised=True)

    status = await run_supervised(board, handler, token_budget=15)

    assert status.state == RunState.FAILED
    assert status.reason == "token budget exceeded: used 20 of 15"
    assert handler.calls == ["a", "b"]
    assert (await board.get("c")).state == State.CHECKING
    halted = [e for e in await board.interventions() if e["type"] == "supervisor.halted"]
    assert len(halted) == 1


async def test_restarted_supervisor_handles_events_the_last_one_never_acknowledged(board):
    handler = Recorder()
    await board.submit([spec("a"), spec("b", None, "a")], supervised=True)

    # A supervisor that received the first events and died before acknowledging them.
    await board.ensure_group(GROUP)
    lost = await board.read_events(GROUP, "supervisor")
    assert [e["type"] for e in lost] == ["run.submitted", "subgoal.checking"]

    status = await run_supervised(board, handler)

    assert status.state == RunState.COMPLETED
    assert handler.calls == ["a", "b"]


async def test_supervisor_started_late_catches_up_from_the_first_event(board):
    handler = Recorder()
    await board.submit([spec("a"), spec("b", None, "a")], supervised=True)
    pool = asyncio.create_task(WorkerPool(board, handler, size=2, poll_interval=0.01).run())
    await asyncio.sleep(0.2)
    assert handler.calls == []  # nothing is dispatched without a supervisor

    async with asyncio.timeout(10):
        await Supervisor(board, tick=0.05).run()
        status = await pool

    assert status.state == RunState.COMPLETED
