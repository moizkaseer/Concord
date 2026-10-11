import asyncio
import contextlib

import pytest

from concord.dedupe import HashingEmbedder
from concord.drift import DriftVerdict, Outcome
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


class ScriptedChecker:
    """Gives the verdicts it was told to give, in order, per subgoal; then on-task."""

    def __init__(self, verdicts: dict[str, list[DriftVerdict]] | None = None) -> None:
        self._verdicts = verdicts or {}
        self.seen: list[tuple[str, str]] = []

    async def check(self, subgoal, inputs, output) -> DriftVerdict:
        self.seen.append((subgoal.id, output))
        queue = self._verdicts.get(subgoal.id)
        return queue.pop(0) if queue else DriftVerdict(Outcome.ON_TASK, tokens=5)


DRIFTED = DriftVerdict(Outcome.DRIFTED, "it is about wind, not solar", tokens=7)


class Attempts:
    """A handler whose output names the attempt, and which records the feedback it got."""

    def __init__(self) -> None:
        self.feedback: list[tuple[str, str | None]] = []

    async def __call__(self, subgoal, inputs) -> WorkResult:
        self.feedback.append((subgoal.id, subgoal.feedback))
        return WorkResult(f"{subgoal.id} attempt {subgoal.attempts}", 10)


async def test_drifted_output_is_sent_back_and_the_retry_is_used(board):
    await board.submit([spec("a"), spec("b", None, "a")], supervised=True, review=True)
    handler = Attempts()
    checker = ScriptedChecker({"a": [DRIFTED]})

    status = await run_supervised(board, handler, checker=checker)

    assert status.state == RunState.COMPLETED
    a = await board.get("a")
    assert (a.output, a.attempts, a.rejections) == ("a attempt 2", 2, 1)
    # The retry was told why the first output was sent back.
    assert handler.feedback[:2] == [("a", None), ("a", "it is about wind, not solar")]
    flagged = [e for e in await board.interventions() if e["type"] == "supervisor.flagged"]
    assert [(e["id"], e["action"], e["reason"]) for e in flagged] == [
        ("a", "retried", "it is about wind, not solar")
    ]
    # Three worker calls at 10 tokens, plus the reviews: 7 + 5 for a, 5 for b.
    assert status.tokens == 30 + 17


async def test_nothing_downstream_sees_an_output_before_it_is_accepted(board):
    await board.submit([spec("a"), spec("b", None, "a")], supervised=True, review=True)
    seen: dict[str, dict[str, str]] = {}

    async def handler(subgoal, inputs):
        seen[subgoal.id] = inputs
        return WorkResult(f"{subgoal.id} attempt {subgoal.attempts}")

    await run_supervised(board, handler, checker=ScriptedChecker({"a": [DRIFTED]}))

    assert seen["b"] == {"a": "a attempt 2"}


async def test_a_second_drifted_output_is_kept_and_flagged_rather_than_retried_again(board):
    await board.submit([spec("a")], supervised=True, review=True)
    checker = ScriptedChecker({"a": [DRIFTED, DRIFTED]})

    status = await run_supervised(board, Attempts(), checker=checker)

    assert status.state == RunState.COMPLETED
    assert (await board.get("a")).output == "a attempt 2"
    flagged = [e for e in await board.interventions() if e["type"] == "supervisor.flagged"]
    assert [e["action"] for e in flagged] == ["retried", "kept"]


async def test_drifted_output_fails_the_subgoal_when_it_is_out_of_attempts(board):
    await board.submit([spec("a"), spec("b", None, "a")], supervised=True, review=True)
    checker = ScriptedChecker({"a": [DRIFTED, DRIFTED, DRIFTED]})

    status = await run_supervised(board, Attempts(), checker=checker, max_rejections=5)

    assert status.state == RunState.FAILED
    a = await board.get("a")
    assert a.state == State.FAILED and a.attempts == 3
    assert a.error == "off-task output: it is about wind, not solar"
    assert a.output is None
    assert (await board.get("b")).state == State.FAILED


async def test_output_for_a_vague_subgoal_is_kept_and_marked_for_clarification(board):
    await board.submit([spec("a")], supervised=True, review=True)
    unclear = DriftVerdict(Outcome.UNCLEAR, "the subgoal does not say what to look for")

    status = await run_supervised(board, Attempts(), checker=ScriptedChecker({"a": [unclear]}))

    assert status.state == RunState.COMPLETED
    assert (await board.get("a")).attempts == 1
    types = [e["type"] for e in await board.interventions()]
    assert types == ["supervisor.needs_clarification"]


async def test_run_continues_when_the_reviewer_itself_fails(board):
    await board.submit([spec("a"), spec("b", None, "a")], supervised=True, review=True)

    class Broken:
        async def check(self, subgoal, inputs, output):
            raise RuntimeError("model is down")

    status = await run_supervised(board, Attempts(), checker=Broken())

    assert status.state == RunState.COMPLETED
    skipped = await board.interventions()
    assert {e["type"] for e in skipped} == {"supervisor.review_skipped"}
    assert "model is down" in skipped[0]["reason"]


async def test_reused_duplicate_is_not_reviewed_a_second_time(board):
    graph = [spec("a", "look up solar"), spec("b", "Look up solar."), spec("c", None, "a", "b")]
    await board.submit(graph, supervised=True, review=True)
    checker = ScriptedChecker()

    status = await run_supervised(board, Recorder(), checker=checker)

    assert status.state == RunState.COMPLETED
    assert sorted(subgoal_id for subgoal_id, _ in checker.seen) == ["a", "c"]
    assert (await board.get("b")).reused_from == "a"


async def test_slow_reviews_run_side_by_side_and_do_not_delay_lease_reclaim(board):
    await board.submit([spec("a"), spec("b"), spec("c")], supervised=True, review=True)
    in_review = 0
    most_at_once = 0

    class Slow:
        async def check(self, subgoal, inputs, output):
            nonlocal in_review, most_at_once
            in_review += 1
            most_at_once = max(most_at_once, in_review)
            await asyncio.sleep(0.3)
            in_review -= 1
            return DriftVerdict(Outcome.ON_TASK)

    status = await run_supervised(board, Recorder(), checker=Slow())

    assert status.state == RunState.COMPLETED
    assert most_at_once == 3


async def test_restarted_supervisor_reviews_an_output_the_last_one_left_waiting(board):
    await board.submit([spec("a"), spec("b", None, "a")], supervised=True, review=True)
    started = asyncio.Event()

    class Stuck:
        async def check(self, subgoal, inputs, output):
            started.set()
            await asyncio.sleep(3600)

    pool = asyncio.create_task(WorkerPool(board, Recorder(), size=2, poll_interval=0.01).run())
    first = asyncio.create_task(Supervisor(board, tick=0.05, checker=Stuck()).run())
    async with asyncio.timeout(5):
        await started.wait()
    first.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await first
    assert (await board.get("a")).state == State.REVIEWING

    async with asyncio.timeout(10):
        await Supervisor(board, tick=0.05, checker=ScriptedChecker()).run()
        status = await pool

    assert status.state == RunState.COMPLETED


async def test_review_needs_a_supervised_run(board):
    with pytest.raises(ValueError, match="supervised"):
        await board.submit([spec("a")], review=True)
