import asyncio
import contextlib
import json

import pytest

from concord.graph import GraphError
from concord.llm import Completion
from concord.models import RunState, State, SubgoalSpec
from concord.replanner import ModelReplanner, Patch, ReplanContext, check_patch
from concord.supervisor import Supervisor
from concord.worker import WorkerPool, WorkResult


def spec(id: str, *deps: str, description: str | None = None) -> SubgoalSpec:
    return SubgoalSpec(id=id, description=description or f"do {id}", depends_on=list(deps))


# Two searches feed a comparison, which feeds the answer. "b" can never succeed.
GRAPH = [spec("a"), spec("b"), spec("compare", "a", "b"), spec("answer", "compare")]


class Handler:
    """Fails every attempt at the subgoals in `broken`; records what every attempt saw."""

    def __init__(self, *broken: str) -> None:
        self.broken = set(broken)
        self.calls: list[str] = []
        self.inputs: dict[str, dict[str, str]] = {}

    async def __call__(self, subgoal, inputs) -> WorkResult:
        self.calls.append(subgoal.id)
        self.inputs[subgoal.id] = inputs
        await asyncio.sleep(0.01)
        if subgoal.id in self.broken:
            raise RuntimeError("source unreachable")
        return WorkResult(f"result-{subgoal.id}", tokens=1)


class Fixed:
    """A re-planner that always proposes the same patch, and records what it was shown."""

    def __init__(self, patch: Patch) -> None:
        self.patch = patch
        self.contexts: list[ReplanContext] = []

    async def replan(self, context: ReplanContext) -> Patch:
        self.contexts.append(context)
        return self.patch


INSTEAD_OF_B = Patch(
    add=[spec("b2", description="do b from another source")],
    replace_with=["b2"],
    reason="the source b used cannot be reached",
)


async def run(board, handler, replanner=None, **supervisor_options):
    pool = WorkerPool(board, handler, size=2, poll_interval=0.01)
    supervisor = Supervisor(board, replanner=replanner, tick=0.05, **supervisor_options)
    async with asyncio.timeout(15):
        status, _ = await asyncio.gather(pool.run(), supervisor.run())
    return status


async def test_run_with_a_permanently_failing_subgoal_recovers_through_a_replan(board):
    """Phase 8 exit test."""
    handler = Handler("b")
    replanner = Fixed(INSTEAD_OF_B)
    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=2)

    status = await run(board, handler, replanner)

    assert status.state == RunState.COMPLETED
    assert (status.done, status.failed, status.replaced, status.total) == (4, 0, 1, 5)
    b = await board.get("b")
    assert (b.state, b.replaced_by, b.error) == (
        State.REPLACED,
        ["b2"],
        "RuntimeError: source unreachable",
    )
    b2 = await board.get("b2")
    assert (b2.state, b2.replaces, b2.output) == (State.DONE, "b", "result-b2")
    assert (await board.get("compare")).depends_on == ["a", "b2"]
    assert handler.inputs["compare"] == {"a": "result-a", "b2": "result-b2"}
    assert handler.calls.count("b") == 2  # every attempt, then the re-plan
    # The re-planner saw the failure and what had finished.
    (context,) = replanner.contexts
    assert context.failed.id == "b" and context.outputs == {"a": "result-a"}
    (replanned,) = [e for e in await board.interventions() if e["type"] == "supervisor.replanned"]
    assert (replanned["id"], replanned["added"], replanned["reason"]) == (
        "b",
        "b2",
        "the source b used cannot be reached",
    )


async def test_without_replans_the_same_run_fails_as_before(board):
    await board.submit(GRAPH, supervised=True, max_attempts=2)

    status = await run(board, Handler("b"), Fixed(INSTEAD_OF_B))

    assert status.state == RunState.FAILED
    assert (await board.get("b")).state == State.FAILED
    assert (await board.get("answer")).state == State.FAILED


async def test_replans_are_bounded_per_run(board):
    # The replacement fails too, and the run has no second re-plan.
    handler = Handler("b", "b2")
    replanner = Fixed(INSTEAD_OF_B)
    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=2)

    status = await run(board, handler, replanner)

    assert status.state == RunState.FAILED
    assert len(replanner.contexts) == 1
    assert (await board.get("b2")).state == State.FAILED
    assert (await board.get("answer")).state == State.FAILED


async def test_replacement_that_fails_again_is_replanned_while_budget_lasts(board):
    second = Patch([spec("b3", description="do b a third way")], ["b3"], "b2 failed as well")

    class Twice:
        patches = [INSTEAD_OF_B, second]

        async def replan(self, context):
            return self.patches.pop(0)

    await board.submit(GRAPH, supervised=True, replans=2, max_attempts=1)

    status = await run(board, Handler("b", "b2"), Twice())

    assert status.state == RunState.COMPLETED
    assert status.replaced == 2
    assert (await board.get("compare")).depends_on == ["a", "b3"]


async def test_failing_replanner_fails_the_subgoal_on_record(board):
    class Broken:
        async def replan(self, context):
            raise TimeoutError("model did not answer")

    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=1)

    status = await run(board, Handler("b"), Broken())

    assert status.state == RunState.FAILED
    b = await board.get("b")
    assert (b.state, b.error) == (State.FAILED, "RuntimeError: source unreachable")
    (gave_up,) = [e for e in await board.interventions() if e["type"] == "supervisor.replan_failed"]
    assert gave_up["reason"] == "re-plan failed: TimeoutError: model did not answer"


async def test_no_replanner_fails_the_subgoal(board):
    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=1)

    status = await run(board, Handler("b"), replanner=None)

    assert status.state == RunState.FAILED
    (gave_up,) = [e for e in await board.interventions() if e["type"] == "supervisor.replan_failed"]
    assert gave_up["reason"] == "no re-planner"


async def test_patch_that_would_make_a_cycle_is_rejected_before_it_is_applied(board):
    # The new subgoal reads the comparison, which would then depend on it.
    cyclic = Patch([spec("b2", "compare")], ["b2"], "loop")
    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=1)

    status = await run(board, Handler("b"), Fixed(cyclic))

    assert status.state == RunState.FAILED
    assert [s.id for s in await board.graph()] == ["a", "b", "compare", "answer"]
    (gave_up,) = [e for e in await board.interventions() if e["type"] == "supervisor.replan_failed"]
    assert "cycle" in gave_up["reason"]


async def test_dependents_can_go_ahead_without_the_failed_subgoal(board):
    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=1)
    handler = Handler("b")

    status = await run(board, handler, Fixed(Patch([], [], "a alone is enough")))

    assert status.state == RunState.COMPLETED
    assert (status.done, status.replaced, status.total) == (3, 1, 4)
    assert handler.inputs["compare"] == {"a": "result-a"}


async def test_final_subgoal_is_replaced_by_one_that_produces_the_result(board):
    patch = Patch(
        [spec("answer2", "compare", description="write the answer again")], ["answer2"], "retry"
    )
    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=1)

    status = await run(board, Handler("answer"), Fixed(patch))

    assert status.state == RunState.COMPLETED
    assert (await board.get("answer")).replaced_by == ["answer2"]
    assert (await board.get("answer2")).output == "result-answer2"


async def test_replanned_subgoal_reuses_nothing_from_the_one_it_replaced(board):
    # Same description as the failed subgoal: the duplicate check must not hold it
    # waiting on a subgoal that will never finish.
    same = Patch([spec("b2", description="do b")], ["b2"], "try again")
    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=1)
    handler = Handler("b")

    status = await run(board, handler, Fixed(same))

    assert status.state == RunState.COMPLETED
    assert "b2" in handler.calls


async def test_restarted_supervisor_finishes_a_replan_the_last_one_left(board):
    class Hangs:
        async def replan(self, context):
            await asyncio.sleep(3600)

    await board.submit(GRAPH, supervised=True, replans=1, max_attempts=1)
    pool = asyncio.create_task(WorkerPool(board, Handler("b"), size=2, poll_interval=0.01).run())
    first = asyncio.create_task(Supervisor(board, replanner=Hangs(), tick=0.05).run())
    async with asyncio.timeout(10):
        while (await board.get("b")).state != State.REPLANNING:
            await asyncio.sleep(0.01)
    first.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await first

    async with asyncio.timeout(10):
        await Supervisor(board, replanner=Fixed(INSTEAD_OF_B), tick=0.05).run()
        status = await pool

    assert status.state == RunState.COMPLETED
    assert (await board.get("b2")).state == State.DONE


async def test_replanning_needs_a_supervised_run(board):
    with pytest.raises(ValueError, match="supervised"):
        await board.submit(GRAPH, replans=1)


# --- check_patch ------------------------------------------------------------------------


async def _context(board, failed: str) -> ReplanContext:
    graph = await board.graph()
    return ReplanContext(next(s for s in graph if s.id == failed), graph)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (Patch([spec("a")], ["a"], ""), "reuse existing ids"),
        (Patch([spec("b2", "b")], ["b2"], ""), "depends on failed subgoals"),
        (Patch([], ["nowhere"], ""), "unknown subgoals"),
        (Patch([], ["b"], ""), "names failed subgoals"),
        (Patch([spec("b2", "answer")], ["b2"], ""), "cycle"),
    ],
)
async def test_check_patch_rejects(board, patch, message):
    await board.submit(GRAPH)

    with pytest.raises(GraphError, match=message):
        check_patch(await _context(board, "b"), patch, max_subgoals=20, max_depth=6)


async def test_check_patch_needs_one_stand_in_for_the_final_subgoal(board):
    await board.submit(GRAPH)

    with pytest.raises(GraphError, match="exactly one"):
        check_patch(
            await _context(board, "answer"), Patch([], [], ""), max_subgoals=20, max_depth=6
        )


async def test_check_patch_enforces_the_size_limit(board):
    await board.submit(GRAPH)
    many = Patch([spec(f"n{i}") for i in range(4)], [f"n{i}" for i in range(4)], "")

    with pytest.raises(GraphError, match="limit"):
        check_patch(await _context(board, "b"), many, max_subgoals=6, max_depth=6)


# --- ModelReplanner ---------------------------------------------------------------------


class Scripted:
    def __init__(self, *replies: dict) -> None:
        self._replies = [json.dumps(r) for r in replies]
        self.prompts: list[str] = []

    async def complete(self, prompt, *, system=None, schema=None, web_search=False):
        self.prompts.append(prompt)
        return Completion(self._replies.pop(0), 10, 5)


def _draft(id: str, *deps: str) -> dict:
    return {
        "id": id,
        "specialist": "research",
        "description": f"find {id}",
        "depends_on": list(deps),
    }


async def test_model_replanner_fixes_a_rejected_patch_once(board):
    await board.submit(GRAPH)
    client = Scripted(
        {"reason": "loop", "subgoals": [_draft("b2", "compare")], "replace_with": ["b2"]},
        {"reason": "another source", "subgoals": [_draft("b2")], "replace_with": ["b2"]},
    )

    patch = await ModelReplanner(client, "Which is better?").replan(await _context(board, "b"))

    assert [s.id for s in patch.add] == ["b2"] and patch.replace_with == ["b2"]
    assert patch.reason == "another source"
    assert patch.tokens == 30
    first, second = client.prompts
    assert "Question: Which is better?" in first and "Failed subgoal: b" in first
    assert "rejected" in second and "cycle" in second


async def test_model_replanner_gives_up_after_its_attempts(board):
    await board.submit(GRAPH)
    bad = {"reason": "x", "subgoals": [_draft("b2", "b")], "replace_with": ["b2"]}
    client = Scripted(bad, bad)

    with pytest.raises(GraphError, match="no valid re-plan after 2 attempts"):
        await ModelReplanner(client, "q").replan(await _context(board, "b"))


async def test_model_replanner_rejects_unknown_specialists(board):
    await board.submit(GRAPH)
    odd = {**_draft("b2"), "specialist": "oracle"}
    client = Scripted({"reason": "x", "subgoals": [odd], "replace_with": ["b2"]})

    with pytest.raises(GraphError, match="unknown specialist"):
        await ModelReplanner(client, "q", attempts=1).replan(await _context(board, "b"))
