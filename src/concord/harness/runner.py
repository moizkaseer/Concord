"""Plays one scenario against the real blackboard, workers and supervisor."""

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum

from redis.asyncio import Redis

from concord.blackboard import Blackboard
from concord.dedupe import HashingEmbedder
from concord.harness.scenarios import DUPLICATE_KINDS, Kind, Scenario
from concord.models import RunState
from concord.supervisor import Supervisor
from concord.worker import WorkerPool, WorkResult


class Mode(StrEnum):
    BASELINE = "no supervisor"
    SUPERVISED = "supervisor"
    SIMILARITY = "supervisor + similarity"


@dataclass(frozen=True)
class Timings:
    lease_ttl: float = 0.4
    handler_timeout: float = 0.4
    tick: float = 0.05
    work_seconds: float = 0.01  # how long a healthy subgoal takes
    restart_delay: float = 0.2  # how long the supervisor stays down when restarted
    timeout: float = 15.0
    # A run that hangs without a supervisor never ends, so it gets a shorter wait.
    baseline_timeout: float = 3.0


@dataclass(frozen=True)
class Outcome:
    scenario: str
    kind: Kind
    mode: Mode
    completed: bool
    correct: bool  # completed, and every subgoal holds its expected output
    tokens: int
    worker_calls: int
    detected: bool | None  # None when the scenario has nothing to detect in this mode
    detect_seconds: float | None
    false_alarms: int  # supervisor interventions the scenario did not call for
    fault_injected: bool  # the scenario's runtime fault really happened
    seconds: float


class _Handler:
    """Stands in for the model: returns the correct output, except where the scenario
    says the first attempt at the target subgoal goes wrong."""

    def __init__(self, board: Blackboard, scenario: Scenario, timings: Timings) -> None:
        self._board = board
        self._scenario = scenario
        self._timings = timings
        self.calls = 0

    async def __call__(self, subgoal, inputs) -> WorkResult:
        self.calls += 1
        await asyncio.sleep(self._timings.work_seconds)
        output = self._scenario.expected[subgoal.id]
        # Word counts stand in for tokens, as in the mock model client.
        words = [subgoal.description, output, *inputs.values()]
        tokens = sum(len(text.split()) for text in words)

        if subgoal.id == self._scenario.target and subgoal.attempts == 1:
            kind = self._scenario.kind
            if kind is Kind.DEAD_WORKER:
                await self._mark_fault(subgoal.id)
                # The worker vanishes: no result, no failure report, no more heartbeats.
                asyncio.current_task().cancel()
                await asyncio.sleep(3600)
            if kind is Kind.HUNG_WORKER:
                await self._mark_fault(subgoal.id)
                await asyncio.sleep(3600)
            if kind is Kind.OFF_TASK:
                await self._mark_fault(subgoal.id)
                return WorkResult("An unrelated answer about something else entirely", tokens)
        return WorkResult(output, tokens)

    async def _mark_fault(self, subgoal_id: str) -> None:
        await self._board.log_event("fault.injected", id=subgoal_id, kind=self._scenario.kind)


async def _restart_supervisor(board, make_supervisor, first: asyncio.Task, delay: float) -> None:
    """Kill the supervisor once work is under way, then bring up a new one."""
    while (await board.status()).done < 1:
        await asyncio.sleep(0.005)
    await board.log_event("fault.injected", kind=Kind.SUPERVISOR_RESTART)
    first.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await first
    await asyncio.sleep(delay)
    await make_supervisor().run()


async def run_scenario(
    redis: Redis, scenario: Scenario, mode: Mode, timings: Timings | None = None
) -> Outcome:
    timings = timings or Timings()
    board = Blackboard(redis, f"harness-{uuid.uuid4().hex}")
    supervised = mode is not Mode.BASELINE
    handler = _Handler(board, scenario, timings)
    pool = WorkerPool(
        board,
        handler,
        size=4,
        lease_ttl=timings.lease_ttl,
        handler_timeout=timings.handler_timeout,
        poll_interval=0.01,
    )

    def make_supervisor() -> Supervisor:
        embedder = HashingEmbedder() if mode is Mode.SIMILARITY else None
        return Supervisor(board, embedder=embedder, tick=timings.tick)

    started = time.monotonic()
    try:
        await board.submit(scenario.specs, supervised=supervised)
        supervisor = None
        if supervised:
            supervisor = asyncio.create_task(make_supervisor().run())
            if scenario.kind is Kind.SUPERVISOR_RESTART:
                supervisor = asyncio.create_task(
                    _restart_supervisor(board, make_supervisor, supervisor, timings.restart_delay)
                )
        with contextlib.suppress(TimeoutError):
            await pool.run(timeout=timings.timeout if supervised else timings.baseline_timeout)
        if supervisor is not None:
            # The supervisor stops by itself once the run has ended; if the run hung,
            # the timeout cancels it.
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(2):
                    await supervisor
        seconds = time.monotonic() - started
        return await _score(board, scenario, mode, handler.calls, seconds)
    finally:
        await board.destroy()


def _millis(event: dict[str, str]) -> int:
    return int(event["seq"].split("-")[0])


async def _score(
    board: Blackboard, scenario: Scenario, mode: Mode, calls: int, seconds: float
) -> Outcome:
    status = await board.status()
    events = await board.events()
    completed = status.state == RunState.COMPLETED
    outputs = await board.outputs(list(scenario.expected))
    correct = completed and outputs == scenario.expected

    kind, target = scenario.kind, scenario.target
    pair = {target, scenario.partner}

    def expected(event: dict[str, str]) -> bool:
        """Is this the intervention the injected fault calls for?"""
        if kind in DUPLICATE_KINDS:
            # Either one of the pair may be reused from the other, depending on which
            # the supervisor happens to see first.
            return event["type"] in ("supervisor.held", "supervisor.reused") and (
                {event["id"], event["source"]} == pair
            )
        if kind is Kind.DEAD_WORKER:
            return event["type"] == "supervisor.requeued" and event["id"] == target
        if kind is Kind.OFF_TASK:
            # No drift check exists yet, so nothing emits this.
            return event["type"] == "supervisor.flagged" and event["id"] == target
        return False

    def detects(event: dict[str, str]) -> bool:
        if kind is Kind.HUNG_WORKER:
            # Caught by the worker's own deadline, with or without a supervisor.
            return (
                event["type"] == "subgoal.retried"
                and event["id"] == target
                and event["reason"].startswith("TimeoutError")
            )
        return expected(event) and event["type"] != "supervisor.held"

    interventions = [e for e in events if e["type"].startswith("supervisor.")]
    hits = [e for e in events if detects(e)]
    supervised = mode is not Mode.BASELINE
    detectable = kind is Kind.HUNG_WORKER or (supervised and target is not None)
    detect_seconds = None
    injected = [e for e in events if e["type"] == "fault.injected"]
    if hits and injected:
        detect_seconds = (_millis(hits[0]) - _millis(injected[0])) / 1000

    return Outcome(
        scenario=scenario.name,
        kind=kind,
        mode=mode,
        completed=completed,
        correct=correct,
        tokens=status.tokens,
        worker_calls=calls,
        detected=bool(hits) if detectable else None,
        detect_seconds=detect_seconds,
        false_alarms=sum(1 for e in interventions if not expected(e)),
        fault_injected=bool(injected),
        seconds=seconds,
    )


async def run_all(
    redis: Redis,
    scenarios: list[Scenario],
    modes: tuple[Mode, ...] = tuple(Mode),
    timings: Timings | None = None,
    concurrency: int = 8,
) -> list[Outcome]:
    gate = asyncio.Semaphore(concurrency)

    async def one(scenario: Scenario, mode: Mode) -> Outcome:
        async with gate:
            return await run_scenario(redis, scenario, mode, timings)

    jobs = [one(scenario, mode) for scenario in scenarios for mode in modes]
    return list(await asyncio.gather(*jobs))
