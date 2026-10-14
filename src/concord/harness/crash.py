"""Kill-and-resume trials.

Each trial submits a run, starts a separate process that works it (a worker pool and a
supervisor), kills that process with SIGKILL partway through, then starts a new process
on the same run. The trial passes when the resumed run ends with every expected output
and no subgoal was worked on again after its output had been recorded.

Redis stays up throughout: it holds the run's state, and this measures whether a run
survives losing everything else. Work lost in the kill (an attempt that was in flight)
is expected to be redone and is reported, not counted as a failure.

`python -m concord.harness.crash --child ...` is the process that gets killed.
"""

import argparse
import asyncio
import contextlib
import os
import random
import signal
import sys
import time
import uuid
from dataclasses import dataclass

from redis.asyncio import Redis

from concord.blackboard import Blackboard
from concord.drift import DriftVerdict, Outcome
from concord.harness.scenarios import Kind, Scenario, build
from concord.models import RunState
from concord.supervisor import Supervisor
from concord.worker import WorkerPool, WorkResult

# Events that mean a subgoal's output has been recorded on the board.
FINISHED = ("subgoal.reviewing", "subgoal.done", "supervisor.reused")


@dataclass(frozen=True)
class CrashTimings:
    work_seconds: float = 0.15  # long enough that a kill can land inside an attempt
    lease_ttl: float = 0.5  # how long the resumed run waits for the dead process's leases
    tick: float = 0.05
    timeout: float = 20.0  # for the resumed process to finish the run


# The child process reads these too, so they are fixed rather than passed in.
TIMINGS = CrashTimings()


@dataclass(frozen=True)
class CrashTrial:
    seed: int
    review: bool
    killed: bool  # the process was killed while the run was still in progress
    completed: bool
    correct: bool  # completed, and every subgoal holds its expected output
    reruns: int  # times a subgoal was worked on after its output had been recorded
    lost_attempts: int  # attempts in flight when the process died, and redone
    seconds: float


# --- the process that gets killed ---------------------------------------------------


class _AlwaysOnTask:
    """A reviewer that passes everything, so reviewed runs go through `reviewing`."""

    async def check(self, subgoal, inputs, output) -> DriftVerdict:
        return DriftVerdict(Outcome.ON_TASK)


def _handler(board: Blackboard, scenario: Scenario, timings: CrashTimings):
    async def handle(subgoal, inputs) -> WorkResult:
        # Logged before any work, so a re-run shows up even if this attempt never ends.
        await board.log_event("work.started", id=subgoal.id, attempt=str(subgoal.attempts))
        await asyncio.sleep(timings.work_seconds)
        return WorkResult(scenario.expected[subgoal.id])

    return handle


async def work_run(redis: Redis, run_id: str, seed: int) -> RunState:
    """Work an already submitted run until it ends, as a fresh process would."""
    timings = TIMINGS
    board = Blackboard(redis, run_id)
    scenario = build(Kind.HEALTHY, seed)
    pool = WorkerPool(
        board,
        _handler(board, scenario, timings),
        size=2,
        lease_ttl=timings.lease_ttl,
        handler_timeout=timings.timeout,
        poll_interval=0.01,
    )
    supervisor = Supervisor(board, checker=_AlwaysOnTask(), tick=timings.tick)
    status, _ = await asyncio.gather(pool.run(timeout=timings.timeout), supervisor.run())
    return status.state


# --- the trial ------------------------------------------------------------------------


async def _start(redis_url: str, run_id: str, seed: int) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "concord.harness.crash",
        "--child",
        "--run-id",
        run_id,
        "--seed",
        str(seed),
        "--redis-url",
        redis_url,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )


async def _kill(process: asyncio.subprocess.Process) -> None:
    with contextlib.suppress(ProcessLookupError):
        process.send_signal(signal.SIGKILL)
    await process.wait()


async def run_trial(redis: Redis, redis_url: str, seed: int, *, review: bool) -> CrashTrial:
    timings = TIMINGS
    rng = random.Random(f"crash/{seed}/{review}")
    scenario = build(Kind.HEALTHY, seed)
    board = Blackboard(redis, f"crash-{uuid.uuid4().hex}")
    started = time.monotonic()
    first = second = None
    try:
        await board.submit(scenario.specs, supervised=True, review=review)
        total = len(scenario.specs)

        # Kill once some work has finished but before the last subgoal could have, and
        # at a random point within an attempt.
        first = await _start(redis_url, board.run_id, seed)
        threshold = rng.randint(1, total - 2)
        async with asyncio.timeout(timings.timeout):
            while (await board.status()).done < threshold:
                await asyncio.sleep(0.005)
        await asyncio.sleep(rng.uniform(0, timings.work_seconds))
        killed = (await board.status()).state == RunState.RUNNING
        if killed:
            await board.log_event("fault.injected", kind="process killed")
        await _kill(first)

        second = await _start(redis_url, board.run_id, seed)
        try:
            async with asyncio.timeout(timings.timeout + 5):
                await second.wait()
        except TimeoutError:
            await _kill(second)
        if second.returncode != 0:
            message = (await second.stderr.read()).decode(errors="replace")
            print(f"resumed process for seed {seed} failed:\n{message}", file=sys.stderr)

        return await _score(board, scenario, seed, review, killed, time.monotonic() - started)
    finally:
        for process in (first, second):
            if process is not None and process.returncode is None:
                await _kill(process)
        await board.destroy()


async def _score(
    board: Blackboard, scenario: Scenario, seed: int, review: bool, killed: bool, seconds: float
) -> CrashTrial:
    status = await board.status()
    completed = status.state == RunState.COMPLETED
    outputs = await board.outputs(list(scenario.expected))

    finished: set[str] = set()
    reruns = starts = 0
    for event in await board.events():
        if event["type"] in FINISHED:
            finished.add(event["id"])
        elif event["type"] == "work.started":
            starts += 1
            reruns += event["id"] in finished

    return CrashTrial(
        seed=seed,
        review=review,
        killed=killed,
        completed=completed,
        correct=completed and outputs == scenario.expected,
        reruns=reruns,
        lost_attempts=starts - reruns - len(scenario.expected),
        seconds=round(seconds, 2),
    )


async def run_trials(
    redis: Redis, redis_url: str, trials: int, concurrency: int = 5
) -> list[CrashTrial]:
    """`trials` kill-and-resume trials, alternating between runs with and without review."""
    gate = asyncio.Semaphore(concurrency)

    async def one(seed: int) -> CrashTrial:
        async with gate:
            return await run_trial(redis, redis_url, seed, review=seed % 2 == 1)

    return list(await asyncio.gather(*(one(seed) for seed in range(trials))))


# --- child entry point ----------------------------------------------------------------


async def _child(args: argparse.Namespace) -> int:
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    try:
        state = await work_run(redis, args.run_id, args.seed)
    finally:
        await redis.aclose()
    return 0 if state == RunState.COMPLETED else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m concord.harness.crash")
    parser.add_argument("--child", action="store_true", help="work one run (used by the trials)")
    parser.add_argument("--run-id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/14")
    )
    args = parser.parse_args()
    if not args.child:
        parser.error("run the trials with `python -m concord.harness`")
    return asyncio.run(_child(args))


if __name__ == "__main__":
    sys.exit(main())
