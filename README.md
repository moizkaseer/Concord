# Concord

A reliability layer for multi-agent runs. A planner splits a task into subgoals, workers run
them from a shared blackboard, and a supervisor catches stalls, repeated work and off-task
output.

**Status: early development.** The blackboard, scheduler, worker pool, supervisor and
fault-injection harness and tracing work and are measured against a mock model. The planner,
specialists and Claude client are built and tested against fakes, but **have not yet been
run against the real API**, so there are no real-question results yet. The drift check is
not built.
See [ROADMAP.md](ROADMAP.md) for the build plan.

## What works today

- **Graph validation**: a submitted subgoal graph must be acyclic, reference only known
  subgoals, and stay within size and depth limits.
- **Blackboard**: subgoal records live in Redis. Every state change is an atomic Lua script
  that also appends an event to a Redis Stream.
- **Scheduling**: a subgoal is released when all its dependencies are done.
- **Workers**: an asyncio pool leases ready subgoals, keeps the lease alive with heartbeats
  and writes the output back. A failed or timed-out attempt is retried up to a limit; after
  that the subgoal fails along with everything that depends on it.
- **Supervisor**: reads the event stream through a consumer group, so it resumes after a
  restart. It
  - requeues subgoals whose worker stopped sending heartbeats;
  - checks each subgoal before dispatch and reuses the output of an earlier duplicate
    (same normalized text and same inputs, or embedding similarity above a threshold);
  - halts the run when it goes over its token budget;
  - logs every intervention as a `supervisor.*` event with its reason.

- **Tracing**: one run is one OpenTelemetry trace, with spans for the run, each
  pre-dispatch check, each attempt, model and tool calls, and blackboard writes, named after
  the GenAI semantic conventions. An attempt lost to a dead worker is recorded by the
  supervisor, so the failure, the requeue and the retry all show up. See
  [docs/tracing.md](docs/tracing.md).

- **Planner**: asks the model for a subgoal graph as structured output, checks it (no
  cycles, exactly one writer, no work that reaches no answer) and sends a rejected plan
  back once with the reason.
- **Specialists**: research (web search), retrieval (keyword search over local documents)
  and synthesis, the only one that writes the final answer.
- **Claude client**: retries, timeouts, token accounting, web search and structured output.
  Replies can be recorded to a file and replayed without the model.

Similarity matching currently ships with a word-overlap embedder for tests. It catches
reordered or lightly edited subgoals, not true paraphrases; those need a real embedding
model, planned for the real-model phase.

```python
import asyncio

from redis.asyncio import Redis

from concord.blackboard import Blackboard
from concord.llm import MockModelClient
from concord.models import SubgoalSpec
from concord.supervisor import Supervisor
from concord.worker import WorkerPool, model_handler

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
board = Blackboard(redis, run_id="demo")
await board.submit(
    [
        SubgoalSpec(id="search", description="find sources"),
        SubgoalSpec(id="answer", description="write the answer", depends_on=["search"]),
    ],
    supervised=True,
)
pool = WorkerPool(board, model_handler(MockModelClient(default="ok")))
status, _ = await asyncio.gather(pool.run(), Supervisor(board, token_budget=50_000).run())
print(await board.interventions())
```

## Scorecard

Measured by the fault-injection harness: 72 scripted runs, each with one injected fault,
played with and without the supervisor. Everything runs on a mock model, so the token
figures are word counts and only the relative difference means anything.

| Measure | Target | Result |
|---------|--------|--------|
| Exact duplicate subgoals caught | 100% | 8/8 (100%) |
| Reworded duplicate subgoals caught | ≥ 90% | 0/16 (0%) |
| False alarms on healthy runs | ≤ 5% | 0/16 (0%) |
| Time to detect a dead worker | ≤ 0.80s (2 × lease) | about 0.4s at worst |
| Off-task output caught | precision and recall ≥ 0.8 | 0/8, no drift check yet |
| Tokens saved on runs with an exact duplicate | report the figure | 11.2% |
| Finished subgoals re-run after a crash | 0 | not measured, recovery not built |

What the supervisor changes, by fault type:

| Fault | Correct result without supervisor | With supervisor |
|-------|-----------------------------------|-----------------|
| Dead worker | 0/8, the run hangs | 8/8 |
| Exact duplicate | 8/8, but the work is done twice | 8/8, duplicate never reaches a worker |
| Hung worker | 8/8 | 8/8 (caught by the worker's own deadline, not the supervisor) |
| Supervisor restart mid-run | not applicable | 8/8 |
| Off-task output | 0/8 | 0/8 |

Two gaps are visible and left in on purpose:

- **Reworded duplicates are not caught by default.** Similarity matching is off unless an
  embedder is supplied. With the built-in word-overlap embedder switched on, light edits
  are caught (7/8 here, 38/40 in a larger run) and true paraphrases are not (0/40). In
  that larger run it also reused the wrong output once in 40 look-alike runs, where two
  subgoals differed only in their topic. Word overlap cannot tell which word changed, so
  it stays off until a real embedding model is plugged in.
- **Off-task output is not caught at all.** The drift check is a later phase.

Reproduce it with `uv run python -m concord.harness`. CI runs the same command with
`--check`, which fails the build if a gate for an already-built detector regresses.

## Running on real questions

`evals/questions.json` holds 20 research questions. The evaluation answers each one with the
agent team and with a single agent that has the same tools, then records cost, time and a
blind comparison of the two answers.

```sh
export ANTHROPIC_API_KEY=...
uv run python -m concord.evals --limit 1 --yes                      # try one question
uv run python -m concord.evals --yes --record evals/recording.json  # all 20
uv run python -m concord.evals --replay evals/recording.json        # rerun for free
```

It spends real money, so it refuses to start without `--yes` and stops starting new
questions at `--max-cost` (default $25 of token spend; web search fees are extra). The
answer comparison is made by the same model, not by a person, so treat it as a signal
rather than a verdict. This has not been run yet.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker.

```sh
uv sync                 # install dependencies
docker compose up -d    # start Redis, Postgres and Jaeger
uv run pytest           # run the tests
uv run ruff check .     # lint
uv run python -m concord.harness   # print the scorecard
```

## Layout

```
src/concord/   library code
examples/      runnable examples
evals/         the 20 evaluation questions and their results
docs/          tracing guide
tests/         tests (mock model, no API key needed; blackboard tests need Redis running)
```
