# Concord

A reliability layer for multi-agent runs. A planner splits a task into subgoals, workers run
them from a shared blackboard, and a supervisor catches stalls, repeated work and off-task
output.

**Status: early development.** The blackboard, scheduler, worker pool and supervisor work
against a mock model. Tracing, real model calls and the measured scorecard are not built yet.
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

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker.

```sh
uv sync                 # install dependencies
docker compose up -d    # start Redis and Postgres
uv run pytest           # run the tests
uv run ruff check .     # lint
```

## Layout

```
src/concord/   library code
tests/         tests (mock model, no API key needed; blackboard tests need Redis running)
```
