# Concord

A reliability layer for multi-agent runs. A planner splits a task into subgoals, workers run
them from a shared blackboard, and a supervisor catches stalls, repeated work and off-task
output.

**Status: early development.** The blackboard, scheduler and worker pool work against a mock
model. The supervisor, tracing and real model calls are not built yet. See
[ROADMAP.md](ROADMAP.md) for the build plan.

## What works today

- **Graph validation**: a submitted subgoal graph must be acyclic, reference only known
  subgoals, and stay within size and depth limits.
- **Blackboard**: subgoal records live in Redis. Every state change is an atomic Lua script
  that also appends an event to a Redis Stream.
- **Scheduling**: a subgoal becomes ready when all its dependencies are done.
- **Workers**: an asyncio pool leases ready subgoals, keeps the lease alive with heartbeats
  and writes the output back.

```python
from redis.asyncio import Redis

from concord.blackboard import Blackboard
from concord.llm import MockModelClient
from concord.models import SubgoalSpec
from concord.worker import WorkerPool, model_handler

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
board = Blackboard(redis, run_id="demo")
await board.submit([
    SubgoalSpec(id="search", description="find sources"),
    SubgoalSpec(id="answer", description="write the answer", depends_on=["search"]),
])
status = await WorkerPool(board, model_handler(MockModelClient(default="ok"))).run()
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
