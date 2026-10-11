# Concord

A reliability layer for multi-agent runs. A planner splits a task into subgoals, workers run
them from a shared blackboard, and a supervisor catches stalls, repeated work and off-task
output.

**Status: early development.** Only the project setup and a mock model client exist so far.
See [ROADMAP.md](ROADMAP.md) for the build plan.

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
tests/         tests (run against the mock model, no API key needed)
```
