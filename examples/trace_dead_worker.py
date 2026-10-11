"""Send one traced run to Jaeger: a worker dies mid-subgoal and the supervisor recovers.

    docker compose up -d redis jaeger
    uv run python examples/trace_dead_worker.py

Then open the printed link. The trace shows the lost attempt, the requeue and the retry.
"""

import asyncio
import contextlib
import uuid

from opentelemetry import trace
from redis.asyncio import Redis

from concord.blackboard import Blackboard
from concord.llm import MockModelClient
from concord.models import SubgoalSpec
from concord.supervisor import Supervisor
from concord.tracing import TracedModelClient, configure, tool_span, workflow_span
from concord.worker import WorkerPool, model_handler

GRAPH = [
    SubgoalSpec(id="search-cost", description="Find sources on the cost of solar power"),
    SubgoalSpec(id="search-risks", description="Find sources on the risks of solar power"),
    SubgoalSpec(
        id="summary-cost", description="Summarize the cost sources", depends_on=["search-cost"]
    ),
    SubgoalSpec(
        id="summary-risks", description="Summarize the risk sources", depends_on=["search-risks"]
    ),
    SubgoalSpec(
        id="answer",
        description="Write the final answer",
        depends_on=["summary-cost", "summary-risks"],
    ),
]


async def main() -> None:
    configure(service_name="concord-demo")
    redis = Redis.from_url("redis://localhost:6379/13", decode_responses=True)
    board = Blackboard(redis, f"demo-{uuid.uuid4().hex[:8]}")
    model = TracedModelClient(
        MockModelClient(default="A short mock reply."), model="mock-1", provider="concord.mock"
    )
    respond = model_handler(model)
    killed = asyncio.Event()

    async def handler(subgoal, inputs):
        if subgoal.id.startswith("search"):
            with tool_span("web_search"):
                await asyncio.sleep(0.05)
        if subgoal.id == "summary-cost" and subgoal.attempts == 1:
            # This worker dies here: no result, no failure report, no more heartbeats.
            killed.set()
            await asyncio.sleep(3600)
        await asyncio.sleep(0.05)
        return await respond(subgoal, inputs)

    def pool() -> WorkerPool:
        return WorkerPool(board, handler, size=2, lease_ttl=1.0, poll_interval=0.02)

    with workflow_span("research-question", board.run_id) as root:
        await board.submit(GRAPH, supervised=True)
        supervisor = asyncio.create_task(Supervisor(board, tick=0.1).run())
        doomed = asyncio.create_task(pool().run())
        await killed.wait()
        doomed.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await doomed
        status = await pool().run(timeout=30)
        await supervisor

    trace_id = format(root.get_span_context().trace_id, "032x")
    trace.get_tracer_provider().shutdown()  # flush spans before exiting
    await board.destroy()
    await redis.aclose()
    print(f"run {status.state}: {status.done} of {status.total} subgoals done")
    print(f"trace: http://localhost:16686/trace/{trace_id}")


if __name__ == "__main__":
    asyncio.run(main())
