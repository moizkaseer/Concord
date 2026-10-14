"""Workers that lease ready subgoals from the blackboard and run them."""

import asyncio
import contextlib
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from concord.blackboard import Blackboard
from concord.llm import ModelClient
from concord.models import RunState, RunStatus, Subgoal
from concord.tracing import (
    ATTEMPT,
    RUN_ID,
    SUBGOAL_ID,
    TOTAL_TOKENS,
    WORKER,
    add_link,
    agent_span,
    mark_error,
)


@dataclass(frozen=True)
class WorkResult:
    output: str
    tokens: int = 0  # counted against the run's token budget


# A handler turns a subgoal and the outputs of its dependencies into a result.
Handler = Callable[[Subgoal, dict[str, str]], Awaitable[WorkResult]]


def model_handler(client: ModelClient) -> Handler:
    """A handler that sends the subgoal and its dependency outputs to a model."""

    async def handle(subgoal: Subgoal, inputs: dict[str, str]) -> WorkResult:
        lines = [f"Subgoal: {subgoal.description}"]
        lines += [f"Input from {dep}: {text}" for dep, text in inputs.items()]
        completion = await client.complete("\n".join(lines))
        return WorkResult(completion.text, completion.input_tokens + completion.output_tokens)

    return handle


class WorkerPool:
    def __init__(
        self,
        board: Blackboard,
        handler: Handler,
        *,
        size: int = 4,
        lease_ttl: float = 30.0,
        handler_timeout: float = 300.0,
        poll_interval: float = 0.05,
        agent_name: str = "specialist",
        pool_id: str | None = None,
    ) -> None:
        self._board = board
        self._handler = handler
        self._size = size
        self._lease_ttl = lease_ttl
        # Heartbeats keep running while a handler hangs, so a hung handler needs its
        # own deadline; lease expiry only catches a worker that has died.
        self._handler_timeout = handler_timeout
        self._poll_interval = poll_interval
        self._agent_name = agent_name  # names the attempt spans in traces
        # Worker names are lease owners, so they must not repeat across processes: a pool
        # resuming a run must never be mistaken for the one that held its leases before.
        self.pool_id = pool_id or uuid.uuid4().hex[:8]

    async def run(self, *, timeout: float | None = None) -> RunStatus:
        """Work the run until it completes or fails, then return its final status."""
        async with asyncio.timeout(timeout), asyncio.TaskGroup() as group:
            for n in range(self._size):
                group.create_task(self._work(f"{self.pool_id}-worker-{n}"))
        return await self._board.status()

    async def _work(self, worker: str) -> None:
        while (await self._board.status()).state == RunState.RUNNING:
            subgoal = await self._board.lease(worker, self._lease_ttl)
            if subgoal is None:
                await asyncio.sleep(self._poll_interval)
                continue
            heartbeat = asyncio.create_task(self._keep_lease(subgoal.id, worker))
            try:
                await self._attempt(subgoal, worker)
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat

    async def _attempt(self, subgoal: Subgoal, worker: str) -> None:
        attributes = {
            RUN_ID: self._board.run_id,
            SUBGOAL_ID: subgoal.id,
            ATTEMPT: subgoal.attempts,
            WORKER: worker,
        }
        with agent_span(
            self._agent_name, parent=subgoal.parent_span, attributes=attributes
        ) as span:
            if span.is_recording():
                # Every attempt sits directly under the run; links to the spans that
                # finished its dependencies carry the shape of the subgoal graph.
                for dep_span in (await self._board.spans(subgoal.depends_on)).values():
                    add_link(span, dep_span)
            try:
                inputs = await self._board.outputs(subgoal.depends_on)
                async with asyncio.timeout(self._handler_timeout):
                    result = await self._handler(subgoal, inputs)
            except Exception as exc:
                mark_error(span, exc)
                await self._board.fail(subgoal.id, worker, f"{type(exc).__name__}: {exc}")
            else:
                span.set_attribute(TOTAL_TOKENS, result.tokens)
                await self._board.complete(subgoal.id, worker, result.output, result.tokens)

    async def _keep_lease(self, subgoal_id: str, worker: str) -> None:
        while True:
            await asyncio.sleep(self._lease_ttl / 3)
            if not await self._board.heartbeat(subgoal_id, worker, self._lease_ttl):
                return
