"""Workers that lease ready subgoals from the blackboard and run them."""

import asyncio
import contextlib
from collections.abc import Awaitable, Callable

from concord.blackboard import Blackboard
from concord.llm import ModelClient
from concord.models import RunState, RunStatus, Subgoal

# A handler turns a subgoal and the outputs of its dependencies into an output.
Handler = Callable[[Subgoal, dict[str, str]], Awaitable[str]]


def model_handler(client: ModelClient) -> Handler:
    """A handler that sends the subgoal and its dependency outputs to a model."""

    async def handle(subgoal: Subgoal, inputs: dict[str, str]) -> str:
        lines = [f"Subgoal: {subgoal.description}"]
        lines += [f"Input from {dep}: {text}" for dep, text in inputs.items()]
        completion = await client.complete("\n".join(lines))
        return completion.text

    return handle


class WorkerPool:
    def __init__(
        self,
        board: Blackboard,
        handler: Handler,
        *,
        size: int = 4,
        lease_ttl: float = 30.0,
        poll_interval: float = 0.05,
    ) -> None:
        self._board = board
        self._handler = handler
        self._size = size
        self._lease_ttl = lease_ttl
        self._poll_interval = poll_interval

    async def run(self, *, timeout: float | None = None) -> RunStatus:
        """Work the run until it completes or fails, then return its final status."""
        async with asyncio.timeout(timeout), asyncio.TaskGroup() as group:
            for n in range(self._size):
                group.create_task(self._work(f"worker-{n}"))
        return await self._board.status()

    async def _work(self, worker: str) -> None:
        while (await self._board.status()).state == RunState.RUNNING:
            subgoal = await self._board.lease(worker, self._lease_ttl)
            if subgoal is None:
                await asyncio.sleep(self._poll_interval)
                continue
            heartbeat = asyncio.create_task(self._keep_lease(subgoal.id, worker))
            try:
                inputs = await self._board.outputs(subgoal.depends_on)
                output = await self._handler(subgoal, inputs)
            except Exception as exc:
                await self._board.fail(subgoal.id, worker, f"{type(exc).__name__}: {exc}")
            else:
                await self._board.complete(subgoal.id, worker, output)
            finally:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat

    async def _keep_lease(self, subgoal_id: str, worker: str) -> None:
        while True:
            await asyncio.sleep(self._lease_ttl / 3)
            if not await self._board.heartbeat(subgoal_id, worker, self._lease_ttl):
                return
