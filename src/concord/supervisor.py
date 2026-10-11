"""The supervisor: watches a run's event stream and steps in when something goes wrong.

It does five things:

* checks every subgoal before dispatch and reuses the output of an earlier duplicate;
* reviews every finished output when the run asks for it, and sends back one that did
  not do the work its subgoal asked for;
* requeues subgoals whose worker stopped sending heartbeats;
* halts the run when it goes over its token budget;
* records each of these as a `supervisor.*` event with a reason, and as a span.

Events are read through a Redis consumer group and acknowledged only after they are
handled, so a restarted supervisor continues from where the last one stopped. Handlers
are safe to run twice for the same event. Run one supervisor per run: duplicate
detection relies on seeing events in stream order.

Reviews call a model and take seconds, so each runs as its own task. The supervisor keeps
reading events and reclaiming leases while they are in flight.
"""

import asyncio
import contextlib
import time

from concord.blackboard import Blackboard
from concord.dedupe import Embedder, cosine, duplicate_key
from concord.drift import DriftChecker, Outcome
from concord.models import RunState, State, Subgoal
from concord.tracing import (
    ATTEMPT,
    REASON,
    RUN_ID,
    SUBGOAL_ID,
    WORKER,
    add_link,
    concord_span,
    mark_error,
    record_lost_attempt,
)

GROUP = "supervisor"
REVIEW_OUTCOME = "concord.review.outcome"


class Supervisor:
    def __init__(
        self,
        board: Blackboard,
        *,
        embedder: Embedder | None = None,
        similarity_threshold: float = 0.9,
        checker: DriftChecker | None = None,
        max_rejections: int = 1,
        token_budget: int | None = None,
        tick: float = 0.2,
        consumer: str = "supervisor",
    ) -> None:
        self._board = board
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._checker = checker
        # A wrong "drifted" verdict costs a retry. Capping how many outputs of one
        # subgoal may be sent back bounds what a bad reviewer can cost a run.
        self._max_rejections = max_rejections
        self._reviews: set[asyncio.Task[None]] = set()
        self._token_budget = token_budget
        self._tick = tick
        self._consumer = consumer

    async def run(self) -> None:
        """Supervise until the run has ended and every event has been handled."""
        await self._board.ensure_group(GROUP)
        try:
            # Events delivered to a previous supervisor that stopped before acknowledging.
            # They come back all at once, so one read is enough.
            await self._handle_batch(pending=True)
            while True:
                handled = await self._handle_batch(pending=False)
                self._reap_reviews()
                await self._reclaim()
                await self._enforce_budget()
                if handled or self._reviews:
                    continue
                if (await self._board.status()).state != RunState.RUNNING:
                    return
        finally:
            # A review cut short leaves its event unacknowledged, so the next supervisor
            # is handed it again.
            for task in self._reviews:
                task.cancel()
            for task in self._reviews:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

    async def _handle_batch(self, *, pending: bool) -> int:
        events = await self._board.read_events(
            GROUP, self._consumer, pending=pending, block=None if pending else self._tick
        )
        for event in events:
            if event["type"] == "subgoal.reviewing":
                task = asyncio.create_task(self._review_and_ack(event))
                self._reviews.add(task)
                continue
            await self._handle(event)
            await self._board.ack(GROUP, event["seq"])
        return len(events)

    def _reap_reviews(self) -> None:
        for task in [t for t in self._reviews if t.done()]:
            self._reviews.discard(task)
            task.result()  # a review that crashed stops the supervisor, like any handler

    async def _review_and_ack(self, event: dict[str, str]) -> None:
        await self._review(event["id"])
        await self._board.ack(GROUP, event["seq"])

    async def _handle(self, event: dict[str, str]) -> None:
        match event["type"]:
            case "subgoal.checking":
                await self._check(event["id"])
            case "subgoal.done":
                await self._release_waiters(event["id"], finished=True)
            case "subgoal.failed":
                await self._release_waiters(event["id"], finished=False)

    def _attributes(self, subgoal_id: str) -> dict[str, str | int]:
        return {RUN_ID: self._board.run_id, SUBGOAL_ID: subgoal_id}

    async def _check(self, subgoal_id: str) -> None:
        """Pre-dispatch check: reuse an earlier duplicate's output, or send to a worker."""
        subgoal = await self._board.get(subgoal_id)
        if subgoal.state != State.CHECKING:
            return  # already handled before a restart
        if await self._enforce_budget():
            return
        with concord_span(
            "supervisor.check", parent=subgoal.parent_span, attributes=self._attributes(subgoal_id)
        ) as span:
            outcome, match, reason = await self._decide(subgoal)
            span.set_attribute("concord.check.outcome", outcome)
            if match is not None:
                span.set_attribute("concord.check.duplicate_of", match.id)
                span.set_attribute(REASON, reason)
                add_link(span, match.span)
            if outcome == "reuse":
                await self._board.reuse(subgoal_id, match.id, reason)
            elif outcome == "hold":
                await self._board.hold(subgoal_id, match.id, reason)
            else:
                await self._board.dispatch(subgoal_id)

    async def _decide(self, subgoal: Subgoal) -> tuple[str, Subgoal | None, str]:
        """Choose between dispatch, reuse and hold, and name the duplicate if there is one."""
        # Compare dependencies by where their output really came from, so two subgoals
        # fed by a subgoal and its reused duplicate still count as having the same inputs.
        sources = [await self._source_of(dep) for dep in subgoal.depends_on]
        deps = ",".join(sorted(sources))
        key = duplicate_key(subgoal.description, sources)

        match_id = await self._board.dedupe_owner(key)
        reason = "exact match"
        vector = None
        if self._embedder is not None:
            vector = (await self._embedder.embed([subgoal.description]))[0]
            if match_id is None:
                match_id, score = await self._most_similar(subgoal.id, deps, vector)
                reason = f"similarity {score:.2f}"

        if match_id is None or match_id == subgoal.id:
            await self._board.dedupe_register(subgoal.id, key, deps, vector)
            return "dispatch", None, ""

        match = await self._board.get(match_id)
        if match.state == State.DONE:
            return "reuse", match, reason
        if match.state == State.FAILED:
            # Nothing to reuse, so this one has to do the work itself.
            return "dispatch", match, reason
        return "hold", match, reason

    async def _most_similar(
        self, subgoal_id: str, deps: str, vector: list[float]
    ) -> tuple[str | None, float]:
        best_id, best_score = None, 0.0
        for other_id, entry in (await self._board.dedupe_vectors()).items():
            if other_id == subgoal_id or entry["deps"] != deps:
                continue
            score = cosine(vector, entry["vector"])
            if score > best_score:
                best_id, best_score = other_id, score
        if best_score >= self._threshold:
            return best_id, best_score
        return None, best_score

    async def _source_of(self, subgoal_id: str) -> str:
        return (await self._board.get(subgoal_id)).reused_from or subgoal_id

    async def _release_waiters(self, source_id: str, *, finished: bool) -> None:
        """The subgoal that duplicates were waiting on has ended."""
        waiters = await self._board.waiters(source_id)
        if not waiters:
            return
        source = await self._board.get(source_id)
        for waiter_id, reason in waiters.items():
            waiter = await self._board.get(waiter_id)
            attributes = {
                **self._attributes(waiter_id),
                "concord.check.duplicate_of": source_id,
                REASON: reason,
            }
            with concord_span(
                "supervisor.release", parent=waiter.parent_span, attributes=attributes
            ) as span:
                span.set_attribute("concord.check.outcome", "reuse" if finished else "dispatch")
                if finished:
                    add_link(span, source.span)
                    await self._board.reuse(waiter_id, source_id, reason)
                else:
                    # Nothing to reuse, so the duplicate has to do the work itself.
                    await self._board.dispatch(waiter_id)
        await self._board.clear_waiters(source_id)

    async def _review(self, subgoal_id: str) -> None:
        """Drift check on a finished output: accept it, or send it back with the reason."""
        subgoal = await self._board.get(subgoal_id)
        if subgoal.state != State.REVIEWING:
            return  # already handled before a restart
        with concord_span(
            "supervisor.review", parent=subgoal.parent_span, attributes=self._attributes(subgoal_id)
        ) as span:
            add_link(span, subgoal.span)
            if self._checker is None:
                span.set_attribute(REVIEW_OUTCOME, "skipped")
                await self._board.accept(
                    subgoal_id, note="supervisor.review_skipped", reason="no drift checker"
                )
                return
            try:
                inputs = await self._board.outputs(subgoal.depends_on)
                verdict = await self._checker.check(subgoal, inputs, subgoal.output or "")
            except Exception as exc:
                # A reviewer that is down must not stop the run: pass the output, on record.
                mark_error(span, exc)
                span.set_attribute(REVIEW_OUTCOME, "skipped")
                reason = f"review failed: {type(exc).__name__}: {exc}"
                await self._board.accept(
                    subgoal_id, note="supervisor.review_skipped", reason=reason
                )
                return
            span.set_attribute(REVIEW_OUTCOME, verdict.outcome)
            if verdict.outcome == Outcome.ON_TASK:
                await self._board.accept(subgoal_id, tokens=verdict.tokens)
                return
            span.set_attribute(REASON, verdict.reason)
            if verdict.outcome == Outcome.UNCLEAR:
                # Retrying a vague subgoal would get another guess. Keep the output and
                # record that the subgoal itself needs clarifying.
                await self._board.accept(
                    subgoal_id,
                    tokens=verdict.tokens,
                    note="supervisor.needs_clarification",
                    reason=verdict.reason,
                )
            elif subgoal.rejections >= self._max_rejections:
                await self._board.accept(
                    subgoal_id,
                    tokens=verdict.tokens,
                    note="supervisor.flagged",
                    reason=verdict.reason,
                )
            else:
                await self._board.reject(subgoal_id, verdict.reason, tokens=verdict.tokens)

    async def _reclaim(self) -> None:
        """Take back subgoals from workers that stopped sending heartbeats."""
        for lost in await self._board.reclaim_expired():
            subgoal = await self._board.get(lost.id)
            record_lost_attempt(
                parent=subgoal.parent_span,
                started_ms=lost.leased_at_ms,
                ended_ns=time.time_ns(),
                requeued=lost.requeued,
                attributes={
                    **self._attributes(lost.id),
                    ATTEMPT: lost.attempt,
                    WORKER: lost.worker,
                },
            )

    async def _enforce_budget(self) -> bool:
        if self._token_budget is None:
            return False
        status = await self._board.status()
        if status.state != RunState.RUNNING:
            return True  # already stopped; nothing more may be dispatched
        if status.tokens <= self._token_budget:
            return False
        reason = f"token budget exceeded: used {status.tokens} of {self._token_budget}"
        attributes = {RUN_ID: self._board.run_id, REASON: reason}
        with concord_span("supervisor.halt", parent=status.span, attributes=attributes):
            await self._board.halt(reason)
        return True
