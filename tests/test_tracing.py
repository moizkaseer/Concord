import asyncio
import contextlib

import pytest
from opentelemetry.trace import SpanKind, StatusCode

from concord.llm import MockModelClient
from concord.models import RunState, SubgoalSpec
from concord.supervisor import Supervisor
from concord.tracing import (
    TracedModelClient,
    add_link,
    concord_span,
    context_from,
    current_traceparent,
    tool_span,
    workflow_span,
)
from concord.worker import WorkerPool, WorkResult, model_handler


def spec(id: str, description: str | None = None, *deps: str) -> SubgoalSpec:
    return SubgoalSpec(id=id, description=description or f"do {id}", depends_on=list(deps))


def named(spans, name: str, **attributes):
    """Finished spans with this name whose attributes include the given ones."""
    wanted = {key.replace("__", "."): value for key, value in attributes.items()}
    return [s for s in spans if s.name == name and wanted.items() <= dict(s.attributes).items()]


async def run_traced(board, graph, handler, *, supervised=True, **supervisor_options):
    """Run a graph inside one root span, the way an application would."""
    with workflow_span("test-run", board.run_id) as root:
        await board.submit(graph, supervised=supervised)
        pool = WorkerPool(board, handler, size=4, poll_interval=0.01)
        jobs = [pool.run()]
        if supervised:
            jobs.append(Supervisor(board, tick=0.05, **supervisor_options).run())
        async with asyncio.timeout(15):
            status, *_ = await asyncio.gather(*jobs)
    return status, root.get_span_context()


def test_traceparent_survives_a_round_trip(spans):
    assert current_traceparent() == ""  # nothing active

    with concord_span("outer") as outer:
        carried = current_traceparent()
    with concord_span("inner", parent=carried) as inner:
        pass

    assert inner.get_span_context().trace_id == outer.get_span_context().trace_id
    assert inner.parent.span_id == outer.get_span_context().span_id
    assert context_from("") is None


def test_link_to_a_missing_traceparent_is_ignored(spans):
    with concord_span("a") as span:
        add_link(span, None)
        add_link(span, "")

    assert spans.get_finished_spans()[0].links == ()


async def test_model_call_becomes_a_chat_span(spans):
    client = TracedModelClient(
        MockModelClient(default="four words of reply"), model="mock-1", provider="concord.mock"
    )

    await client.complete("three word prompt")

    (chat,) = spans.get_finished_spans()
    assert chat.name == "chat mock-1"
    assert chat.kind == SpanKind.CLIENT
    assert dict(chat.attributes) == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "concord.mock",
        "gen_ai.request.model": "mock-1",
        "gen_ai.usage.input_tokens": 3,
        "gen_ai.usage.output_tokens": 4,
    }


def test_tool_span_records_the_tool_and_any_error(spans):
    with pytest.raises(ConnectionError), tool_span("web_search"):
        raise ConnectionError("offline")

    (tool,) = spans.get_finished_spans()
    assert tool.name == "execute_tool web_search"
    assert tool.kind == SpanKind.INTERNAL
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "web_search"
    assert tool.attributes["error.type"] == "ConnectionError"
    assert tool.status.status_code == StatusCode.ERROR


async def test_a_run_is_one_trace_that_follows_the_subgoal_graph(board, spans):
    client = TracedModelClient(
        MockModelClient(default="ok"), model="mock-1", provider="concord.mock"
    )
    respond = model_handler(client)

    graph = [spec("a"), spec("b"), spec("c", None, "a", "b")]
    status, root = await run_traced(board, graph, respond)
    finished = spans.get_finished_spans()

    assert status.state == RunState.COMPLETED
    assert {s.context.trace_id for s in finished} == {root.trace_id}

    attempt = {
        s.attributes["concord.subgoal.id"]: s for s in named(finished, "invoke_agent specialist")
    }
    assert set(attempt) == {"a", "b", "c"}
    for span in attempt.values():
        assert span.kind == SpanKind.INTERNAL
        assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
        assert span.attributes["gen_ai.agent.name"] == "specialist"
        assert span.attributes["concord.subgoal.attempt"] == 1
        assert span.attributes["concord.usage.total_tokens"] > 0

    # Every attempt sits under the run; c links to both of its dependencies.
    for span in attempt.values():
        assert span.parent.span_id == root.span_id
    assert attempt["a"].links == () and attempt["b"].links == ()
    assert {link.context.span_id for link in attempt["c"].links} == {
        attempt["a"].context.span_id,
        attempt["b"].context.span_id,
    }

    # Model calls and blackboard writes nest inside the attempt that made them.
    for subgoal_id, span in attempt.items():
        for name in ("chat mock-1", "concord.blackboard.complete"):
            children = [
                s for s in named(finished, name) if s.parent.span_id == span.context.span_id
            ]
            assert len(children) == 1, (subgoal_id, name)

    checks = named(finished, "concord.supervisor.check")
    assert sorted(s.attributes["concord.subgoal.id"] for s in checks) == ["a", "b", "c"]
    assert {s.attributes["concord.check.outcome"] for s in checks} == {"dispatch"}
    assert len(named(finished, "concord.blackboard.submit")) == 1


async def test_dead_worker_run_shows_lost_attempt_requeue_and_retry_in_one_trace(board, spans):
    """Phase 4 exit test."""
    reached_b = asyncio.Event()

    async def doomed(subgoal, inputs):
        if subgoal.id == "b":
            reached_b.set()
            await asyncio.sleep(3600)
        return WorkResult("ok")

    async def healthy(subgoal, inputs):
        return WorkResult("ok", tokens=5)

    with workflow_span("test-run", board.run_id) as root:
        await board.submit([spec("a"), spec("b", None, "a"), spec("c", None, "b")], supervised=True)
        supervisor = asyncio.create_task(Supervisor(board, tick=0.05).run())
        first = asyncio.create_task(
            WorkerPool(board, doomed, size=2, lease_ttl=0.3, poll_interval=0.01).run()
        )
        async with asyncio.timeout(5):
            await reached_b.wait()
        first.cancel()  # the workers die while one of them holds b
        with contextlib.suppress(asyncio.CancelledError):
            await first
        survivors = WorkerPool(board, healthy, size=2, lease_ttl=0.3, poll_interval=0.01)
        status = await survivors.run(timeout=10)
        async with asyncio.timeout(5):
            await supervisor
    finished = spans.get_finished_spans()
    trace_id = root.get_span_context().trace_id

    assert status.state == RunState.COMPLETED
    assert {s.context.trace_id for s in finished} == {trace_id}

    # The failed attempt, written by the supervisor on the dead worker's behalf.
    (lost,) = named(finished, "concord.attempt.lost")
    assert lost.attributes["concord.subgoal.id"] == "b"
    assert lost.attributes["concord.subgoal.attempt"] == 1
    assert lost.attributes["error.type"] == "lease_expired"
    assert lost.status.status_code == StatusCode.ERROR
    assert 0.25 <= (lost.end_time - lost.start_time) / 1e9 <= 1.0  # about one lease

    # The requeue, as its child.
    (requeue,) = named(finished, "concord.supervisor.requeue")
    assert requeue.parent.span_id == lost.context.span_id
    assert requeue.attributes["concord.reason"] == "lease expired"

    # The retry, by a surviving worker, which then unblocks c.
    (retry,) = named(
        finished, "invoke_agent specialist", concord__subgoal__id="b", concord__subgoal__attempt=2
    )
    assert retry.status.status_code != StatusCode.ERROR
    assert retry.start_time >= requeue.end_time
    (last,) = named(finished, "invoke_agent specialist", concord__subgoal__id="c")
    assert [link.context.span_id for link in last.links] == [retry.context.span_id]


async def test_reused_subgoal_links_to_its_source_and_its_dependents_link_to_it(board, spans):
    async def handler(subgoal, inputs):
        await asyncio.sleep(0.02)
        return WorkResult("ok")

    graph = [
        spec("first", "Find sources"),
        spec("second", "find sources."),
        spec("after", "Use the sources", "second"),
    ]
    status, _ = await run_traced(board, graph, handler)
    finished = spans.get_finished_spans()

    assert status.state == RunState.COMPLETED
    (source,) = named(finished, "invoke_agent specialist", concord__subgoal__id="first")
    (held,) = named(finished, "concord.supervisor.check", concord__subgoal__id="second")
    assert held.attributes["concord.check.outcome"] == "hold"
    assert held.attributes["concord.check.duplicate_of"] == "first"

    (release,) = named(finished, "concord.supervisor.release")
    assert release.attributes["concord.check.outcome"] == "reuse"
    assert release.attributes["concord.reason"] == "exact match"
    assert [link.context.span_id for link in release.links] == [source.context.span_id]
    assert named(finished, "invoke_agent specialist", concord__subgoal__id="second") == []

    # Work that waited on the reused subgoal links to the span that reused it.
    (after,) = named(finished, "invoke_agent specialist", concord__subgoal__id="after")
    assert [link.context.span_id for link in after.links] == [release.context.span_id]


async def test_failed_attempt_is_marked_as_an_error(board, spans):
    async def handler(subgoal, inputs):
        raise ValueError("bad input")

    await run_traced(board, [spec("a")], handler, supervised=False)
    attempts = named(spans.get_finished_spans(), "invoke_agent specialist")

    assert len(attempts) == 3  # the default attempt limit
    for span in attempts:
        assert span.attributes["error.type"] == "ValueError"
        assert span.status.status_code == StatusCode.ERROR


async def test_budget_halt_is_recorded_under_the_run(board, spans):
    async def handler(subgoal, inputs):
        return WorkResult("ok", tokens=10)

    graph = [spec("a"), spec("b", None, "a"), spec("c", None, "b")]
    _, root = await run_traced(board, graph, handler, token_budget=15)

    (halt,) = named(spans.get_finished_spans(), "concord.supervisor.halt")
    assert halt.context.trace_id == root.trace_id
    assert halt.attributes["concord.reason"] == "token budget exceeded: used 20 of 15"
