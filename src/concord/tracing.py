"""OpenTelemetry tracing for a run.

One run is one trace. The trace context travels through the blackboard as W3C
`traceparent` strings, so spans created by different workers and by the supervisor join
the same trace:

    invoke_workflow <name>            the whole run, opened by the caller
      concord.blackboard.submit
      concord.supervisor.check        the pre-dispatch check of a subgoal
      invoke_agent <agent>            one attempt at a subgoal
        chat <model>                  a model call
        execute_tool <tool>           a tool call
        concord.blackboard.complete   the write that records the result
      concord.attempt.lost            an attempt whose worker died, recorded by the supervisor
        concord.supervisor.requeue

Every attempt sits directly under the run's root span and carries a span link to the span
that finished each of its dependencies, so the links follow the subgoal graph. Attempts
are not nested under their dependencies: a dependency's span has already ended by then,
and trace viewers treat a child that starts after its parent ended as clock skew.

Span and attribute names for agents, model calls and tools follow the OpenTelemetry GenAI
semantic conventions. Those conventions are still in Development status and have no
tagged release, so this module is written against a fixed commit:
https://github.com/open-telemetry/semantic-conventions-genai/tree/6fd0d763a0 (2026-10-09).

Nothing here needs the OpenTelemetry SDK. Until `configure()` (or the application's own
SDK setup) installs a tracer provider, every span is a no-op.
"""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Span, SpanKind, StatusCode
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from concord import __version__
from concord.llm import Completion, ModelClient

tracer = trace.get_tracer("concord", __version__)

# GenAI semantic conventions.
OPERATION = "gen_ai.operation.name"
WORKFLOW_NAME = "gen_ai.workflow.name"
AGENT_NAME = "gen_ai.agent.name"
PROVIDER = "gen_ai.provider.name"
REQUEST_MODEL = "gen_ai.request.model"
INPUT_TOKENS = "gen_ai.usage.input_tokens"
OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
TOOL_NAME = "gen_ai.tool.name"
ERROR_TYPE = "error.type"

# Concord's own attributes.
RUN_ID = "concord.run.id"
SUBGOAL_ID = "concord.subgoal.id"
ATTEMPT = "concord.subgoal.attempt"
WORKER = "concord.worker.id"
TOTAL_TOKENS = "concord.usage.total_tokens"
REASON = "concord.reason"

_propagator = TraceContextTextMapPropagator()


def current_traceparent() -> str:
    """The active span as a `traceparent` string, or "" when nothing is being recorded."""
    carrier: dict[str, str] = {}
    _propagator.inject(carrier)
    return carrier.get("traceparent", "")


def context_from(traceparent: str | None) -> Context | None:
    """A context that makes new spans children of the given `traceparent`."""
    if not traceparent:
        return None
    return _propagator.extract({"traceparent": traceparent})


def add_link(span: Span, traceparent: str | None) -> None:
    """Link `span` to the span named by `traceparent`, if there is one."""
    context = context_from(traceparent)
    if context is None:
        return
    target = trace.get_current_span(context).get_span_context()
    if target.is_valid:
        span.add_link(target)


def mark_error(span: Span, error: BaseException | str) -> None:
    if isinstance(error, BaseException):
        span.set_attribute(ERROR_TYPE, type(error).__qualname__)
        span.record_exception(error)
        description = f"{type(error).__name__}: {error}"
    else:
        span.set_attribute(ERROR_TYPE, error)
        description = error
    span.set_status(StatusCode.ERROR, description)


@contextmanager
def _span(
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    parent: str | None = None,
    attributes: Mapping[str, str | int] | None = None,
) -> Iterator[Span]:
    """Start a span, optionally under a remote parent, and tag it if the body raises."""
    with tracer.start_as_current_span(
        name,
        context=context_from(parent),
        kind=kind,
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except Exception as exc:
            mark_error(span, exc)
            raise


def workflow_span(name: str, run_id: str):
    """The root span of a run. Open it around `submit` and the code that waits for the run."""
    return _span(
        f"invoke_workflow {name}",
        attributes={OPERATION: "invoke_workflow", WORKFLOW_NAME: name, RUN_ID: run_id},
    )


def agent_span(agent: str, *, parent: str | None, attributes: Mapping[str, str | int]):
    """One attempt at a subgoal by an in-process agent."""
    return _span(
        f"invoke_agent {agent}",
        parent=parent,
        attributes={OPERATION: "invoke_agent", AGENT_NAME: agent, **attributes},
    )


def tool_span(tool: str):
    """A tool call. Use inside a handler: `with tool_span("web_search"): ...`."""
    return _span(f"execute_tool {tool}", attributes={OPERATION: "execute_tool", TOOL_NAME: tool})


def concord_span(
    name: str, *, parent: str | None = None, attributes: Mapping[str, str | int] | None = None
):
    """A span for one of Concord's own operations, such as `supervisor.check`."""
    return _span(f"concord.{name}", parent=parent, attributes=attributes)


def record_lost_attempt(
    *,
    parent: str | None,
    started_ms: int,
    ended_ns: int,
    requeued: bool,
    attributes: Mapping[str, str | int],
) -> None:
    """Record an attempt whose worker died, on the worker's behalf.

    The dead worker can never end its own span, so the supervisor writes one covering
    the time the lease was held, with the requeue (or the final failure) as its child.
    """
    start_ns = min(started_ms * 1_000_000, ended_ns)
    lost = tracer.start_span(
        "concord.attempt.lost",
        context=context_from(parent),
        attributes=attributes,
        start_time=start_ns,
    )
    mark_error(lost, "lease_expired")
    outcome = "requeue" if requeued else "fail"
    with trace.use_span(lost, end_on_exit=False):
        followup = tracer.start_span(
            f"concord.supervisor.{outcome}",
            attributes={**attributes, REASON: "lease expired"},
            start_time=ended_ns,
        )
        followup.end(end_time=ended_ns)
    lost.end(end_time=ended_ns)


class TracedModelClient:
    """Wraps any model client so each call becomes a `chat <model>` span."""

    def __init__(self, client: ModelClient, *, model: str, provider: str) -> None:
        self._client = client
        self._attributes = {OPERATION: "chat", PROVIDER: provider, REQUEST_MODEL: model}
        self._name = f"chat {model}"

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> Completion:
        with _span(self._name, kind=SpanKind.CLIENT, attributes=self._attributes) as span:
            completion = await self._client.complete(
                prompt, system=system, schema=schema, web_search=web_search
            )
            span.set_attribute(INPUT_TOKENS, completion.input_tokens)
            span.set_attribute(OUTPUT_TOKENS, completion.output_tokens)
            if web_search:
                # Web search runs inside the API call, so it cannot have its own span.
                span.set_attribute("concord.web_search.requests", completion.searches)
            return completion


def configure(
    endpoint: str = "http://localhost:4318/v1/traces", service_name: str = "concord"
) -> None:
    """Send spans to an OTLP/HTTP endpoint, such as the Jaeger in docker-compose.yml."""
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
