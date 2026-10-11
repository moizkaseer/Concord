import os
import uuid

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from redis.asyncio import Redis
from redis.exceptions import ConnectionError as RedisConnectionError

from concord.blackboard import Blackboard

# Database 15 keeps test data away from anything else on a local Redis.
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/15")


@pytest.fixture
async def redis():
    client = Redis.from_url(REDIS_URL, decode_responses=True)
    try:
        await client.ping()
    except RedisConnectionError:
        await client.aclose()
        if os.environ.get("CONCORD_REQUIRE_REDIS"):
            raise
        pytest.skip("Redis is not running; start it with `docker compose up -d redis`")
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()


@pytest.fixture
def board(redis) -> Blackboard:
    return Blackboard(redis, uuid.uuid4().hex)


# The tracer provider can only be installed once per process, so every test shares it.
_exporter = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_exporter))
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans() -> InMemorySpanExporter:
    """The spans finished during this test: `spans.get_finished_spans()`."""
    _exporter.clear()
    return _exporter
