import os
import uuid

import pytest
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
