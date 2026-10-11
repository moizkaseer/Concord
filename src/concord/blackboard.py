"""The shared blackboard: subgoal records in Redis, with every state change logged to a stream.

Layout for one run (`p` is the run's key prefix):

    p:run              hash    state, total, done
    p:sg:<id>          hash    one subgoal record
    p:dependents:<id>  set     ids that depend on <id>
    p:ready            list    ids waiting for a worker
    p:leases           zset    leased ids scored by lease expiry (ms)
    p:events           stream  every state change, in order

State changes run as Lua scripts so each one is atomic: a subgoal can never be leased
twice, and a state change is never recorded without its event. The scripts build key
names from the prefix, which works on a single Redis node but not on Redis Cluster.
"""

import json
from collections.abc import Sequence

from redis.asyncio import Redis

from concord.graph import DEFAULT_MAX_DEPTH, DEFAULT_MAX_SUBGOALS, validate_graph
from concord.models import RunState, RunStatus, State, Subgoal, SubgoalSpec

# Lease times use the Redis server clock so every worker agrees on what "expired" means.
_NOW_MS = """
local function now_ms()
  local t = redis.call('TIME')
  return t[1] * 1000 + math.floor(t[2] / 1000)
end
"""

_LEASE = (
    _NOW_MS
    + """
local p, worker, ttl = ARGV[1], ARGV[2], tonumber(ARGV[3])
local id = redis.call('RPOP', p .. 'ready')
if not id then return false end
local key = p .. 'sg:' .. id
local expires = now_ms() + ttl
redis.call('HSET', key, 'state', 'leased', 'lease_owner', worker, 'lease_expires_at', expires)
local attempt = redis.call('HINCRBY', key, 'attempts', 1)
redis.call('ZADD', p .. 'leases', expires, id)
redis.call('XADD', p .. 'events', '*',
  'type', 'subgoal.leased', 'id', id, 'worker', worker, 'attempt', attempt)
return id
"""
)

_HEARTBEAT = (
    _NOW_MS
    + """
local p, id, worker, ttl = ARGV[1], ARGV[2], ARGV[3], tonumber(ARGV[4])
local key = p .. 'sg:' .. id
local cur = redis.call('HMGET', key, 'state', 'lease_owner')
if cur[1] ~= 'leased' or cur[2] ~= worker then return 0 end
local expires = now_ms() + ttl
redis.call('HSET', key, 'lease_expires_at', expires)
redis.call('ZADD', p .. 'leases', expires, id)
return 1
"""
)

_COMPLETE = """
local p, id, worker, output = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local key = p .. 'sg:' .. id
local cur = redis.call('HMGET', key, 'state', 'lease_owner')
if cur[1] ~= 'leased' or cur[2] ~= worker then return 0 end
redis.call('HSET', key, 'state', 'done', 'output', output)
redis.call('HDEL', key, 'lease_owner', 'lease_expires_at')
redis.call('ZREM', p .. 'leases', id)
redis.call('XADD', p .. 'events', '*', 'type', 'subgoal.done', 'id', id, 'worker', worker)
for _, child in ipairs(redis.call('SMEMBERS', p .. 'dependents:' .. id)) do
  local ckey = p .. 'sg:' .. child
  local unmet = redis.call('HINCRBY', ckey, 'unmet', -1)
  if unmet == 0 and redis.call('HGET', ckey, 'state') == 'pending' then
    redis.call('HSET', ckey, 'state', 'ready')
    redis.call('LPUSH', p .. 'ready', child)
    redis.call('XADD', p .. 'events', '*', 'type', 'subgoal.ready', 'id', child)
  end
end
local done = redis.call('HINCRBY', p .. 'run', 'done', 1)
if done == tonumber(redis.call('HGET', p .. 'run', 'total')) then
  redis.call('HSET', p .. 'run', 'state', 'completed')
  redis.call('XADD', p .. 'events', '*', 'type', 'run.completed')
end
return 1
"""

_FAIL = """
local p, id, worker, err = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
local key = p .. 'sg:' .. id
local cur = redis.call('HMGET', key, 'state', 'lease_owner')
if cur[1] ~= 'leased' or cur[2] ~= worker then return 0 end
redis.call('HSET', key, 'state', 'failed', 'error', err)
redis.call('HDEL', key, 'lease_owner', 'lease_expires_at')
redis.call('ZREM', p .. 'leases', id)
redis.call('XADD', p .. 'events', '*',
  'type', 'subgoal.failed', 'id', id, 'worker', worker, 'error', err)
redis.call('HSET', p .. 'run', 'state', 'failed')
redis.call('XADD', p .. 'events', '*', 'type', 'run.failed', 'id', id)
return 1
"""


class RunExistsError(RuntimeError):
    """A graph was already submitted under this run id."""


class Blackboard:
    """One run's shared state. `redis` must be created with `decode_responses=True`."""

    def __init__(self, redis: Redis, run_id: str) -> None:
        self._redis = redis
        self.run_id = run_id
        # The braces are a Redis hash tag: they keep a run's keys on one cluster slot.
        self._p = f"concord:{{{run_id}}}:"
        self._lease = redis.register_script(_LEASE)
        self._heartbeat = redis.register_script(_HEARTBEAT)
        self._complete = redis.register_script(_COMPLETE)
        self._fail = redis.register_script(_FAIL)

    async def submit(
        self,
        specs: Sequence[SubgoalSpec],
        *,
        max_subgoals: int = DEFAULT_MAX_SUBGOALS,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        """Validate the graph, store it, and mark subgoals with no dependencies ready."""
        validate_graph(specs, max_subgoals=max_subgoals, max_depth=max_depth)
        if await self._redis.exists(self._p + "run"):
            raise RunExistsError(f"run {self.run_id!r} already has a graph")

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                self._p + "run",
                mapping={"state": RunState.RUNNING, "total": len(specs), "done": 0},
            )
            pipe.xadd(self._p + "events", {"type": "run.submitted", "total": len(specs)})
            for spec in specs:
                is_root = not spec.depends_on
                pipe.hset(
                    self._p + "sg:" + spec.id,
                    mapping={
                        "description": spec.description,
                        "depends_on": json.dumps(spec.depends_on),
                        "state": State.READY if is_root else State.PENDING,
                        "unmet": len(spec.depends_on),
                        "attempts": 0,
                    },
                )
                for dep in spec.depends_on:
                    pipe.sadd(self._p + "dependents:" + dep, spec.id)
                if is_root:
                    pipe.lpush(self._p + "ready", spec.id)
                    pipe.xadd(self._p + "events", {"type": "subgoal.ready", "id": spec.id})
            await pipe.execute()

    async def lease(self, worker: str, ttl: float) -> Subgoal | None:
        """Take one ready subgoal for `ttl` seconds, or return None if nothing is ready."""
        subgoal_id = await self._lease(args=[self._p, worker, int(ttl * 1000)])
        if subgoal_id is None:
            return None
        return await self.get(subgoal_id)

    async def heartbeat(self, subgoal_id: str, worker: str, ttl: float) -> bool:
        """Extend a lease. False means this worker no longer holds it."""
        return bool(await self._heartbeat(args=[self._p, subgoal_id, worker, int(ttl * 1000)]))

    async def complete(self, subgoal_id: str, worker: str, output: str) -> bool:
        """Record the output and release any dependents that are now unblocked.

        False means this worker no longer holds the lease, and nothing was written.
        """
        return bool(await self._complete(args=[self._p, subgoal_id, worker, output]))

    async def fail(self, subgoal_id: str, worker: str, error: str) -> bool:
        """Mark the subgoal and the run failed. False means the lease was lost."""
        return bool(await self._fail(args=[self._p, subgoal_id, worker, error]))

    async def get(self, subgoal_id: str) -> Subgoal:
        record = await self._redis.hgetall(self._p + "sg:" + subgoal_id)
        if not record:
            raise KeyError(subgoal_id)
        expires = record.get("lease_expires_at")
        return Subgoal(
            id=subgoal_id,
            description=record["description"],
            depends_on=json.loads(record["depends_on"]),
            state=State(record["state"]),
            attempts=int(record["attempts"]),
            lease_owner=record.get("lease_owner"),
            lease_expires_at=int(expires) if expires is not None else None,
            output=record.get("output"),
            error=record.get("error"),
        )

    async def outputs(self, subgoal_ids: Sequence[str]) -> dict[str, str]:
        """Outputs of the given subgoals; ids with no output yet are left out."""
        async with self._redis.pipeline(transaction=False) as pipe:
            for subgoal_id in subgoal_ids:
                pipe.hget(self._p + "sg:" + subgoal_id, "output")
            values = await pipe.execute()
        return {i: v for i, v in zip(subgoal_ids, values, strict=True) if v is not None}

    async def status(self) -> RunStatus:
        record = await self._redis.hgetall(self._p + "run")
        if not record:
            raise KeyError(self.run_id)
        return RunStatus(
            state=RunState(record["state"]), total=int(record["total"]), done=int(record["done"])
        )

    async def events(self) -> list[dict[str, str]]:
        """Every event of the run in order. Each carries its stream id under `seq`."""
        entries = await self._redis.xrange(self._p + "events")
        return [{"seq": seq, **fields} for seq, fields in entries]
