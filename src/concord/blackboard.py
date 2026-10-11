"""The shared blackboard: subgoal records in Redis, with every state change logged to a stream.

Layout for one run (`p` is the run's key prefix):

    p:run              hash    state, total, done, failed, tokens, max_attempts, supervised,
                               review, span
    p:sg:<id>          hash    one subgoal record
    p:dependents:<id>  set     ids that depend on <id>
    p:ready            list    ids waiting for a worker
    p:leases           zset    leased ids scored by lease expiry (ms)
    p:events           stream  every state change and supervisor intervention, in order
    p:dedupe:keys      hash    duplicate key -> id of the subgoal that owns it
    p:dedupe:vectors   hash    id -> JSON {deps, vector} for similarity matching
    p:waiters:<id>     hash    duplicates held until <id> finishes -> match reason

State changes run as Lua scripts so each one is atomic: a subgoal can never be leased
twice, and a state change is never recorded without its event. The scripts build key
names from the prefix, which works on a single Redis node but not on Redis Cluster.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass

from redis.asyncio import Redis
from redis.exceptions import ResponseError

from concord.graph import DEFAULT_MAX_DEPTH, DEFAULT_MAX_SUBGOALS, validate_graph
from concord.models import RunState, RunStatus, State, Subgoal, SubgoalSpec
from concord.tracing import RUN_ID, SUBGOAL_ID, concord_span, current_traceparent

DEFAULT_MAX_ATTEMPTS = 3

# Shared by every script. Lease times use the Redis server clock so every worker agrees
# on what "expired" means.
_LIB = """
local function now_ms()
  local t = redis.call('TIME')
  return t[1] * 1000 + math.floor(t[2] / 1000)
end

local function log(p, ...)
  redis.call('XADD', p .. 'events', '*', ...)
end

local function enqueue(p, id)
  redis.call('HSET', p .. 'sg:' .. id, 'state', 'ready')
  redis.call('LPUSH', p .. 'ready', id)
  log(p, 'type', 'subgoal.ready', 'id', id)
end

-- Dependencies are met. A supervised run holds the subgoal for the supervisor's
-- pre-dispatch check; otherwise it goes straight to the worker queue.
local function unblock(p, id)
  if redis.call('HGET', p .. 'run', 'supervised') == '1' then
    redis.call('HSET', p .. 'sg:' .. id, 'state', 'checking')
    log(p, 'type', 'subgoal.checking', 'id', id)
  else
    enqueue(p, id)
  end
end

local function finish_if_terminal(p)
  local r = redis.call('HMGET', p .. 'run', 'state', 'total', 'done', 'failed')
  if r[1] ~= 'running' then return end
  local failed = tonumber(r[4])
  if tonumber(r[3]) + failed < tonumber(r[2]) then return end
  if failed == 0 then
    redis.call('HSET', p .. 'run', 'state', 'completed')
    log(p, 'type', 'run.completed')
  else
    redis.call('HSET', p .. 'run', 'state', 'failed', 'reason', failed .. ' subgoal(s) failed')
    log(p, 'type', 'run.failed')
  end
end

local function release_dependents(p, id)
  for _, child in ipairs(redis.call('SMEMBERS', p .. 'dependents:' .. id)) do
    local ckey = p .. 'sg:' .. child
    local unmet = redis.call('HINCRBY', ckey, 'unmet', -1)
    if unmet == 0 and redis.call('HGET', ckey, 'state') == 'pending' then
      unblock(p, child)
    end
  end
end

local function clear_lease(p, id)
  redis.call('HDEL', p .. 'sg:' .. id, 'lease_owner', 'lease_expires_at')
  redis.call('ZREM', p .. 'leases', id)
end

-- Fail a subgoal and everything that can no longer run because of it.
local function fail_tree(p, id, err)
  redis.call('HSET', p .. 'sg:' .. id, 'state', 'failed', 'error', err)
  redis.call('HINCRBY', p .. 'run', 'failed', 1)
  log(p, 'type', 'subgoal.failed', 'id', id, 'error', err)
  local stack = {id}
  while #stack > 0 do
    local cur = table.remove(stack)
    for _, child in ipairs(redis.call('SMEMBERS', p .. 'dependents:' .. cur)) do
      local ckey = p .. 'sg:' .. child
      local state = redis.call('HGET', ckey, 'state')
      if state ~= 'failed' and state ~= 'done' then
        local cerr = 'dependency ' .. cur .. ' failed'
        redis.call('HSET', ckey, 'state', 'failed', 'error', cerr)
        redis.call('HINCRBY', p .. 'run', 'failed', 1)
        log(p, 'type', 'subgoal.failed', 'id', child, 'error', cerr)
        stack[#stack + 1] = child
      end
    end
  end
  finish_if_terminal(p)
end

-- Put a subgoal back in the queue if it has attempts left, otherwise fail it.
local function retry_or_fail(p, id, err, retry_event, worker)
  local attempts = tonumber(redis.call('HGET', p .. 'sg:' .. id, 'attempts'))
  local max = tonumber(redis.call('HGET', p .. 'run', 'max_attempts'))
  clear_lease(p, id)
  if attempts < max then
    log(p, 'type', retry_event, 'id', id, 'reason', err, 'attempt', attempts, 'worker', worker)
    enqueue(p, id)
    return 'requeued'
  end
  fail_tree(p, id, err)
  return 'failed'
end

local function holds_lease(p, id, worker)
  local cur = redis.call('HMGET', p .. 'sg:' .. id, 'state', 'lease_owner')
  return cur[1] == 'leased' and cur[2] == worker
end
"""

_LEASE = """
local p, worker, ttl = ARGV[1], ARGV[2], tonumber(ARGV[3])
if redis.call('HGET', p .. 'run', 'state') ~= 'running' then return false end
local id = redis.call('RPOP', p .. 'ready')
if not id then return false end
local key = p .. 'sg:' .. id
local now = now_ms()
local expires = now + ttl
redis.call('HSET', key, 'state', 'leased', 'lease_owner', worker,
  'leased_at', now, 'lease_expires_at', expires)
local attempt = redis.call('HINCRBY', key, 'attempts', 1)
redis.call('ZADD', p .. 'leases', expires, id)
log(p, 'type', 'subgoal.leased', 'id', id, 'worker', worker, 'attempt', attempt)
return id
"""

_HEARTBEAT = """
local p, id, worker, ttl = ARGV[1], ARGV[2], ARGV[3], tonumber(ARGV[4])
if not holds_lease(p, id, worker) then return 0 end
local expires = now_ms() + ttl
redis.call('HSET', p .. 'sg:' .. id, 'lease_expires_at', expires)
redis.call('ZADD', p .. 'leases', expires, id)
return 1
"""

_COMPLETE = """
local p, id, worker, output, tokens = ARGV[1], ARGV[2], ARGV[3], ARGV[4], tonumber(ARGV[5])
local span = ARGV[6]
if not holds_lease(p, id, worker) then return 0 end
clear_lease(p, id)
redis.call('HINCRBY', p .. 'run', 'tokens', tokens)
if redis.call('HGET', p .. 'run', 'review') == '1' then
  -- Dependents stay blocked until the supervisor has accepted the output.
  redis.call('HSET', p .. 'sg:' .. id, 'state', 'reviewing', 'output', output, 'span', span)
  log(p, 'type', 'subgoal.reviewing', 'id', id, 'worker', worker, 'tokens', tokens)
  return 1
end
redis.call('HSET', p .. 'sg:' .. id, 'state', 'done', 'output', output, 'span', span)
redis.call('HINCRBY', p .. 'run', 'done', 1)
log(p, 'type', 'subgoal.done', 'id', id, 'worker', worker, 'tokens', tokens)
release_dependents(p, id)
finish_if_terminal(p)
return 1
"""

_FAIL = """
local p, id, worker, err = ARGV[1], ARGV[2], ARGV[3], ARGV[4]
if not holds_lease(p, id, worker) then return 0 end
retry_or_fail(p, id, err, 'subgoal.retried', worker)
return 1
"""

_DISPATCH = """
local p, id = ARGV[1], ARGV[2]
if redis.call('HGET', p .. 'run', 'state') ~= 'running' then return 0 end
if redis.call('HGET', p .. 'sg:' .. id, 'state') ~= 'checking' then return 0 end
enqueue(p, id)
return 1
"""

_REUSE = """
local p, id, source, reason, span = ARGV[1], ARGV[2], ARGV[3], ARGV[4], ARGV[5]
local key = p .. 'sg:' .. id
if redis.call('HGET', key, 'state') ~= 'checking' then return 0 end
local src = redis.call('HMGET', p .. 'sg:' .. source, 'state', 'output')
if src[1] ~= 'done' then return 0 end
redis.call('HSET', key, 'state', 'done', 'output', src[2], 'reused_from', source, 'span', span)
redis.call('HINCRBY', p .. 'run', 'done', 1)
log(p, 'type', 'supervisor.reused', 'id', id, 'source', source, 'reason', reason)
release_dependents(p, id)
finish_if_terminal(p)
return 1
"""

# `note` is an optional supervisor event recorded with the acceptance: the check could not
# run, the subgoal was too vague to judge, or the output was kept despite being flagged.
_ACCEPT = """
local p, id, tokens, note, reason = ARGV[1], ARGV[2], tonumber(ARGV[3]), ARGV[4], ARGV[5]
local key = p .. 'sg:' .. id
if redis.call('HGET', p .. 'run', 'state') ~= 'running' then return 0 end
if redis.call('HGET', key, 'state') ~= 'reviewing' then return 0 end
redis.call('HINCRBY', p .. 'run', 'tokens', tokens)
if note ~= '' then
  log(p, 'type', note, 'id', id, 'reason', reason, 'action', 'kept')
end
redis.call('HSET', key, 'state', 'done')
redis.call('HINCRBY', p .. 'run', 'done', 1)
log(p, 'type', 'subgoal.done', 'id', id, 'tokens', tokens)
release_dependents(p, id)
finish_if_terminal(p)
return 1
"""

_REJECT = """
local p, id, tokens, reason = ARGV[1], ARGV[2], tonumber(ARGV[3]), ARGV[4]
local key = p .. 'sg:' .. id
if redis.call('HGET', p .. 'run', 'state') ~= 'running' then return false end
if redis.call('HGET', key, 'state') ~= 'reviewing' then return false end
redis.call('HINCRBY', p .. 'run', 'tokens', tokens)
redis.call('HINCRBY', key, 'rejections', 1)
redis.call('HSET', key, 'feedback', reason)
redis.call('HDEL', key, 'output', 'span')
local attempts = tonumber(redis.call('HGET', key, 'attempts'))
if attempts < tonumber(redis.call('HGET', p .. 'run', 'max_attempts')) then
  log(p, 'type', 'supervisor.flagged', 'id', id, 'reason', reason, 'attempt', attempts,
    'action', 'retried')
  enqueue(p, id)
  return 'requeued'
end
log(p, 'type', 'supervisor.flagged', 'id', id, 'reason', reason, 'attempt', attempts,
  'action', 'failed')
fail_tree(p, id, 'off-task output: ' .. reason)
return 'failed'
"""

_RECLAIM = """
local p = ARGV[1]
local reclaimed = {}
if redis.call('HGET', p .. 'run', 'state') ~= 'running' then return reclaimed end
local expired = redis.call('ZRANGEBYSCORE', p .. 'leases', '-inf', now_ms())
for _, id in ipairs(expired) do
  local cur = redis.call('HMGET', p .. 'sg:' .. id, 'lease_owner', 'attempts', 'leased_at')
  local worker = cur[1] or ''
  local outcome = retry_or_fail(p, id, 'lease expired', 'supervisor.requeued', worker)
  reclaimed[#reclaimed + 1] = {id, worker, cur[2], cur[3] or '0', outcome}
end
return reclaimed
"""

_HALT = """
local p, reason = ARGV[1], ARGV[2]
if redis.call('HGET', p .. 'run', 'state') ~= 'running' then return 0 end
redis.call('HSET', p .. 'run', 'state', 'failed', 'reason', reason)
log(p, 'type', 'supervisor.halted', 'reason', reason)
return 1
"""


@dataclass(frozen=True)
class Reclaimed:
    """A subgoal taken back from a worker whose lease expired."""

    id: str
    worker: str
    attempt: int
    leased_at_ms: int
    requeued: bool  # False when it was out of attempts and failed instead


class RunExistsError(RuntimeError):
    """A graph was already submitted under this run id."""


class Blackboard:
    """One run's shared state. `redis` must be created with `decode_responses=True`."""

    def __init__(self, redis: Redis, run_id: str) -> None:
        self._redis = redis
        self.run_id = run_id
        # The braces are a Redis hash tag: they keep a run's keys on one cluster slot.
        self._p = f"concord:{{{run_id}}}:"
        self._events = self._p + "events"
        self._lease = redis.register_script(_LIB + _LEASE)
        self._heartbeat = redis.register_script(_LIB + _HEARTBEAT)
        self._complete = redis.register_script(_LIB + _COMPLETE)
        self._fail = redis.register_script(_LIB + _FAIL)
        self._dispatch = redis.register_script(_LIB + _DISPATCH)
        self._reuse = redis.register_script(_LIB + _REUSE)
        self._accept = redis.register_script(_LIB + _ACCEPT)
        self._reject = redis.register_script(_LIB + _REJECT)
        self._reclaim = redis.register_script(_LIB + _RECLAIM)
        self._halt = redis.register_script(_LIB + _HALT)

    # --- submitting and reading -------------------------------------------------

    async def submit(
        self,
        specs: Sequence[SubgoalSpec],
        *,
        supervised: bool = False,
        review: bool = False,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_subgoals: int = DEFAULT_MAX_SUBGOALS,
        max_depth: int = DEFAULT_MAX_DEPTH,
    ) -> None:
        """Validate the graph, store it, and release subgoals with no dependencies.

        With `supervised=True` each subgoal waits for a supervisor's check before it
        reaches the worker queue, so a supervisor must be running for the run to progress.
        With `review=True` as well, each finished output waits for the supervisor's drift
        check before the subgoals depending on it are released.
        """
        if review and not supervised:
            raise ValueError("review needs a supervised run: the supervisor does the reviewing")
        validate_graph(specs, max_subgoals=max_subgoals, max_depth=max_depth)
        if await self._redis.exists(self._p + "run"):
            raise RunExistsError(f"run {self.run_id!r} already has a graph")

        # Every span of the run hangs off the caller's span when there is one, so the
        # caller can wrap the whole run in a single root span.
        enclosing = current_traceparent()
        with self._write("submit"):
            span = enclosing or current_traceparent()
            await self._store(specs, supervised, review, max_attempts, span)

    async def _store(
        self,
        specs: Sequence[SubgoalSpec],
        supervised: bool,
        review: bool,
        max_attempts: int,
        span: str,
    ) -> None:
        root_state = State.CHECKING if supervised else State.READY
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                self._p + "run",
                mapping={
                    "state": RunState.RUNNING,
                    "total": len(specs),
                    "done": 0,
                    "failed": 0,
                    "tokens": 0,
                    "max_attempts": max_attempts,
                    "supervised": int(supervised),
                    "review": int(review),
                    "span": span,
                },
            )
            pipe.xadd(self._events, {"type": "run.submitted", "total": len(specs)})
            for spec in specs:
                is_root = not spec.depends_on
                pipe.hset(
                    self._p + "sg:" + spec.id,
                    mapping={
                        "description": spec.description,
                        "specialist": spec.specialist,
                        "depends_on": json.dumps(spec.depends_on),
                        "state": root_state if is_root else State.PENDING,
                        "unmet": len(spec.depends_on),
                        "attempts": 0,
                        "parent_span": span,
                    },
                )
                for dep in spec.depends_on:
                    pipe.sadd(self._p + "dependents:" + dep, spec.id)
                if is_root:
                    if not supervised:
                        pipe.lpush(self._p + "ready", spec.id)
                    pipe.xadd(self._events, {"type": f"subgoal.{root_state}", "id": spec.id})
            await pipe.execute()

    async def get(self, subgoal_id: str) -> Subgoal:
        record = await self._redis.hgetall(self._p + "sg:" + subgoal_id)
        if not record:
            raise KeyError(subgoal_id)
        expires = record.get("lease_expires_at")
        return Subgoal(
            id=subgoal_id,
            description=record["description"],
            specialist=record.get("specialist", "general"),
            depends_on=json.loads(record["depends_on"]),
            state=State(record["state"]),
            attempts=int(record["attempts"]),
            lease_owner=record.get("lease_owner"),
            lease_expires_at=int(expires) if expires is not None else None,
            output=record.get("output"),
            error=record.get("error"),
            reused_from=record.get("reused_from"),
            rejections=int(record.get("rejections", 0)),
            feedback=record.get("feedback"),
            parent_span=record.get("parent_span") or None,
            span=record.get("span") or None,
        )

    async def outputs(self, subgoal_ids: Sequence[str]) -> dict[str, str]:
        """Outputs of the given subgoals; ids with no output yet are left out."""
        async with self._redis.pipeline(transaction=False) as pipe:
            for subgoal_id in subgoal_ids:
                pipe.hget(self._p + "sg:" + subgoal_id, "output")
            values = await pipe.execute()
        return {i: v for i, v in zip(subgoal_ids, values, strict=True) if v is not None}

    async def spans(self, subgoal_ids: Sequence[str]) -> dict[str, str]:
        """Trace context of the span that finished each subgoal, where one was recorded."""
        async with self._redis.pipeline(transaction=False) as pipe:
            for subgoal_id in subgoal_ids:
                pipe.hget(self._p + "sg:" + subgoal_id, "span")
            values = await pipe.execute()
        return {i: v for i, v in zip(subgoal_ids, values, strict=True) if v}

    async def status(self) -> RunStatus:
        record = await self._redis.hgetall(self._p + "run")
        if not record:
            raise KeyError(self.run_id)
        return RunStatus(
            state=RunState(record["state"]),
            total=int(record["total"]),
            done=int(record["done"]),
            failed=int(record["failed"]),
            tokens=int(record["tokens"]),
            reason=record.get("reason"),
            span=record.get("span") or None,
        )

    async def events(self) -> list[dict[str, str]]:
        """Every event of the run in order. Each carries its stream id under `seq`."""
        entries = await self._redis.xrange(self._events)
        return [{"seq": seq, **fields} for seq, fields in entries]

    async def interventions(self) -> list[dict[str, str]]:
        """Only the events that record a supervisor action."""
        return [e for e in await self.events() if e["type"].startswith("supervisor.")]

    # --- worker operations ------------------------------------------------------

    async def lease(self, worker: str, ttl: float) -> Subgoal | None:
        """Take one ready subgoal for `ttl` seconds, or return None if nothing is ready."""
        subgoal_id = await self._lease(args=[self._p, worker, int(ttl * 1000)])
        if subgoal_id is None:
            return None
        return await self.get(subgoal_id)

    async def heartbeat(self, subgoal_id: str, worker: str, ttl: float) -> bool:
        """Extend a lease. False means this worker no longer holds it."""
        return bool(await self._heartbeat(args=[self._p, subgoal_id, worker, int(ttl * 1000)]))

    async def complete(self, subgoal_id: str, worker: str, output: str, tokens: int = 0) -> bool:
        """Record the output and release any dependents that are now unblocked.

        False means this worker no longer holds the lease, and nothing was written.
        """
        # Dependents link to the span that is active now (the attempt), not to the
        # short write span opened below.
        span = current_traceparent()
        with self._write("complete", subgoal_id):
            return bool(
                await self._complete(args=[self._p, subgoal_id, worker, output, tokens, span])
            )

    async def fail(self, subgoal_id: str, worker: str, error: str) -> bool:
        """Report a failed attempt. The subgoal is retried until it runs out of attempts,
        then it and everything depending on it fail. False means the lease was lost.
        """
        with self._write("fail", subgoal_id):
            return bool(await self._fail(args=[self._p, subgoal_id, worker, error]))

    # --- supervisor operations --------------------------------------------------

    async def dispatch(self, subgoal_id: str) -> bool:
        """Send a checked subgoal to the worker queue. False if it was not waiting."""
        with self._write("dispatch", subgoal_id):
            return bool(await self._dispatch(args=[self._p, subgoal_id]))

    async def reuse(self, subgoal_id: str, source_id: str, reason: str) -> bool:
        """Finish a waiting subgoal with the output of a finished one, without a worker."""
        span = current_traceparent()
        with self._write("reuse", subgoal_id):
            return bool(await self._reuse(args=[self._p, subgoal_id, source_id, reason, span]))

    async def accept(
        self, subgoal_id: str, *, tokens: int = 0, note: str = "", reason: str = ""
    ) -> bool:
        """Pass a reviewed output and release its dependents. `tokens` is what the review
        cost. `note` names a supervisor event to record with it, such as an output kept
        despite being flagged. False if the subgoal was not waiting for review.
        """
        with self._write("accept", subgoal_id):
            return bool(await self._accept(args=[self._p, subgoal_id, tokens, note, reason]))

    async def reject(self, subgoal_id: str, reason: str, *, tokens: int = 0) -> bool | None:
        """Send a reviewed output back. The subgoal is retried with `reason` as feedback,
        or fails if it is out of attempts. Returns True when it was requeued, False when
        it failed, None when it was not waiting for review.
        """
        with self._write("reject", subgoal_id):
            outcome = await self._reject(args=[self._p, subgoal_id, tokens, reason])
        return None if outcome is None else outcome == "requeued"

    async def reclaim_expired(self) -> list[Reclaimed]:
        """Requeue (or fail, if out of attempts) every subgoal whose lease has expired."""
        rows = await self._reclaim(args=[self._p])
        return [
            Reclaimed(subgoal_id, worker, int(attempt), int(leased_at), outcome == "requeued")
            for subgoal_id, worker, attempt, leased_at, outcome in rows
        ]

    async def halt(self, reason: str) -> bool:
        """Stop the run. False if it had already ended."""
        with self._write("halt"):
            return bool(await self._halt(args=[self._p, reason]))

    async def hold(self, subgoal_id: str, source_id: str, reason: str) -> None:
        """Keep a duplicate waiting until the subgoal it matches finishes."""
        with self._write("hold", subgoal_id):
            await self._hold(subgoal_id, source_id, reason)

    async def _hold(self, subgoal_id: str, source_id: str, reason: str) -> None:
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(self._p + "waiters:" + source_id, subgoal_id, reason)
            pipe.xadd(
                self._events,
                {
                    "type": "supervisor.held",
                    "id": subgoal_id,
                    "source": source_id,
                    "reason": reason,
                },
            )
            await pipe.execute()

    async def waiters(self, source_id: str) -> dict[str, str]:
        """Duplicates held for `source_id`, mapped to the reason they matched."""
        return await self._redis.hgetall(self._p + "waiters:" + source_id)

    async def clear_waiters(self, source_id: str) -> None:
        await self._redis.delete(self._p + "waiters:" + source_id)

    async def dedupe_owner(self, key: str) -> str | None:
        return await self._redis.hget(self._p + "dedupe:keys", key)

    async def dedupe_vectors(self) -> dict[str, dict]:
        raw = await self._redis.hgetall(self._p + "dedupe:vectors")
        return {subgoal_id: json.loads(value) for subgoal_id, value in raw.items()}

    async def dedupe_register(
        self, subgoal_id: str, key: str, deps: str, vector: list[float] | None
    ) -> None:
        """Make this subgoal the one later duplicates are matched against."""
        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hsetnx(self._p + "dedupe:keys", key, subgoal_id)
            if vector is not None:
                entry = json.dumps({"deps": deps, "vector": vector})
                pipe.hset(self._p + "dedupe:vectors", subgoal_id, entry)
            await pipe.execute()

    async def ensure_group(self, group: str) -> None:
        """Create a consumer group that starts from the first event of the run."""
        try:
            await self._redis.xgroup_create(self._events, group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def read_events(
        self,
        group: str,
        consumer: str,
        *,
        pending: bool = False,
        block: float | None = None,
        count: int = 100,
    ) -> list[dict[str, str]]:
        """Read events for a consumer group.

        `pending=True` returns events this consumer was given earlier but never
        acknowledged, which is how a restarted consumer picks up where it stopped.
        """
        response = await self._redis.xreadgroup(
            group,
            consumer,
            {self._events: "0" if pending else ">"},
            count=count,
            block=int(block * 1000) if block else None,
        )
        entries = response[0][1] if response else []
        return [{"seq": seq, **fields} for seq, fields in entries]

    async def ack(self, group: str, seq: str) -> None:
        await self._redis.xack(self._events, group, seq)

    # --- housekeeping -----------------------------------------------------------

    def _write(self, operation: str, subgoal_id: str | None = None):
        """A span around a write that changes the state of the run."""
        attributes = {"db.system.name": "redis", RUN_ID: self.run_id}
        if subgoal_id is not None:
            attributes[SUBGOAL_ID] = subgoal_id
        return concord_span(f"blackboard.{operation}", attributes=attributes)

    async def log_event(self, type: str, **fields: str) -> None:
        """Append a note to the event stream without changing any state."""
        await self._redis.xadd(self._events, {"type": type, **fields})

    async def destroy(self) -> None:
        """Delete every key of this run."""
        keys = [key async for key in self._redis.scan_iter(match=self._p + "*", count=500)]
        if keys:
            await self._redis.delete(*keys)
