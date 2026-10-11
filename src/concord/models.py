"""Records stored on the blackboard."""

from enum import StrEnum

from pydantic import BaseModel, Field


class State(StrEnum):
    PENDING = "pending"  # waiting on dependencies
    CHECKING = "checking"  # dependencies done, waiting for the supervisor's check
    READY = "ready"  # in the queue, waiting for a worker
    LEASED = "leased"  # a worker holds it
    REVIEWING = "reviewing"  # a worker finished it; waiting for the supervisor's drift check
    DONE = "done"
    FAILED = "failed"


class RunState(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SubgoalSpec(BaseModel):
    """What a planner submits: one node of the subgoal graph."""

    # Ids become part of Redis keys, so keep them to a safe character set.
    id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=64)
    description: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    specialist: str = "general"  # which kind of worker should handle it


class Subgoal(SubgoalSpec):
    """A subgoal as it lives on the blackboard."""

    state: State = State.PENDING
    attempts: int = 0
    lease_owner: str | None = None
    lease_expires_at: int | None = None  # Redis server time, milliseconds
    output: str | None = None
    error: str | None = None
    reused_from: str | None = None  # set when the supervisor copied another subgoal's output
    rejections: int = 0  # outputs the drift check sent back
    feedback: str | None = None  # why the last output was sent back, for the next attempt
    # Trace context, as W3C traceparent strings.
    parent_span: str | None = None  # the run's root span; this subgoal's spans go under it
    span: str | None = None  # the span that finished this subgoal; dependents link to it


class RunStatus(BaseModel):
    state: RunState
    total: int
    done: int
    failed: int = 0
    tokens: int = 0
    reason: str | None = None  # why the run failed or was halted
    span: str | None = None  # trace context of the run's root span
