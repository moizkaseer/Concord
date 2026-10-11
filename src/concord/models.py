"""Records stored on the blackboard."""

from enum import StrEnum

from pydantic import BaseModel, Field


class State(StrEnum):
    PENDING = "pending"  # waiting on dependencies
    READY = "ready"  # dependencies done, waiting for a worker
    LEASED = "leased"  # a worker holds it
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


class Subgoal(SubgoalSpec):
    """A subgoal as it lives on the blackboard."""

    state: State = State.PENDING
    attempts: int = 0
    lease_owner: str | None = None
    lease_expires_at: int | None = None  # Redis server time, milliseconds
    output: str | None = None
    error: str | None = None


class RunStatus(BaseModel):
    state: RunState
    total: int
    done: int
