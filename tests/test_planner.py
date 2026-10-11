import json

import pytest

from concord.graph import GraphError
from concord.llm import Completion
from concord.models import SubgoalSpec
from concord.planner import PLAN_SCHEMA, PlanError, Planner, check_plan


def draft(id: str, specialist: str, *deps: str) -> dict:
    return {"id": id, "specialist": specialist, "description": f"do {id}", "depends_on": list(deps)}


GOOD = [
    draft("temporal", "research"),
    draft("celery", "research"),
    draft("answer", "synthesis", "temporal", "celery"),
]


class Scripted:
    """Replies with each given plan in turn and records what it was asked."""

    def __init__(self, *plans) -> None:
        self._replies = [p if isinstance(p, str) else json.dumps({"subgoals": p}) for p in plans]
        self.requests: list[dict] = []

    async def complete(self, prompt, *, system=None, schema=None, web_search=False):
        self.requests.append({"prompt": prompt, "system": system, "schema": schema})
        return Completion(self._replies.pop(0), 10, 10)


def specs(drafts: list[dict]) -> list[SubgoalSpec]:
    return [SubgoalSpec.model_validate(d) for d in drafts]


async def test_valid_plan_is_returned_as_subgoal_specs():
    client = Scripted(GOOD)

    plan = await Planner(client, corpus_description="the Concord docs").plan("Compare them")

    assert [(s.id, s.specialist, s.depends_on) for s in plan.subgoals] == [
        ("temporal", "research", []),
        ("celery", "research", []),
        ("answer", "synthesis", ["temporal", "celery"]),
    ]
    assert (plan.answer_id, plan.attempts) == ("answer", 1)
    (request,) = client.requests
    assert request["schema"] is PLAN_SCHEMA
    assert request["prompt"] == "Question: Compare them"
    assert "the Concord docs" in request["system"]


async def test_rejected_plan_is_sent_back_with_the_reason():
    cyclic = [
        draft("a", "research", "b"),
        draft("b", "research", "a"),
        draft("w", "synthesis", "a"),
    ]
    client = Scripted(cyclic, GOOD)

    plan = await Planner(client).plan("Compare them")

    assert plan.attempts == 2
    retry = client.requests[1]["prompt"]
    assert "Your previous plan was rejected: dependency cycle" in retry
    assert '"id": "a"' in retry


async def test_planner_gives_up_after_its_attempts():
    no_writer = [draft("a", "research")]
    client = Scripted(no_writer, no_writer)

    with pytest.raises(PlanError, match="exactly one synthesis subgoal, found 0"):
        await Planner(client).plan("Compare them")
    assert len(client.requests) == 2


async def test_malformed_reply_counts_as_a_rejected_plan():
    client = Scripted("not json", [{"id": "bad id!", "specialist": "research"}], GOOD)

    plan = await Planner(client, attempts=3).plan("Compare them")

    assert plan.attempts == 3


@pytest.mark.parametrize(
    ("drafts", "message"),
    [
        ([draft("a", "research"), draft("w", "synthesis", "a"), draft("x", "synthesis", "a")],
         "exactly one synthesis subgoal, found 2"),
        ([draft("a", "research"), draft("stray", "research"), draft("w", "synthesis", "a")],
         r"do not feed the synthesis subgoal: \['stray'\]"),
        ([draft("a", "coding"), draft("w", "synthesis", "a")], "unknown specialist types"),
        ([draft("a", "research"), draft("w", "synthesis", "missing")], "unknown subgoal"),
    ],
)  # fmt: skip
def test_check_plan_rejects(drafts, message):
    with pytest.raises(GraphError, match=message):
        check_plan(specs(drafts), max_subgoals=10)


def test_check_plan_enforces_the_size_limit():
    with pytest.raises(GraphError, match="limit is 2"):
        check_plan(specs(GOOD), max_subgoals=2)


def test_check_plan_accepts_a_chain_through_intermediate_subgoals():
    chain = [draft("a", "research"), draft("b", "retrieval", "a"), draft("w", "synthesis", "b")]

    assert check_plan(specs(chain), max_subgoals=10) == "w"
