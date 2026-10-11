"""The planner: turns a question into a validated subgoal graph."""

import json
from dataclasses import dataclass

from pydantic import ValidationError

from concord.graph import GraphError, validate_graph
from concord.llm import ModelClient
from concord.models import SubgoalSpec

SPECIALISTS = ("research", "retrieval", "synthesis")

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "specialist": {"type": "string", "enum": list(SPECIALISTS)},
                    "description": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "specialist", "description", "depends_on"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["subgoals"],
    "additionalProperties": False,
}

SYSTEM = """\
You plan how a small team of specialists will answer a research question. Your plan is a \
graph of subgoals. Subgoals with no dependency on each other run at the same time, so a \
good plan splits the question into parts that can be worked on independently and then \
brought together.

Each subgoal goes to one kind of specialist:

- research: searches the public web and reports what it finds, with sources.
- retrieval: searches a local set of project documents and reports what they say. Use it \
only for parts of the question about that project ({corpus}).
- synthesis: writes the final answer from the outputs of the subgoals it depends on. It \
has no search tools.

The specialist working on a subgoal sees the original question, the subgoal's \
description, and the outputs of the subgoals it depends on. It sees nothing else, so \
write each description as a complete instruction: name the specific thing to find out \
rather than referring to "the first part" or "the above".

Rules the plan is checked against:

- Exactly one synthesis subgoal, and nothing depends on it. It is the only place the \
final answer is written.
- Every other subgoal must feed into the synthesis subgoal, directly or through other \
subgoals. Work that reaches no answer is wasted.
- At most {max_subgoals} subgoals. Use as few as the question needs: do not split \
something a single search can answer, and do not ask for the same thing twice in \
different words.
- Ids are short lowercase names using letters, digits and hyphens, such as \
"temporal-retries". A subgoal may only depend on ids that exist in the plan, and the \
graph must have no cycles."""


class PlanError(ValueError):
    """The planner could not produce a valid plan."""


@dataclass(frozen=True)
class Plan:
    subgoals: list[SubgoalSpec]
    answer_id: str  # the synthesis subgoal, whose output is the final answer
    attempts: int


def check_plan(specs: list[SubgoalSpec], *, max_subgoals: int) -> str:
    """Raise `GraphError` unless the plan is valid; return the id of its answer subgoal."""
    validate_graph(specs, max_subgoals=max_subgoals)
    unknown = sorted({s.specialist for s in specs} - set(SPECIALISTS))
    if unknown:
        raise GraphError(f"unknown specialist types: {unknown}")
    writers = [s.id for s in specs if s.specialist == "synthesis"]
    if len(writers) != 1:
        raise GraphError(f"plan needs exactly one synthesis subgoal, found {len(writers)}")
    answer_id = writers[0]

    by_id = {s.id: s for s in specs}
    reaches_answer, stack = {answer_id}, [answer_id]
    while stack:
        for dep in by_id[stack.pop()].depends_on:
            if dep not in reaches_answer:
                reaches_answer.add(dep)
                stack.append(dep)
    unused = sorted(set(by_id) - reaches_answer)
    if unused:
        raise GraphError(f"subgoals that do not feed the synthesis subgoal: {unused}")
    if any(answer_id in s.depends_on for s in specs):
        raise GraphError("nothing may depend on the synthesis subgoal")
    return answer_id


class Planner:
    def __init__(
        self,
        client: ModelClient,
        *,
        corpus_description: str = "none is available, so do not use retrieval",
        max_subgoals: int = 10,
        attempts: int = 2,
    ) -> None:
        self._client = client
        self._system = SYSTEM.format(corpus=corpus_description, max_subgoals=max_subgoals)
        self._max_subgoals = max_subgoals
        self._attempts = attempts

    async def plan(self, question: str) -> Plan:
        """Ask for a plan, check it, and give the model its mistake to fix if it fails."""
        prompt = f"Question: {question}"
        error = ""
        for attempt in range(1, self._attempts + 1):
            completion = await self._client.complete(
                prompt, system=self._system, schema=PLAN_SCHEMA
            )
            try:
                drafts = json.loads(completion.text)["subgoals"]
                specs = [SubgoalSpec.model_validate(draft) for draft in drafts]
                answer_id = check_plan(specs, max_subgoals=self._max_subgoals)
            except (GraphError, ValidationError, KeyError, TypeError, ValueError) as exc:
                error = str(exc)
                prompt = (
                    f"Question: {question}\n\n"
                    f"Your previous plan was rejected: {error}\n\n"
                    f"Previous plan:\n{completion.text}\n\n"
                    "Write a corrected plan."
                )
                continue
            return Plan(specs, answer_id, attempt)
        raise PlanError(f"no valid plan after {self._attempts} attempts; last error: {error}")
