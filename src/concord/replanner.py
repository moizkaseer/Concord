"""Re-planning around a subgoal that keeps failing.

When a subgoal runs out of attempts in a run that allows re-planning, the supervisor asks
a re-planner for a patch: subgoals to add, and the subgoals that should take the failed
one's place for everything that depended on it. The re-planner proposes; `check_patch`
decides whether the patch is applied, whichever re-planner wrote it.
"""

import json
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import ValidationError

from concord.graph import GraphError, validate_graph
from concord.llm import ModelClient
from concord.models import State, Subgoal, SubgoalSpec


@dataclass(frozen=True)
class Patch:
    add: list[SubgoalSpec]
    # What the failed subgoal's dependents depend on in its place: added subgoals or
    # existing ones. Empty means they go ahead without it.
    replace_with: list[str]
    reason: str
    tokens: int = 0  # what writing the patch cost, counted against the run's budget


@dataclass(frozen=True)
class ReplanContext:
    failed: Subgoal
    graph: list[Subgoal]  # the whole run as it stands, the failed subgoal included
    outputs: dict[str, str] = field(default_factory=dict)  # of the finished subgoals


class Replanner(Protocol):
    async def replan(self, context: ReplanContext) -> Patch: ...


def check_patch(context: ReplanContext, patch: Patch, *, max_subgoals: int, max_depth: int) -> None:
    """Raise `GraphError` unless the patch can be applied to the run as it stands."""
    failed = context.failed
    by_id = {s.id: s for s in context.graph}
    new_ids = [s.id for s in patch.add]
    taken = sorted(set(new_ids) & set(by_id))
    if taken:
        raise GraphError(f"added subgoals reuse existing ids: {taken}")

    dead = {s.id for s in context.graph if s.state in (State.FAILED, State.REPLACED)}
    dead.add(failed.id)
    for spec in patch.add:
        bad = sorted(set(spec.depends_on) & dead)
        if bad:
            raise GraphError(f"subgoal {spec.id!r} depends on failed subgoals: {bad}")
    unknown = sorted(set(patch.replace_with) - set(new_ids) - set(by_id))
    if unknown:
        raise GraphError(f"replace_with names unknown subgoals: {unknown}")
    stand_ins = sorted(set(patch.replace_with) & dead)
    if stand_ins:
        raise GraphError(f"replace_with names failed subgoals: {stand_ins}")

    dependents = [s for s in context.graph if failed.id in s.depends_on]
    if not dependents and len(patch.replace_with) != 1:
        # Nothing reads the failed subgoal, so its output is the run's result. Exactly
        # one subgoal must produce that result instead.
        raise GraphError(
            f"{failed.id!r} is the final subgoal, so replace_with must name exactly one "
            f"subgoal to produce the result in its place, not {len(patch.replace_with)}"
        )

    # The run as it will be: without the failed subgoal and the others already given up
    # on, its dependents rewired, the new subgoals added. A full check is cheap at this
    # size and catches any cycle the new edges create.
    rewired = []
    for s in context.graph:
        if s.id in dead:
            continue
        deps = [d for d in s.depends_on if d not in dead]
        if failed.id in s.depends_on:
            deps += [r for r in patch.replace_with if r not in deps]
        rewired.append(SubgoalSpec(id=s.id, description=s.description, depends_on=deps))
    validate_graph([*rewired, *patch.add], max_subgoals=max_subgoals, max_depth=max_depth)


PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        "subgoals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "specialist": {"type": "string"},
                    "description": {"type": "string"},
                    "depends_on": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "specialist", "description", "depends_on"],
                "additionalProperties": False,
            },
        },
        "replace_with": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reason", "subgoals", "replace_with"],
    "additionalProperties": False,
}

SYSTEM = """\
A team of specialists is answering a research question by working through a graph of \
subgoals. One subgoal has failed on every attempt, and the subgoals that depend on it are \
blocked. Change the plan so the run can still finish.

You may add new subgoals and say which subgoals take the failed one's place: everything \
that depended on the failed subgoal will depend on those instead. Some choices:

- Ask for the same information another way, if the error suggests the approach was the \
problem, such as a source that cannot be reached or a request that was too broad.
- Split the failed subgoal into smaller ones.
- Leave `replace_with` empty if the rest of the plan can do without that information. \
Not possible for the final subgoal: if nothing depends on the failed subgoal, \
`replace_with` must name exactly one subgoal that produces the final result instead.

Do not add a subgoal that repeats the failed one in other words: it will fail the same \
way. New subgoals may depend on finished subgoals and on each other, never on the failed \
subgoal. Use new ids. Specialist types: {specialists}. Give the reason for the change in \
one sentence."""


class ModelReplanner:
    """Asks a model for the patch, and gives it its mistake to fix once if the patch is
    rejected."""

    def __init__(
        self,
        client: ModelClient,
        question: str,
        *,
        specialists: tuple[str, ...] = ("research", "retrieval", "synthesis"),
        attempts: int = 2,
        max_subgoals: int = 20,
        max_depth: int = 6,
        output_chars: int = 600,
    ) -> None:
        self._client = client
        self._question = question
        self._system = SYSTEM.format(specialists=", ".join(specialists))
        self._specialists = specialists
        self._attempts = attempts
        self._max_subgoals = max_subgoals
        self._max_depth = max_depth
        self._output_chars = output_chars  # finished outputs are shown cut to this length

    async def replan(self, context: ReplanContext) -> Patch:
        prompt = self._prompt(context)
        tokens = 0
        error = ""
        for _ in range(self._attempts):
            completion = await self._client.complete(
                prompt, system=self._system, schema=PATCH_SCHEMA
            )
            tokens += completion.input_tokens + completion.output_tokens
            try:
                patch = self._parse(completion.text, tokens)
                check_patch(
                    context, patch, max_subgoals=self._max_subgoals, max_depth=self._max_depth
                )
            except (GraphError, ValidationError, KeyError, TypeError, ValueError) as exc:
                error = str(exc)
                prompt = (
                    f"{self._prompt(context)}\n\nYour previous change was rejected: {error}\n\n"
                    f"Previous change:\n{completion.text}\n\nWrite a corrected change."
                )
                continue
            return patch
        raise GraphError(f"no valid re-plan after {self._attempts} attempts; last error: {error}")

    def _parse(self, text: str, tokens: int) -> Patch:
        data = json.loads(text)
        add = [SubgoalSpec.model_validate(draft) for draft in data["subgoals"]]
        unknown = sorted({s.specialist for s in add} - set(self._specialists))
        if unknown:
            raise GraphError(f"unknown specialist types: {unknown}")
        return Patch(add, list(data["replace_with"]), data["reason"], tokens)

    def _prompt(self, context: ReplanContext) -> str:
        failed = context.failed
        lines = [f"Question: {self._question}", "", "The plan as it stands:"]
        for s in context.graph:
            deps = f" (depends on {', '.join(s.depends_on)})" if s.depends_on else ""
            lines.append(f"- {s.id} [{s.specialist}, {s.state}]{deps}: {s.description}")
            if s.id in context.outputs:
                lines.append(f"  output: {context.outputs[s.id][: self._output_chars]}")
        lines += [
            "",
            f"Failed subgoal: {failed.id}",
            f"Its last error: {failed.error}",
        ]
        return "\n".join(lines)
