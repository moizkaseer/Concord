"""The drift check: did a specialist's output do the work its subgoal asked for?

The check is a rubric of yes/no questions put to a model. The model only answers the
questions; the verdict is computed from the answers here, so the rule that turns answers
into a verdict can be measured against labelled outputs and changed without touching the
prompt. A raw "how on-task is this, 0 to 1" score would have no such handle.
"""

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from concord.llm import ModelClient
from concord.models import Subgoal

# The rubric. Each entry is a question whose honest answer for a good output is "yes".
RUBRIC = {
    "subgoal_clear": (
        "Is the subgoal specific enough that you can tell what a correct output would contain?"
    ),
    "addresses_subgoal": (
        "Does the output do the work this subgoal describes? An output that honestly "
        "reports it looked and could not find something still counts as yes. An output "
        "that does some other piece of work, however well, is a no."
    ),
    "serves_question": "Is the output about the subject of the original question?",
    "stays_in_scope": (
        "Does the output keep to this subgoal, rather than spending most of its length "
        "on work that belongs to another part of the task?"
    ),
    "grounded": (
        "Are the output's main claims tied to something a reader could check: a source "
        "it names, or the inputs it was given? If it was given inputs, does it use them?"
    ),
}

# Which "no" answers make an output drifted. `core` is the default; `strict` exists so
# the two can be compared on the labelled set (see `concord.evals.drift`).
RULES = {
    "core": ("addresses_subgoal", "serves_question"),
    "strict": ("addresses_subgoal", "serves_question", "stays_in_scope", "grounded"),
}
DEFAULT_RULE = "core"

SYSTEM = """\
You are reviewing one piece of work inside a larger research task. A planner split a \
question into subgoals and a specialist produced the output below for one of them. \
Other specialists handle the other subgoals, so the output is not supposed to answer \
the whole question.

Answer each rubric question with true or false. Judge only whether the output is the \
work that was asked for. Do not judge whether its facts are correct, how well it is \
written, or how long it is: a thin but relevant output passes, and a polished output \
about something else does not. First give the reason in one or two sentences, naming \
what the output is actually about."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},
        **{name: {"type": "boolean"} for name in RUBRIC},
    },
    "required": ["reason", *RUBRIC],
    "additionalProperties": False,
}

# The reviewer reads the start of long texts rather than all of them: drift shows in
# what an output is about, which the opening establishes, and the check runs once per
# subgoal so its cost matters.
MAX_OUTPUT_CHARS = 6000
MAX_INPUT_CHARS = 1500


class Outcome(StrEnum):
    ON_TASK = "on_task"
    DRIFTED = "drifted"
    # The output fails the rubric, but the subgoal was too vague to hold it to.
    UNCLEAR = "unclear"


@dataclass(frozen=True)
class DriftVerdict:
    outcome: Outcome
    reason: str = ""
    answers: dict[str, bool] = field(default_factory=dict)
    tokens: int = 0  # what the check cost, counted against the run's budget


def decide(answers: dict[str, bool], rule: str = DEFAULT_RULE) -> Outcome:
    """Turn rubric answers into a verdict."""
    if all(answers[name] for name in RULES[rule]):
        return Outcome.ON_TASK
    # A vague subgoal is the planner's fault, not the specialist's: retrying the same
    # vague request would not help, so it is reported differently.
    return Outcome.DRIFTED if answers["subgoal_clear"] else Outcome.UNCLEAR


class DriftChecker(Protocol):
    async def check(self, subgoal: Subgoal, inputs: dict[str, str], output: str) -> DriftVerdict:
        """Review one output. `inputs` are the outputs of the subgoal's dependencies."""
        ...


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... {len(text) - limit} more characters not shown]"


def review_prompt(question: str, subgoal: Subgoal, inputs: dict[str, str], output: str) -> str:
    parts = [
        f"Original question: {question}",
        f"Subgoal ({subgoal.specialist} specialist): {subgoal.description}",
    ]
    parts += [
        f"Input the specialist was given, from {dep}:\n{_clip(text, MAX_INPUT_CHARS)}"
        for dep, text in inputs.items()
    ]
    parts.append(f"Output to review:\n{_clip(output, MAX_OUTPUT_CHARS)}")
    questions = "\n".join(f"- {name}: {text}" for name, text in RUBRIC.items())
    parts.append(f"Rubric:\n{questions}")
    return "\n\n".join(parts)


class ModelDriftChecker:
    """Puts the rubric to a model. One instance serves one run, since it holds the
    original question."""

    def __init__(self, client: ModelClient, question: str, *, rule: str = DEFAULT_RULE) -> None:
        self._client = client
        self._question = question
        self._rule = rule

    async def check(self, subgoal: Subgoal, inputs: dict[str, str], output: str) -> DriftVerdict:
        completion = await self._client.complete(
            review_prompt(self._question, subgoal, inputs, output), system=SYSTEM, schema=SCHEMA
        )
        reply: dict[str, Any] = json.loads(completion.text)
        answers = {name: bool(reply[name]) for name in RUBRIC}
        return DriftVerdict(
            decide(answers, self._rule),
            str(reply["reason"]),
            answers,
            completion.input_tokens + completion.output_tokens,
        )
