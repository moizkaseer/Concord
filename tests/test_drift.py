import json

import pytest

from concord.drift import RUBRIC, ModelDriftChecker, Outcome, decide, review_prompt
from concord.llm import MockModelClient
from concord.models import Subgoal

ALL_YES = dict.fromkeys(RUBRIC, True)
SUBGOAL = Subgoal(id="s1", description="Find the price of solar panels", specialist="research")


def reply(**answers: bool) -> str:
    return json.dumps({"reason": "the output is about wind turbines", **ALL_YES, **answers})


def test_an_output_that_passes_every_question_is_on_task():
    assert decide(ALL_YES) is Outcome.ON_TASK


@pytest.mark.parametrize("question", ["addresses_subgoal", "serves_question"])
def test_failing_a_core_question_is_drift(question):
    assert decide({**ALL_YES, question: False}) is Outcome.DRIFTED


@pytest.mark.parametrize("question", ["stays_in_scope", "grounded"])
def test_the_other_questions_only_count_under_the_strict_rule(question):
    answers = {**ALL_YES, question: False}

    assert decide(answers) is Outcome.ON_TASK
    assert decide(answers, "strict") is Outcome.DRIFTED


def test_a_vague_subgoal_is_reported_as_unclear_instead_of_drift():
    answers = {**ALL_YES, "subgoal_clear": False}

    assert decide(answers) is Outcome.ON_TASK  # vague, but the output is fine
    assert decide({**answers, "addresses_subgoal": False}) is Outcome.UNCLEAR


async def test_checker_turns_the_models_answers_into_a_verdict():
    client = MockModelClient(default=reply(addresses_subgoal=False))

    verdict = await ModelDriftChecker(client, "Is solar cheap?").check(
        SUBGOAL, {}, "Wind turbines cost a lot."
    )

    assert verdict.outcome is Outcome.DRIFTED
    assert verdict.reason == "the output is about wind turbines"
    assert verdict.answers["addresses_subgoal"] is False
    assert verdict.tokens > 0


def test_prompt_shows_the_question_subgoal_inputs_and_output():
    prompt = review_prompt("Is solar cheap?", SUBGOAL, {"s0": "an input"}, "the output")

    for expected in ("Is solar cheap?", "Find the price of solar panels", "research", "s0"):
        assert expected in prompt
    assert "an input" in prompt and "the output" in prompt
    for name in RUBRIC:
        assert name in prompt


def test_prompt_shows_only_the_start_of_a_long_output():
    prompt = review_prompt("q", SUBGOAL, {"s0": "i" * 5000}, "o" * 20000)

    assert "o" * 6000 in prompt and "o" * 6001 not in prompt
    assert "14000 more characters not shown" in prompt
    assert "i" * 1501 not in prompt
