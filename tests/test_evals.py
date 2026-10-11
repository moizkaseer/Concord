import json

import pytest

from concord.corpus import Corpus, Passage
from concord.evals.__main__ import report
from concord.evals.runner import (
    JUDGE_SCHEMA,
    Meter,
    answer_alone,
    answer_with_team,
    cost_usd,
    evaluate,
    judge,
)
from concord.llm import Completion
from concord.planner import PLAN_SCHEMA
from concord.recording import RecordingClient, ReplayClient
from concord.specialists import RESEARCH_SYSTEM, RETRIEVAL_SYSTEM, SYNTHESIS_SYSTEM

QUESTION = "How does Concord compare with Temporal on dead workers?"
CORPUS = Corpus([Passage("README.md § Supervisor", "Concord requeues on lease expiry.")])
PLAN = {
    "subgoals": [
        {"id": "temporal", "specialist": "research", "depends_on": [],
         "description": "Find how Temporal detects dead workers"},
        {"id": "concord", "specialist": "retrieval", "depends_on": [],
         "description": "Find how Concord detects dead workers"},
        {"id": "answer", "specialist": "synthesis", "depends_on": ["temporal", "concord"],
         "description": "Write the comparison"},
    ]
}  # fmt: skip


class FakeModel:
    """Plays every role in a run, choosing its reply from the request it is given."""

    def __init__(self, *, plan=PLAN, judge_winner="A", fail_research=False) -> None:
        self._plan, self._winner, self._fail_research = plan, judge_winner, fail_research
        self.systems: list[str] = []

    async def complete(self, prompt, *, system=None, schema=None, web_search=False):
        self.systems.append(system)
        if schema is PLAN_SCHEMA:
            return Completion(json.dumps(self._plan), 300, 100)
        if schema is JUDGE_SCHEMA:
            reply = {"winner": self._winner, "reason": "More complete."}
            return Completion(json.dumps(reply), 400, 30)
        if system == RESEARCH_SYSTEM:
            if self._fail_research:
                raise ConnectionError("search is down")
            return Completion(
                "Temporal uses heartbeat timeouts.", 1000, 200, ("https://t.example",), 2
            )
        if system == RETRIEVAL_SYSTEM:
            return Completion("Concord requeues. [README.md § Supervisor]", 500, 50)
        if system == SYNTHESIS_SYSTEM:
            assert "Temporal uses heartbeat timeouts." in prompt and "Concord requeues." in prompt
            return Completion("TEAM ANSWER", 800, 300)
        return Completion("SINGLE ANSWER", 2000, 400, ("https://t.example",), 3)


def test_cost_uses_the_price_table_and_knows_its_limits():
    assert cost_usd("claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(6.0)
    assert cost_usd("some-other-model", 1_000_000, 1_000_000) is None


async def test_meter_adds_up_everything_spent_through_it():
    meter = Meter(FakeModel())

    await meter.complete("q", system=RESEARCH_SYSTEM, web_search=True)
    await meter.complete("q", system=SYNTHESIS_SYSTEM + "x")

    assert (meter.calls, meter.input_tokens, meter.output_tokens, meter.searches) == (
        2,
        3000,
        600,
        5,
    )


async def test_team_plans_runs_every_specialist_and_returns_the_writers_answer(redis):
    answer = await answer_with_team(redis, FakeModel(), QUESTION, CORPUS, corpus_description="docs")

    assert answer.completed and answer.text == "TEAM ANSWER"
    assert answer.subgoals == 3
    assert answer.model_calls == 4  # planner, research, retrieval, synthesis
    assert (answer.input_tokens, answer.output_tokens, answer.searches) == (2600, 650, 2)
    assert answer.interventions == []
    assert await redis.dbsize() == 0  # the run cleans up after itself


async def test_team_reports_a_plan_it_could_not_validate(redis):
    broken = {"subgoals": [PLAN["subgoals"][0]]}  # no synthesis subgoal

    answer = await answer_with_team(
        redis, FakeModel(plan=broken), QUESTION, CORPUS, corpus_description="docs"
    )

    assert not answer.completed
    assert answer.error.startswith("PlanError: no valid plan after 2 attempts")
    assert answer.model_calls == 2


async def test_team_reports_a_run_that_failed(redis):
    answer = await answer_with_team(
        redis, FakeModel(fail_research=True), QUESTION, CORPUS, corpus_description="docs"
    )

    assert not answer.completed
    assert answer.error == "2 subgoal(s) failed: dependency temporal failed"
    assert answer.text == ""


async def test_single_agent_gets_web_search_and_the_same_documents():
    model = FakeModel()

    answer = await answer_alone(model, QUESTION, CORPUS)

    assert answer.completed and answer.text == "SINGLE ANSWER"
    assert (answer.model_calls, answer.searches) == (1, 3)


async def test_judge_is_blind_and_maps_the_letter_back_to_the_system():
    verdicts = {
        winner: (
            await judge(FakeModel(judge_winner=winner), "q1", QUESTION, "team", "single")
        ).winner
        for winner in ("A", "B", "tie")
    }

    assert verdicts["tie"] == "tie"
    assert {verdicts["A"], verdicts["B"]} == {"team", "single"}


async def test_evaluate_compares_both_answers_and_the_report_renders(redis):
    result = await evaluate(redis, FakeModel(), "q1", QUESTION, CORPUS, corpus_description="docs")

    assert result.team.completed and result.single.completed
    assert result.verdict.winner in ("team", "single")
    text = report([result], "claude-opus-5-5")
    assert "Team finished 1 of 1" in text
    assert "$0.02 -> $0.02" in text  # single: 2000 in + 400 out; team: 2600 in + 650 out


async def test_a_recorded_evaluation_replays_without_the_model(redis, tmp_path):
    path = tmp_path / "cassette.json"
    recorded = await evaluate(
        redis, RecordingClient(FakeModel(), path), "q1", QUESTION, CORPUS, corpus_description="docs"
    )

    replayed = await evaluate(
        redis, ReplayClient(path), "q1", QUESTION, CORPUS, corpus_description="docs"
    )

    assert replayed.team.text == recorded.team.text == "TEAM ANSWER"
    assert replayed.single.text == recorded.single.text
    assert replayed.verdict.winner == recorded.verdict.winner
    assert replayed.team.input_tokens == recorded.team.input_tokens
