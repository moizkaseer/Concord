import json
from pathlib import Path

import pytest

from concord.corpus import Corpus, Passage
from concord.drift import RUBRIC
from concord.drift import SCHEMA as DRIFT_SCHEMA
from concord.evals.__main__ import CORPUS_DESCRIPTION, CORPUS_FILES, report
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
from concord.replanner import PATCH_SCHEMA
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


async def test_real_recorded_questions_replay_to_the_recorded_results(redis):
    """Replays replies recorded from Claude Sonnet 5.5, so this runs the real pipeline
    on real model output without calling the API."""
    recorded = {
        r["id"]: r for r in json.loads(Path("evals/results/sonnet-5-5.json").read_text())["results"]
    }
    questions = json.loads(Path("evals/questions.json").read_text())
    client = ReplayClient("evals/recording.json")
    corpus = Corpus.from_files(CORPUS_FILES)

    # One web-only question and one that also uses the local documents.
    for item in (q for q in questions if q["id"] in ("fencing-tokens", "concord-known-limits")):
        result = await evaluate(
            redis,
            client,
            item["id"],
            item["question"],
            corpus,
            corpus_description=CORPUS_DESCRIPTION,
        )

        expected = recorded[item["id"]]
        assert result.team.completed and result.single.completed
        assert result.team.text == expected["team"]["text"]
        assert result.single.text == expected["single"]["text"]
        assert result.verdict.winner == expected["verdict"]["winner"]
        assert result.team.input_tokens == expected["team"]["input_tokens"]


class ReviewedModel(FakeModel):
    """The first research output is about the wrong thing, and the reviewer notices."""

    def __init__(self) -> None:
        super().__init__()
        self.research_prompts: list[str] = []

    async def complete(self, prompt, *, system=None, schema=None, web_search=False):
        if schema is DRIFT_SCHEMA:
            wrong = "Kafka uses consumer offsets." in prompt.split("Output to review:")[1]
            answers = {**dict.fromkeys(RUBRIC, True), "addresses_subgoal": not wrong}
            reason = "it is about Kafka, not Temporal" if wrong else "it does the work"
            return Completion(json.dumps({"reason": reason, **answers}), 200, 20)
        if system == RESEARCH_SYSTEM:
            self.research_prompts.append(prompt)
            if len(self.research_prompts) == 1:
                return Completion("Kafka uses consumer offsets.", 1000, 200)
        return await super().complete(prompt, system=system, schema=schema, web_search=web_search)


async def test_team_run_with_the_drift_check_retries_an_off_task_output(redis):
    model = ReviewedModel()

    answer = await answer_with_team(
        redis, model, QUESTION, CORPUS, corpus_description="docs", drift_check=True
    )

    assert answer.completed and answer.text == "TEAM ANSWER"
    assert answer.interventions == ["supervisor.flagged"]
    # The retry was told what was wrong with the first attempt.
    assert "it is about Kafka, not Temporal" in model.research_prompts[1]
    # One plan, four specialist calls (research ran twice) and four reviews.
    assert answer.model_calls == 9


class ReplannedModel(FakeModel):
    """Web search never works; the re-plan asks the project documents instead."""

    def __init__(self) -> None:
        super().__init__(fail_research=True)
        self.patch_prompts: list[str] = []

    async def complete(self, prompt, *, system=None, schema=None, web_search=False):
        if schema is PATCH_SCHEMA:
            self.patch_prompts.append(prompt)
            draft = {
                "id": "temporal-docs",
                "specialist": "retrieval",
                "description": "Find what Temporal's own documentation says about dead workers",
                "depends_on": [],
            }
            reply = {
                "reason": "web search is down",
                "subgoals": [draft],
                "replace_with": [draft["id"]],
            }
            return Completion(json.dumps(reply), 600, 80)
        if system == RETRIEVAL_SYSTEM and "Temporal's own documentation" in prompt:
            return Completion("Temporal uses heartbeat timeouts.", 500, 50)
        return await super().complete(prompt, system=system, schema=schema, web_search=web_search)


async def test_team_run_with_replanning_recovers_from_a_subgoal_that_always_fails(redis):
    model = ReplannedModel()

    answer = await answer_with_team(
        redis, model, QUESTION, CORPUS, corpus_description="docs", replan=True
    )

    assert answer.completed and answer.text == "TEAM ANSWER"
    assert answer.interventions == ["supervisor.replanned"]
    (prompt,) = model.patch_prompts
    assert "Failed subgoal: temporal" in prompt and "search is down" in prompt


async def test_team_run_without_replanning_fails_on_the_same_subgoal(redis):
    answer = await answer_with_team(
        redis, ReplannedModel(), QUESTION, CORPUS, corpus_description="docs"
    )

    assert not answer.completed
