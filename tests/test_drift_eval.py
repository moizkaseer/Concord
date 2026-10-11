import json
from pathlib import Path

from concord.corpus import Corpus
from concord.drift import RUBRIC, DriftVerdict, Outcome
from concord.evals.__main__ import CORPUS_FILES
from concord.evals.drift import (
    DRIFTED,
    ON_TASK,
    Item,
    Review,
    build_cases,
    estimate_cost,
    replay_runs,
    report,
    review_cases,
    score,
)
from concord.llm import Completion
from concord.models import Subgoal
from concord.recording import ReplayClient

ALL_YES = dict.fromkeys(RUBRIC, True)


def item(question: str, subgoal: str, specialist: str = "research", *deps: str) -> Item:
    return Item(
        question,
        f"question {question}",
        Subgoal(
            id=subgoal, description=f"do {subgoal}", specialist=specialist, depends_on=list(deps)
        ),
        {dep: f"output of {question}/{dep}" for dep in deps},
        f"output of {question}/{subgoal}",
    )


def items() -> dict[tuple[str, str], Item]:
    found = []
    for question in ("q1", "q2", "q3"):
        found += [
            item(question, "a"),
            item(question, "b"),
            item(question, "write", "synthesis", "a", "b"),
        ]
    return {(i.question_id, i.subgoal.id): i for i in found}


def review(label: str, kind: str = "own", **answers: bool) -> Review:
    verdict = DriftVerdict(Outcome.ON_TASK, "because", {**ALL_YES, **answers})
    return Review(
        {"id": "c", "kind": kind, "label": label, "question": "q", "subgoal": "s"}, verdict
    )


def test_cases_are_reproducible_and_labelled_by_how_they_were_built():
    cases = build_cases(items(), seed=1, own=5, sibling=3, stranger=3)

    assert cases == build_cases(items(), seed=1, own=5, sibling=3, stranger=3)
    assert [c["kind"] for c in cases] == ["own"] * 5 + ["sibling"] * 3 + ["stranger"] * 3
    for case in cases:
        target = (case["question"], case["subgoal"])
        source = (case["output_question"], case["output_subgoal"])
        if case["kind"] == "own":
            assert case["label"] == ON_TASK and source == target
        else:
            assert case["label"] == DRIFTED and source != target
            assert (source[0] == target[0]) == (case["kind"] == "sibling")


def test_a_swapped_output_comes_from_the_same_kind_of_specialist_and_not_from_an_input():
    all_items = items()

    for case in build_cases(all_items, own=0, sibling=9, stranger=9):
        target = all_items[(case["question"], case["subgoal"])].subgoal
        source = all_items[(case["output_question"], case["output_subgoal"])].subgoal
        assert source.specialist == target.specialist
        if case["kind"] == "sibling":
            assert source.id not in target.depends_on and target.id not in source.depends_on


def test_precision_and_recall_count_drifted_as_the_thing_to_find():
    reviews = [
        review(DRIFTED, addresses_subgoal=False),  # caught
        review(DRIFTED, addresses_subgoal=False),  # caught
        review(DRIFTED),  # missed
        review(ON_TASK, serves_question=False),  # wrongly flagged
        review(ON_TASK),
        review(ON_TASK, grounded=False),  # only the strict rule flags this one
    ]

    core, strict = score(reviews, "core"), score(reviews, "strict")

    assert (core.true_positives, core.false_positives, core.false_negatives) == (2, 1, 1)
    assert round(core.precision, 2) == 0.67 and round(core.recall, 2) == 0.67
    assert (strict.false_positives, strict.precision) == (2, 0.5)


def test_an_unclear_verdict_is_not_counted_as_flagging_the_output():
    reviews = [review(DRIFTED, addresses_subgoal=False, subgoal_clear=False)]

    result = score(reviews, "core")

    assert (result.true_positives, result.false_negatives, result.unclear) == (0, 1, 1)
    assert result.precision is None and result.recall == 0.0


def test_report_gives_both_rules_and_lists_the_disagreements():
    reviews = [review(DRIFTED, "sibling", addresses_subgoal=False), review(DRIFTED, "stranger")]

    text = report(reviews, "some-model", "core")

    assert "core (in use)" in text and "strict" in text
    assert "Disagreements with the label" in text and "labelled drifted" in text
    assert "1/1" in text and "0/1" in text


async def test_every_case_is_reviewed_with_its_own_subgoal_and_the_paired_output():
    all_items = items()
    cases = build_cases(all_items, own=2, sibling=2, stranger=2)
    prompts = []

    class Reviewer:
        async def complete(self, prompt, *, system=None, schema=None, web_search=False):
            prompts.append(prompt)
            return Completion(json.dumps({"reason": "r", **ALL_YES}), 10, 5)

    reviews = await review_cases(Reviewer(), cases, all_items)

    assert len(reviews) == 6 and all(r.verdict for r in reviews)
    for case in cases:
        wanted = f"output of {case['output_question']}/{case['output_subgoal']}"
        assert any(f"do {case['subgoal']}" in p and wanted in p for p in prompts)


async def test_a_review_that_fails_is_reported_and_left_out_of_the_score():
    class Broken:
        async def complete(self, prompt, **options):
            raise RuntimeError("model is down")

    reviews = await review_cases(
        Broken(), build_cases(items(), own=1, sibling=0, stranger=0), items()
    )

    assert reviews[0].verdict is None and "model is down" in reviews[0].error
    assert score(reviews, "core").true_negatives == 0
    assert "could not be reviewed" in report(reviews, "m", "core")


def test_cost_estimate_needs_a_listed_price():
    cases = build_cases(items(), own=3, sibling=0, stranger=0)

    assert estimate_cost(cases, items(), "claude-sonnet-5-5") > 0
    assert estimate_cost(cases, items(), "unknown-model") is None


async def test_the_committed_cases_all_point_at_recorded_outputs():
    questions = json.loads(Path("evals/questions.json").read_text())
    recorded = await replay_runs(
        ReplayClient("evals/recording.json"), questions, Corpus.from_files(CORPUS_FILES)
    )
    cases = json.loads(Path("evals/drift/cases.json").read_text())

    # Nineteen questions were recorded; the twentieth is skipped rather than failing.
    assert len({question for question, _ in recorded}) == 19
    assert cases == build_cases(recorded)
    assert sum(c["label"] == DRIFTED for c in cases) == 40
