import dataclasses

import pytest

from concord.graph import validate_graph
from concord.harness.crash import CrashTrial, _score, run_trial
from concord.harness.report import crash_gates, gates, render, summarize
from concord.harness.runner import Mode, Outcome, Timings, run_scenario
from concord.harness.scenarios import DUPLICATE_KINDS, Kind, build, build_all

FAST = Timings(baseline_timeout=1.0)


def test_scenarios_are_reproducible_and_valid():
    first, second = build_all(3), build_all(3)

    assert first == second
    assert len(first) == 3 * len(Kind)
    for scenario in first:
        validate_graph(scenario.specs)
        assert set(scenario.expected) == {s.id for s in scenario.specs}


@pytest.mark.parametrize("kind", DUPLICATE_KINDS)
def test_duplicate_scenarios_repeat_an_existing_subgoal(kind):
    scenario = build(kind, seed=0)
    by_id = {s.id: s for s in scenario.specs}

    duplicate, original = by_id[scenario.target], by_id[scenario.partner]
    assert duplicate.depends_on == original.depends_on
    assert duplicate.description != original.description
    assert scenario.expected[duplicate.id] == scenario.expected[original.id]
    assert duplicate.id in by_id["answer"].depends_on


async def test_exact_duplicate_is_detected_and_saves_tokens(redis):
    scenario = build(Kind.EXACT_DUPLICATE, seed=0)

    without = await run_scenario(redis, scenario, Mode.BASELINE, FAST)
    with_supervisor = await run_scenario(redis, scenario, Mode.SUPERVISED, FAST)

    assert without.detected is None
    assert with_supervisor.detected is True
    assert with_supervisor.correct
    assert with_supervisor.false_alarms == 0
    assert with_supervisor.worker_calls == without.worker_calls - 1
    assert with_supervisor.tokens < without.tokens


async def test_dead_worker_hangs_the_run_without_a_supervisor_and_is_detected_with_one(redis):
    scenario = build(Kind.DEAD_WORKER, seed=0)

    without = await run_scenario(redis, scenario, Mode.BASELINE, FAST)
    with_supervisor = await run_scenario(redis, scenario, Mode.SUPERVISED, FAST)

    assert without.fault_injected and not without.completed
    assert with_supervisor.fault_injected and with_supervisor.correct
    assert with_supervisor.detected is True
    assert 0 < with_supervisor.detect_seconds <= 2 * FAST.lease_ttl


async def test_hung_worker_is_caught_by_the_worker_deadline(redis):
    outcome = await run_scenario(redis, build(Kind.HUNG_WORKER, seed=0), Mode.BASELINE, FAST)

    assert outcome.fault_injected and outcome.detected is True and outcome.correct


async def test_run_survives_a_supervisor_restart(redis):
    scenario = build(Kind.SUPERVISOR_RESTART, seed=0)

    outcome = await run_scenario(redis, scenario, Mode.SUPERVISED, FAST)

    assert outcome.fault_injected and outcome.correct
    assert outcome.false_alarms == 0


async def test_off_task_output_spoils_the_run_unless_the_drift_check_is_on(redis):
    scenario = build(Kind.OFF_TASK, seed=0)

    without = await run_scenario(redis, scenario, Mode.SUPERVISED, FAST)
    with_check = await run_scenario(redis, scenario, Mode.REVIEW, FAST)

    assert without.fault_injected and without.completed
    assert without.detected is False and not without.correct
    assert with_check.fault_injected and with_check.detected is True
    assert with_check.correct and with_check.false_alarms == 0
    # The off-task subgoal ran twice; nothing else did.
    assert with_check.worker_calls == without.worker_calls + 1


@pytest.mark.parametrize("kind", [Kind.HEALTHY, Kind.EXACT_DUPLICATE, Kind.DEAD_WORKER])
async def test_drift_check_leaves_other_runs_alone(redis, kind):
    outcome = await run_scenario(redis, build(kind, seed=0), Mode.REVIEW, FAST)

    assert outcome.correct and outcome.false_alarms == 0


async def test_scenario_leaves_nothing_behind_in_redis(redis):
    await run_scenario(redis, build(Kind.HEALTHY, seed=0), Mode.SUPERVISED, FAST)

    assert await redis.dbsize() == 0


def _outcome(kind: Kind, mode: Mode, **changes) -> Outcome:
    base = Outcome(
        scenario=f"{kind} #0",
        kind=kind,
        mode=mode,
        completed=True,
        correct=True,
        tokens=100,
        worker_calls=8,
        detected=None,
        detect_seconds=None,
        false_alarms=0,
        fault_injected=True,
        seconds=0.1,
    )
    return dataclasses.replace(base, **changes)


def _detected(kind: Kind, mode: Mode) -> bool | None:
    if kind is Kind.OFF_TASK:
        return mode is Mode.REVIEW
    if kind is Kind.PERMANENT_FAILURE:
        return None if mode is Mode.BASELINE else mode is Mode.REPLAN
    always = {Kind.EXACT_DUPLICATE, Kind.DEAD_WORKER, Kind.HUNG_WORKER}
    return True if kind in always and mode is not Mode.BASELINE else None


def _clean_results() -> list[Outcome]:
    return [
        _outcome(
            kind,
            mode,
            detected=_detected(kind, mode),
            correct=kind is not Kind.PERMANENT_FAILURE or mode is Mode.REPLAN,
            detect_seconds=0.4 if kind is Kind.DEAD_WORKER and mode is not Mode.BASELINE else None,
        )
        for kind in Kind
        for mode in Mode
    ]


def test_gates_pass_on_clean_results_and_report_renders():
    outcomes = _clean_results()

    text, checks = render(outcomes, Timings(), [CLEAN_TRIAL])

    assert all(gate.passed for gate in checks)
    assert "SCORECARD" in text and "GATES" in text


@pytest.mark.parametrize(
    ("kind", "change", "failing_gate"),
    [
        (Kind.EXACT_DUPLICATE, {"detected": False}, "every exact duplicate is caught"),
        (Kind.HEALTHY, {"false_alarms": 1}, "no false alarms on healthy runs"),
        (Kind.DEAD_WORKER, {"detect_seconds": 5.0}, "every dead worker is detected"),
        (Kind.SUPERVISOR_RESTART, {"correct": False}, "every run survives a supervisor restart"),
        (Kind.HUNG_WORKER, {"fault_injected": False}, "every runtime fault was really injected"),
    ],
)
def test_a_regression_fails_its_gate(kind, change, failing_gate):
    _assert_gate_fails(kind, Mode.SUPERVISED, change, failing_gate)


@pytest.mark.parametrize(
    ("kind", "change", "failing_gate"),
    [
        (Kind.OFF_TASK, {"detected": False}, "with the drift check on, every off-task output"),
        (Kind.DEAD_WORKER, {"correct": False}, "with the drift check on, no run ends"),
        (Kind.LOOKALIKE, {"false_alarms": 1}, "with the drift check on, no false alarms"),
    ],
)
def test_a_drift_check_regression_fails_its_gate(kind, change, failing_gate):
    _assert_gate_fails(kind, Mode.REVIEW, change, failing_gate)


@pytest.mark.parametrize(
    ("kind", "change", "failing_gate"),
    [
        (Kind.PERMANENT_FAILURE, {"detected": False}, "with re-planning on, every permanently"),
        (Kind.PERMANENT_FAILURE, {"correct": False}, "with re-planning on, every permanently"),
        (Kind.DEAD_WORKER, {"correct": False}, "with re-planning on, no run ends"),
        (Kind.HEALTHY, {"false_alarms": 1}, "with re-planning on, no false alarms"),
    ],
)
def test_a_replanning_regression_fails_its_gate(kind, change, failing_gate):
    _assert_gate_fails(kind, Mode.REPLAN, change, failing_gate)


async def test_permanent_failure_ends_the_run_unless_replanning_is_on(redis):
    scenario = build(Kind.PERMANENT_FAILURE, seed=0)

    without = await run_scenario(redis, scenario, Mode.SUPERVISED, FAST)
    with_replan = await run_scenario(redis, scenario, Mode.REPLAN, FAST)

    assert without.fault_injected and not without.completed and without.detected is False
    assert with_replan.fault_injected and with_replan.detected is True
    assert with_replan.correct and with_replan.false_alarms == 0


def _assert_gate_fails(kind: Kind, mode: Mode, change: dict, failing_gate: str) -> None:
    outcomes = [
        dataclasses.replace(o, **change) if (o.kind, o.mode) == (kind, mode) else o
        for o in _clean_results()
    ]

    failed = [g.name for g in gates(summarize(outcomes), Timings()) if not g.passed]

    assert any(name.startswith(failing_gate) for name in failed)


CLEAN_TRIAL = CrashTrial(
    seed=0,
    review=False,
    killed=True,
    completed=True,
    correct=True,
    reruns=0,
    lost_attempts=1,
    seconds=2.0,
)


@pytest.mark.parametrize("review", [False, True])
async def test_killed_run_resumes_without_re_running_finished_work(redis, redis_url, review):
    """Phase 7 exit test, one trial. The scorecard runs ten."""
    trial = await run_trial(redis, redis_url, seed=0, review=review)

    assert trial.killed and trial.correct
    assert trial.reruns == 0
    assert await redis.dbsize() == 0


async def test_crash_score_counts_work_on_a_subgoal_after_its_output_was_recorded(board):
    scenario = build(Kind.HEALTHY, seed=0)
    await board.submit(scenario.specs)
    first = scenario.specs[0].id
    await board.log_event("work.started", id=first, attempt="1")
    await board.log_event("subgoal.done", id=first)
    await board.log_event("work.started", id=first, attempt="2")

    trial = await _score(board, scenario, seed=0, review=False, killed=True, seconds=1.0)

    assert trial.reruns == 1 and not trial.completed


@pytest.mark.parametrize(
    ("change", "failing_gate"),
    [
        ({"killed": False}, "every crash trial killed the process mid-run"),
        ({"correct": False}, "every killed run resumes to a correct result"),
        ({"reruns": 1}, "no finished subgoal is re-run after a crash"),
    ],
)
def test_a_crash_regression_fails_its_gate(change, failing_gate):
    trials = [CLEAN_TRIAL, dataclasses.replace(CLEAN_TRIAL, seed=1, **change)]

    failed = [g.name for g in crash_gates(trials) if not g.passed]

    assert failed == [failing_gate]


def test_no_crash_trials_fails_the_gates():
    assert not all(g.passed for g in crash_gates([]))
