"""Turns scenario outcomes into the scorecard and the pass/fail gates."""

import statistics
from collections import defaultdict
from dataclasses import dataclass

from concord.harness.crash import CrashTrial
from concord.harness.runner import Mode, Outcome, Timings
from concord.harness.scenarios import DUPLICATE_KINDS, HEALTHY_KINDS, Kind

REWORDED_KINDS = (Kind.LIGHT_REWORD, Kind.PARAPHRASE)
# Faults that happen while the run is in progress, as opposed to being part of the graph.
RUNTIME_FAULTS = (Kind.DEAD_WORKER, Kind.HUNG_WORKER, Kind.SUPERVISOR_RESTART, Kind.OFF_TASK)


@dataclass(frozen=True)
class Summary:
    runs: int
    completed: int
    correct: int
    detectable: int
    detected: int
    false_alarm_runs: int
    faults_injected: int
    mean_tokens: float
    detect_times: tuple[float, ...]


def summarize(outcomes: list[Outcome]) -> dict[tuple[Kind, Mode], Summary]:
    groups: dict[tuple[Kind, Mode], list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[(outcome.kind, outcome.mode)].append(outcome)
    return {
        key: Summary(
            runs=len(group),
            completed=sum(o.completed for o in group),
            correct=sum(o.correct for o in group),
            detectable=sum(o.detected is not None for o in group),
            detected=sum(bool(o.detected) for o in group),
            false_alarm_runs=sum(o.false_alarms > 0 for o in group),
            faults_injected=sum(o.fault_injected for o in group),
            mean_tokens=statistics.fmean(o.tokens for o in group),
            detect_times=tuple(o.detect_seconds for o in group if o.detect_seconds is not None),
        )
        for key, group in groups.items()
    }


def _merge(summaries: dict, kinds: tuple[Kind, ...], mode: Mode) -> Summary:
    parts = [summaries[(kind, mode)] for kind in kinds if (kind, mode) in summaries]
    runs = sum(p.runs for p in parts)
    return Summary(
        runs=runs,
        completed=sum(p.completed for p in parts),
        correct=sum(p.correct for p in parts),
        detectable=sum(p.detectable for p in parts),
        detected=sum(p.detected for p in parts),
        false_alarm_runs=sum(p.false_alarm_runs for p in parts),
        faults_injected=sum(p.faults_injected for p in parts),
        mean_tokens=sum(p.mean_tokens * p.runs for p in parts) / runs if runs else 0.0,
        detect_times=tuple(t for p in parts for t in p.detect_times),
    )


def _ratio(part: int, whole: int) -> str:
    return f"{part}/{whole} ({100 * part / whole:.0f}%)" if whole else "n/a"


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(row[n]) for row in [headers, *rows]) for n in range(len(headers))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths, strict=True))
             for row in [headers, ["-" * width for width in widths], *rows]]  # fmt: skip
    return "\n".join(line.rstrip() for line in lines)


def fault_table(summaries: dict[tuple[Kind, Mode], Summary]) -> str:
    """One row per fault type: the run with a supervisor beside the run without one."""
    rows = []
    for kind in Kind:
        base, sup = summaries[(kind, Mode.BASELINE)], summaries[(kind, Mode.SUPERVISED)]
        times = sup.detect_times
        rows.append(
            [
                kind,
                str(sup.runs),
                _ratio(sup.detected, sup.detectable),
                f"{statistics.median(times):.2f}s / {max(times):.2f}s" if times else "n/a",
                f"{_ratio(base.correct, base.runs)} -> {_ratio(sup.correct, sup.runs)}",
                f"{base.mean_tokens:.0f} -> {sup.mean_tokens:.0f}"
                if base.completed == base.runs
                else "n/a, did not finish without",
                str(sup.false_alarm_runs),
            ]
        )
    headers = [
        "Fault",
        "Scenarios",
        "Detected",
        "Time to detect (median / max)",
        "Correct result (without -> with supervisor)",
        "Tokens (without -> with)",
        "Runs with a false alarm",
    ]
    return table(headers, rows)


def similarity_table(summaries: dict[tuple[Kind, Mode], Summary]) -> str:
    """What turning on word-overlap similarity matching buys and costs."""
    rows = []
    for kind in (*DUPLICATE_KINDS, *HEALTHY_KINDS):
        off, on = summaries[(kind, Mode.SUPERVISED)], summaries[(kind, Mode.SIMILARITY)]
        rows.append(
            [
                kind,
                f"{_ratio(off.detected, off.detectable)} -> {_ratio(on.detected, on.detectable)}",
                f"{off.false_alarm_runs} -> {on.false_alarm_runs}",
                f"{_ratio(off.correct, off.runs)} -> {_ratio(on.correct, on.runs)}",
            ]
        )
    headers = [
        "Scenario type",
        "Detected (off -> on)",
        "Runs with a false alarm (off -> on)",
        "Correct result (off -> on)",
    ]
    return table(headers, rows)


def review_table(summaries: dict[tuple[Kind, Mode], Summary]) -> str:
    """What reviewing every output does to a run, given a reviewer that is never wrong."""
    others = tuple(k for k in Kind if k is not Kind.OFF_TASK and k not in HEALTHY_KINDS)
    rows = []
    for label, kinds in (
        (Kind.OFF_TASK, (Kind.OFF_TASK,)),
        ("healthy (both kinds)", HEALTHY_KINDS),
        ("every other fault", others),
    ):
        off, on = _merge(summaries, kinds, Mode.SUPERVISED), _merge(summaries, kinds, Mode.REVIEW)
        flagged = "n/a"
        if kinds == (Kind.OFF_TASK,):
            flagged = (
                f"{_ratio(off.detected, off.detectable)} -> {_ratio(on.detected, on.detectable)}"
            )
        rows.append(
            [
                label,
                flagged,
                f"{off.false_alarm_runs} -> {on.false_alarm_runs}",
                f"{_ratio(off.correct, off.runs)} -> {_ratio(on.correct, on.runs)}",
            ]
        )
    headers = [
        "Scenario type",
        "Off-task output sent back (off -> on)",
        "Runs with a false alarm (off -> on)",
        "Correct result (off -> on)",
    ]
    return table(headers, rows)


def scorecard(
    summaries: dict[tuple[Kind, Mode], Summary], timings: Timings, trials: list[CrashTrial]
) -> str:
    """The roadmap's seven measures, for the supervisor as it ships by default."""
    sup = Mode.SUPERVISED
    exact = summaries[(Kind.EXACT_DUPLICATE, sup)]
    reworded = _merge(summaries, REWORDED_KINDS, sup)
    healthy = _merge(summaries, HEALTHY_KINDS, sup)
    dead = summaries[(Kind.DEAD_WORKER, sup)]
    off_task = summaries[(Kind.OFF_TASK, Mode.REVIEW)]
    with_dupes = _merge(summaries, (Kind.EXACT_DUPLICATE,), sup).mean_tokens
    without = summaries[(Kind.EXACT_DUPLICATE, Mode.BASELINE)].mean_tokens
    limit = 2 * timings.lease_ttl
    slowest = f"{max(dead.detect_times):.2f}s" if dead.detect_times else "n/a"
    rows = [
        ["Exact duplicate subgoals caught", "100%", _ratio(exact.detected, exact.detectable)],
        [
            "Reworded duplicate subgoals caught",
            ">= 90%",
            _ratio(reworded.detected, reworded.detectable),
        ],
        [
            "False alarms on healthy runs",
            "<= 5%",
            _ratio(healthy.false_alarm_runs, healthy.runs),
        ],
        [
            "Time to detect a dead worker",
            f"<= {limit:.2f}s (2 x lease)",
            f"{slowest} at worst",
        ],
        [
            "Off-task output caught",
            "precision and recall >= 0.8",
            f"{_ratio(off_task.detected, off_task.detectable)} sent back, given a reviewer "
            "that is never wrong",
        ],
        [
            "Tokens saved on runs with an exact duplicate",
            "report the figure",
            f"{100 * (1 - with_dupes / without):.1f}% ({without:.0f} -> {with_dupes:.0f})",
        ],
        [
            "Finished subgoals re-run after a crash",
            "0",
            f"{sum(t.reruns for t in trials)} in {len(trials)} kill-and-resume trials",
        ],
    ]
    return table(["Measure", "Target", "Result"], rows)


def crash_table(trials: list[CrashTrial]) -> str:
    """Kill-and-resume trials, split by whether outputs were reviewed."""
    rows = []
    for label, review in (("no review", False), ("drift check on", True)):
        group = [t for t in trials if t.review is review]
        if not group:
            continue
        rows.append(
            [
                label,
                str(len(group)),
                _ratio(sum(t.killed for t in group), len(group)),
                _ratio(sum(t.correct for t in group), len(group)),
                str(sum(t.reruns for t in group)),
                str(sum(t.lost_attempts for t in group)),
                f"{statistics.median(t.seconds for t in group):.1f}s",
            ]
        )
    headers = [
        "Run",
        "Trials",
        "Killed mid-run",
        "Correct result after resume",
        "Finished subgoals re-run",
        "In-flight attempts redone",
        "Median time",
    ]
    return table(headers, rows)


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    detail: str


def gates(summaries: dict[tuple[Kind, Mode], Summary], timings: Timings) -> list[Gate]:
    """Checks that must hold for what is already built. Reworded duplicates are a known
    gap, reported in the scorecard but not gated."""
    sup = Mode.SUPERVISED
    exact = summaries[(Kind.EXACT_DUPLICATE, sup)]
    healthy = _merge(summaries, HEALTHY_KINDS, sup)
    dead = summaries[(Kind.DEAD_WORKER, sup)]
    hung = summaries[(Kind.HUNG_WORKER, sup)]
    restart = summaries[(Kind.SUPERVISOR_RESTART, sup)]
    sound = _merge(summaries, tuple(k for k in Kind if k is not Kind.OFF_TASK), sup)
    reviewed = _merge(summaries, tuple(Kind), Mode.REVIEW)
    off_task = summaries[(Kind.OFF_TASK, Mode.REVIEW)]
    healthy_reviewed = _merge(summaries, HEALTHY_KINDS, Mode.REVIEW)
    runtime = _merge(summaries, RUNTIME_FAULTS, sup)
    limit = 2 * timings.lease_ttl
    slowest = max(dead.detect_times, default=float("inf"))
    return [
        Gate(
            "every runtime fault was really injected",
            runtime.faults_injected == runtime.runs,
            _ratio(runtime.faults_injected, runtime.runs),
        ),
        Gate(
            "every exact duplicate is caught",
            exact.detected == exact.detectable,
            _ratio(exact.detected, exact.detectable),
        ),
        Gate(
            "no false alarms on healthy runs",
            healthy.false_alarm_runs == 0,
            f"{healthy.false_alarm_runs} of {healthy.runs} runs",
        ),
        Gate(
            "every dead worker is detected within 2 x lease",
            dead.detected == dead.detectable and slowest <= limit,
            f"{_ratio(dead.detected, dead.detectable)}, slowest {slowest:.2f}s, limit {limit:.2f}s",
        ),
        Gate(
            "every run with a hung worker still finishes correctly",
            hung.correct == hung.runs,
            _ratio(hung.correct, hung.runs),
        ),
        Gate(
            "every run survives a supervisor restart",
            restart.correct == restart.runs,
            _ratio(restart.correct, restart.runs),
        ),
        Gate(
            "no supervised run ends with a wrong output (off-task scenarios excluded)",
            sound.correct == sound.runs,
            _ratio(sound.correct, sound.runs),
        ),
        Gate(
            "with the drift check on, every off-task output is sent back",
            off_task.detected == off_task.detectable,
            _ratio(off_task.detected, off_task.detectable),
        ),
        Gate(
            "with the drift check on, no run ends with a wrong output",
            reviewed.correct == reviewed.runs,
            _ratio(reviewed.correct, reviewed.runs),
        ),
        Gate(
            "with the drift check on, no false alarms on healthy runs",
            healthy_reviewed.false_alarm_runs == 0,
            f"{healthy_reviewed.false_alarm_runs} of {healthy_reviewed.runs} runs",
        ),
    ]


def crash_gates(trials: list[CrashTrial]) -> list[Gate]:
    killed = sum(t.killed for t in trials)
    correct = sum(t.correct for t in trials)
    reruns = sum(t.reruns for t in trials)
    return [
        Gate(
            "every crash trial killed the process mid-run",
            bool(trials) and killed == len(trials),
            _ratio(killed, len(trials)),
        ),
        Gate(
            "every killed run resumes to a correct result",
            bool(trials) and correct == len(trials),
            _ratio(correct, len(trials)),
        ),
        Gate("no finished subgoal is re-run after a crash", reruns == 0, f"{reruns} re-runs"),
    ]


def render(
    outcomes: list[Outcome], timings: Timings, trials: list[CrashTrial]
) -> tuple[str, list[Gate]]:
    summaries = summarize(outcomes)
    scenarios = len({o.scenario for o in outcomes})
    checks = gates(summaries, timings) + crash_gates(trials)
    sections = [
        f"Concord fault-injection harness: {scenarios} scenarios, {len(outcomes)} runs",
        "Mock model throughout. Token figures are word counts, so only the relative\n"
        "difference between runs means anything.",
        "SCORECARD (supervisor with its default settings)\n\n"
        + scorecard(summaries, timings, trials),
        "BY FAULT TYPE\n\n" + fault_table(summaries),
        "SIMILARITY MATCHING (off by default; word-overlap embedder, threshold 0.9)\n\n"
        + similarity_table(summaries),
        "DRIFT CHECK (off by default; stand-in reviewer that is never wrong)\n\n"
        + review_table(summaries),
        "KILL AND RESUME (process killed with SIGKILL, run resumed by a new process; "
        "Redis stays up)\n\n" + crash_table(trials),
        "GATES\n\n"
        + "\n".join(f"{'PASS' if g.passed else 'FAIL'}  {g.name}: {g.detail}" for g in checks),
    ]
    return "\n\n".join(sections), checks
