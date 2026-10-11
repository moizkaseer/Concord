"""Measures the drift check against labelled outputs: `python -m concord.evals.drift`.

The labelled set is built from the recorded evaluation runs. Each case pairs a subgoal
with an output:

* `own`: the output the specialist really produced for that subgoal, labelled on-task;
* `sibling`: the output of a different subgoal of the same question, labelled drifted.
  These are the hard cases: right topic, wrong piece of work;
* `stranger`: the output of a subgoal from a different question, labelled drifted.

The labels follow from how each case was built, not from a person reading every output,
so a few may be wrong: a real output can itself be off-task, and two sibling subgoals can
overlap. `cases.json` is plain data so that a label can be corrected by hand, and the
report lists every case where the reviewer disagreed with its label, which is where a
wrong label would show up.
"""

import argparse
import asyncio
import dataclasses
import json
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from concord.anthropic_client import DEFAULT_MODEL, AnthropicClient
from concord.corpus import Corpus
from concord.drift import RULES, DriftVerdict, ModelDriftChecker, Outcome, decide, review_prompt
from concord.evals.__main__ import CORPUS_DESCRIPTION, CORPUS_FILES
from concord.evals.runner import PRICES
from concord.harness.report import table
from concord.llm import ModelClient
from concord.models import Subgoal
from concord.planner import Planner
from concord.recording import MissingRecording, RecordingClient, ReplayClient
from concord.specialists import specialist_handler

ON_TASK, DRIFTED = "on_task", "drifted"
# A reply is a short reason and five booleans.
_REPLY_TOKENS = 150


@dataclass(frozen=True)
class Item:
    """One subgoal of a recorded run, with what its specialist was given and produced."""

    question_id: str
    question: str
    subgoal: Subgoal
    inputs: dict[str, str]
    output: str


async def replay_runs(
    client: ModelClient, questions: list[dict[str, str]], corpus: Corpus | None
) -> dict[tuple[str, str], Item]:
    """Re-run each recorded question's plan in dependency order and collect every
    subgoal's output. Questions the recording does not cover are left out."""
    items: dict[tuple[str, str], Item] = {}
    for entry in questions:
        question_id, question = entry["id"], entry["question"]
        try:
            plan = await Planner(client, corpus_description=CORPUS_DESCRIPTION).plan(question)
            handler = specialist_handler(client, question, corpus)
            outputs: dict[str, str] = {}
            found = []
            waiting = [Subgoal(**spec.model_dump()) for spec in plan.subgoals]
            while waiting:
                subgoal = next(s for s in waiting if all(d in outputs for d in s.depends_on))
                waiting.remove(subgoal)
                inputs = {dep: outputs[dep] for dep in subgoal.depends_on}
                outputs[subgoal.id] = (await handler(subgoal, inputs)).output
                found.append(Item(question_id, question, subgoal, inputs, outputs[subgoal.id]))
        except MissingRecording:
            continue
        items.update({(item.question_id, item.subgoal.id): item for item in found})
    return items


def build_cases(
    items: dict[tuple[str, str], Item],
    *,
    seed: int = 0,
    own: int = 60,
    sibling: int = 20,
    stranger: int = 20,
) -> list[dict[str, str]]:
    """Choose which subgoal is paired with which output. The same seed gives the same set."""
    rng = random.Random(seed)
    keys = sorted(items)

    def candidates(key: tuple[str, str], *, same_question: bool) -> list[tuple[str, str]]:
        target = items[key].subgoal
        return [
            other
            for other in keys
            if other != key
            and (other[0] == key[0]) == same_question
            and items[other].subgoal.specialist == target.specialist
            # An output the subgoal was itself built from is not a clean wrong answer.
            and other[1] not in target.depends_on
            and key[1] not in items[other].subgoal.depends_on
        ]

    def swaps(count: int, *, same_question: bool) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        pool = [key for key in keys if candidates(key, same_question=same_question)]
        chosen = rng.sample(pool, min(count, len(pool)))
        return [(key, rng.choice(candidates(key, same_question=same_question))) for key in chosen]

    plan = [("own", ON_TASK, key, key) for key in rng.sample(keys, min(own, len(keys)))]
    plan += [("sibling", DRIFTED, *pair) for pair in swaps(sibling, same_question=True)]
    plan += [("stranger", DRIFTED, *pair) for pair in swaps(stranger, same_question=False)]
    return [
        {
            "id": f"c{n:03d}",
            "kind": kind,
            "label": label,
            "question": target[0],
            "subgoal": target[1],
            "output_question": source[0],
            "output_subgoal": source[1],
        }
        for n, (kind, label, target, source) in enumerate(plan, 1)
    ]


def _parts(case: dict[str, str], items: dict[tuple[str, str], Item]) -> tuple[Item, str]:
    target = items[(case["question"], case["subgoal"])]
    return target, items[(case["output_question"], case["output_subgoal"])].output


@dataclass
class Review:
    case: dict[str, str]
    verdict: DriftVerdict | None
    error: str = ""


async def review_cases(
    client: ModelClient,
    cases: list[dict[str, str]],
    items: dict[tuple[str, str], Item],
    *,
    concurrency: int = 4,
) -> list[Review]:
    gate = asyncio.Semaphore(concurrency)

    async def one(case: dict[str, str]) -> Review:
        target, output = _parts(case, items)
        async with gate:
            try:
                checker = ModelDriftChecker(client, target.question)
                return Review(case, await checker.check(target.subgoal, target.inputs, output))
            except Exception as exc:
                return Review(case, None, f"{type(exc).__name__}: {exc}")

    return list(await asyncio.gather(*(one(case) for case in cases)))


@dataclass(frozen=True)
class Score:
    """How one rule did, counting "drifted" as the thing to be found."""

    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    unclear: int

    @property
    def precision(self) -> float | None:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else None

    @property
    def recall(self) -> float | None:
        drifted = self.true_positives + self.false_negatives
        return self.true_positives / drifted if drifted else None


def score(reviews: list[Review], rule: str) -> Score:
    counts: dict[tuple[bool, bool], int] = defaultdict(int)
    unclear = 0
    for review in reviews:
        if review.verdict is None:
            continue
        outcome = decide(review.verdict.answers, rule)
        unclear += outcome is Outcome.UNCLEAR
        # An unclear verdict keeps the output, so it does not count as flagging it.
        counts[(review.case["label"] == DRIFTED, outcome is Outcome.DRIFTED)] += 1
    return Score(
        counts[(True, True)], counts[(False, True)], counts[(True, False)], counts[(False, False)],
        unclear,
    )  # fmt: skip


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def report(reviews: list[Review], model: str, rule: str) -> str:
    judged = [r for r in reviews if r.verdict is not None]
    rows = []
    for name in RULES:
        s = score(judged, name)
        rows.append(
            [
                name + (" (in use)" if name == rule else ""),
                ", ".join(RULES[name]),
                _share(s.precision),
                _share(s.recall),
                str(s.false_positives),
                str(s.false_negatives),
                str(s.unclear),
            ]
        )
    headers = [
        "Rule",
        "An output is drifted if any of these is 'no'",
        "Precision",
        "Recall",
        "On-task flagged",
        "Drifted missed",
        "Unclear",
    ]
    by_kind = []
    for kind in ("own", "sibling", "stranger"):
        group = [r for r in judged if r.case["kind"] == kind]
        flagged = sum(decide(r.verdict.answers, rule) is Outcome.DRIFTED for r in group)
        by_kind.append([kind, group[0].case["label"] if group else "", f"{flagged}/{len(group)}"])
    wrong = [
        r for r in judged
        if (decide(r.verdict.answers, rule) is Outcome.DRIFTED) != (r.case["label"] == DRIFTED)
    ]  # fmt: skip
    lines = [
        f"Drift check on {len(reviews)} labelled outputs, reviewed by {model}",
        "Labels come from how each case was built, not from a person reading it. "
        "Read the disagreements below before trusting the figures.",
        table(headers, rows),
        table(["Kind of case", "Label", "Flagged as drifted"], by_kind),
    ]
    if len(judged) < len(reviews):
        lines.append(f"{len(reviews) - len(judged)} cases could not be reviewed and are left out.")
    if wrong:
        shown = "\n".join(
            f"- {r.case['id']} ({r.case['kind']}, labelled {r.case['label']}) "
            f"{r.case['question']}/{r.case['subgoal']}: {r.verdict.reason}"
            for r in wrong
        )
        lines.append(f"Disagreements with the label under the rule in use:\n{shown}")
    return "\n\n".join(lines)


def estimate_cost(
    cases: list[dict[str, str]], items: dict[tuple[str, str], Item], model: str
) -> float | None:
    """A rough token cost: four characters to a token."""
    if model not in PRICES:
        return None
    characters = 0
    for case in cases:
        target, output = _parts(case, items)
        characters += len(review_prompt(target.question, target.subgoal, target.inputs, output))
    input_price, output_price = PRICES[model]
    tokens_in, tokens_out = characters / 4, _REPLY_TOKENS * len(cases)
    return (tokens_in * input_price + tokens_out * output_price) / 1_000_000


async def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m concord.evals.drift",
        description="Build the labelled set for the drift check, or measure the check on it.",
    )
    parser.add_argument("command", choices=["build", "run"])
    parser.add_argument("--cases", default="evals/drift/cases.json")
    parser.add_argument("--questions", default="evals/questions.json")
    parser.add_argument("--recording", default="evals/recording.json",
                        help="the recorded evaluation runs the outputs come from")  # fmt: skip
    parser.add_argument("--seed", type=int, default=0, help="build: which cases to choose")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="low", help="low, medium, high, xhigh or max")
    parser.add_argument("--rule", default="core", choices=sorted(RULES))
    parser.add_argument("--out", help="run: where to write every verdict (default: by model)")
    parser.add_argument("--record", metavar="PATH", help="save every review to this file")
    parser.add_argument("--replay", metavar="PATH", help="review from a recording; no API calls")
    parser.add_argument("--yes", action="store_true", help="confirm that this spends money")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    corpus = Corpus.from_files(CORPUS_FILES)
    items = await replay_runs(ReplayClient(args.recording), questions, corpus)
    cases_path = Path(args.cases)

    if args.command == "build":
        cases = build_cases(items, seed=args.seed)
        cases_path.parent.mkdir(parents=True, exist_ok=True)
        cases_path.write_text(json.dumps(cases, indent=2) + "\n")
        kinds = {
            kind: sum(c["kind"] == kind for c in cases) for kind in ("own", "sibling", "stranger")
        }
        print(
            f"Wrote {len(cases)} cases to {cases_path} from {len(items)} recorded outputs: {kinds}"
        )
        return 0

    cases = json.loads(cases_path.read_text())
    if args.replay:
        client: ModelClient = ReplayClient(args.replay)
    else:
        if not args.yes:
            estimate = estimate_cost(cases, items, args.model)
            cost = "an unknown amount" if estimate is None else f"about ${estimate:.2f}"
            print(
                f"This sends {len(cases)} reviews to {args.model}. It spends real money: "
                f"{cost} in tokens. Re-run with --yes to go ahead."
            )
            return 2
        client = AnthropicClient(model=args.model, effort=args.effort, max_tokens=2000)
        if args.record:
            client = RecordingClient(client, args.record)

    reviews = await review_cases(client, cases, items)
    out = Path(args.out or f"evals/drift/results-{args.model}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "effort": args.effort,
        "rule": args.rule,
        "scores": {name: dataclasses.asdict(score(reviews, name)) for name in RULES},
        "reviews": [
            {
                **r.case,
                "error": r.error,
                **(dataclasses.asdict(r.verdict) if r.verdict else {}),
            }
            for r in reviews
        ],
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(report(reviews, args.model, args.rule))
    print(f"\nEvery verdict: {out}")
    return 0 if all(r.verdict for r in reviews) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
