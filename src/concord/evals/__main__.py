"""Command line entry point: `python -m concord.evals`."""

import argparse
import asyncio
import dataclasses
import json
import os
import statistics
import sys
from pathlib import Path

from redis.asyncio import Redis

from concord.anthropic_client import DEFAULT_MODEL, PROVIDER, AnthropicClient
from concord.corpus import Corpus
from concord.evals.runner import PRICES, Result, cost_usd, evaluate
from concord.harness.report import table
from concord.recording import RecordingClient, ReplayClient
from concord.tracing import TracedModelClient, configure

# A frozen copy of the project's documents as they were when the questions were first
# run. Reading the live files would change the prompts whenever the docs are edited,
# which would break replaying a recording.
CORPUS_FILES = ["evals/corpus/README.md", "evals/corpus/ROADMAP.md", "evals/corpus/tracing.md"]
CORPUS_DESCRIPTION = (
    "the documentation of Concord, a multi-agent runtime: its design, its supervisor, "
    "its measured scorecard and its tracing"
)


def _money(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.2f}"


def report(results: list[Result], model: str) -> str:
    def cost(answer) -> float | None:
        return cost_usd(model, answer.input_tokens, answer.output_tokens)

    rows = []
    for r in results:
        rows.append(
            [
                r.id,
                "yes" if r.team.completed else f"no: {r.team.error[:40]}",
                f"{r.team.subgoals}",
                f"{_money(cost(r.single))} -> {_money(cost(r.team))}",
                f"{r.single.seconds:.0f}s -> {r.team.seconds:.0f}s",
                r.verdict.winner if r.verdict else "not judged",
            ]
        )
    headers = [
        "Question",
        "Team finished",
        "Subgoals",
        "Token cost (single -> team)",
        "Time (single -> team)",
        "Better answer",
    ]
    done = [r for r in results if r.team.completed]
    judged = [r.verdict.winner for r in results if r.verdict]
    lines = [
        f"Concord evaluation: {len(results)} questions on {model}",
        "Costs are token spend only; web search fees are not included. "
        "'Better answer' is a blind comparison by the same model, not a human rating.",
        table(headers, rows),
        f"Team finished {len(done)} of {len(results)}; "
        f"single agent finished {sum(r.single.completed for r in results)} of {len(results)}.",
    ]
    if judged:
        lines.append(
            f"Better answer: team {judged.count('team')}, single agent "
            f"{judged.count('single')}, tie {judged.count('tie')} (of {len(judged)} judged)."
        )
    if results and model in PRICES:
        team_total = sum(cost(r.team) for r in results)
        single_total = sum(cost(r.single) for r in results)
        lines.append(
            f"Total token cost: team {_money(team_total)}, single agent {_money(single_total)}."
        )
        lines.append(
            "Median time: team "
            f"{statistics.median(r.team.seconds for r in results):.0f}s, single agent "
            f"{statistics.median(r.single.seconds for r in results):.0f}s."
        )
    return "\n\n".join(lines)


async def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m concord.evals",
        description="Answer real questions with the agent team and with a single agent, "
        "then compare cost, time and answer quality.",
    )
    parser.add_argument("--questions", default="evals/questions.json")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium", help="low, medium, high, xhigh or max")
    parser.add_argument("--out", default="evals/results/latest.json")
    parser.add_argument("--record", metavar="PATH", help="save every model reply to this file")
    parser.add_argument("--replay", metavar="PATH", help="answer from a recording; no API calls")
    parser.add_argument("--max-cost", type=float, default=25.0, help="stop starting new questions "
                        "once estimated token spend reaches this many dollars")  # fmt: skip
    parser.add_argument("--trace", action="store_true", help="send traces to the local Jaeger")
    parser.add_argument(
        "--drift-check",
        action="store_true",
        help="have the supervisor review every output before it is used (one more call each)",
    )
    parser.add_argument("--yes", action="store_true", help="confirm that this spends money")
    parser.add_argument(
        "--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/12")
    )
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())[: args.limit]
    if args.replay:
        client = ReplayClient(args.replay)
    else:
        if not args.yes:
            print(
                f"This calls {args.model} for {len(questions)} questions, three ways each "
                f"(team, single agent, judge), with web search. It spends real money: "
                f"up to about ${args.max_cost:.0f} in tokens before it stops, plus web "
                "search fees. Re-run with --yes to go ahead, or --limit 1 to try one first."
            )
            return 2
        client = AnthropicClient(model=args.model, effort=args.effort)
        if args.record:
            client = RecordingClient(client, args.record)
    if args.trace:
        configure(service_name="concord-evals")
        client = TracedModelClient(client, model=args.model, provider=PROVIDER)

    corpus = Corpus.from_files(CORPUS_FILES)
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []
    spent = 0.0
    try:
        for n, item in enumerate(questions, 1):
            if spent >= args.max_cost:
                print(f"Stopping before question {n}: spent {_money(spent)} of the "
                      f"{_money(args.max_cost)} limit.", file=sys.stderr)  # fmt: skip
                break
            print(f"[{n}/{len(questions)}] {item['id']}", file=sys.stderr)
            result = await evaluate(
                redis,
                client,
                item["id"],
                item["question"],
                corpus,
                corpus_description=CORPUS_DESCRIPTION,
                drift_check=args.drift_check,
            )
            results.append(result)
            tokens_in = result.team.input_tokens + result.single.input_tokens
            tokens_out = result.team.output_tokens + result.single.output_tokens
            if result.verdict:
                tokens_in += result.verdict.input_tokens
                tokens_out += result.verdict.output_tokens
            spent += cost_usd(args.model, tokens_in, tokens_out) or 0.0
            # Written after every question so an interrupted run keeps what it paid for.
            payload = {"model": args.model, "effort": args.effort,
                       "results": [dataclasses.asdict(r) for r in results]}  # fmt: skip
            out.write_text(json.dumps(payload, indent=2))
    finally:
        await redis.aclose()

    print(report(results, args.model))
    print(f"\nFull answers and details: {out}")
    return 0 if results and all(r.team.completed for r in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
