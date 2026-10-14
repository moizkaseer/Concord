"""Command line entry point: `python -m concord.harness`."""

import argparse
import asyncio
import dataclasses
import json
import os
import sys

from redis.asyncio import Redis

from concord.harness.crash import run_trials
from concord.harness.report import render
from concord.harness.runner import Timings, run_all
from concord.harness.scenarios import build_all


async def _main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m concord.harness",
        description="Run the fault-injection scenarios and print the scorecard.",
    )
    parser.add_argument("--seeds", type=int, default=8, help="scenarios per fault type")
    parser.add_argument("--check", action="store_true", help="exit with status 1 if a gate fails")
    parser.add_argument(
        "--crash-trials", type=int, default=10, help="kill-and-resume trials to run"
    )
    parser.add_argument("--json", metavar="PATH", help="also write every run's outcome as JSON")
    parser.add_argument(
        "--redis-url",
        default=os.environ.get("REDIS_URL", "redis://localhost:6379/14"),
        help="Redis to run against (default: REDIS_URL, or database 14 on localhost)",
    )
    args = parser.parse_args()

    timings = Timings()
    redis = Redis.from_url(args.redis_url, decode_responses=True)
    try:
        outcomes = await run_all(redis, build_all(args.seeds), timings=timings)
        trials = await run_trials(redis, args.redis_url, args.crash_trials)
    finally:
        await redis.aclose()

    text, checks = render(outcomes, timings, trials)
    print(text)
    if args.json:
        with open(args.json, "w") as out:
            runs = [dataclasses.asdict(o) for o in outcomes]
            crashes = [dataclasses.asdict(t) for t in trials]
            json.dump({"scenarios": runs, "crash_trials": crashes}, out, indent=2)
    return 1 if args.check and not all(g.passed for g in checks) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
