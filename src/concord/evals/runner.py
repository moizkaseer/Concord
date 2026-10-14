"""Answers one question two ways and compares the answers."""

import asyncio
import contextlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from redis.asyncio import Redis

from concord.blackboard import Blackboard
from concord.corpus import Corpus
from concord.drift import ModelDriftChecker
from concord.llm import Completion, ModelClient
from concord.models import RunState, State, Subgoal
from concord.planner import Planner
from concord.replanner import ModelReplanner
from concord.specialists import specialist_handler
from concord.supervisor import Supervisor
from concord.tracing import workflow_span
from concord.worker import WorkerPool

# Dollars per million tokens (input, output). Web search fees and cache discounts are
# not included, so costs are estimates of token spend only.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}

SINGLE_SYSTEM = """\
Answer the question for the person who asked it. Search the web for what you need and \
prefer primary sources such as official documentation. Keep the source next to each \
claim it supports (a URL, or the name of a provided passage in square brackets). If \
sources disagree or you could not find something, say so plainly rather than filling the \
gap from memory. Lead with the direct answer, then the supporting detail."""

JUDGE_SYSTEM = """\
You are comparing two answers to the same research question, written by two different \
systems. Decide which answer would serve the person who asked better.

Judge on three things, in this order of importance:
1. Correctness: are the claims accurate as far as you can tell, and does the answer \
avoid stating things it gives no support for?
2. Coverage: does it address every part of the question?
3. Sourcing: are claims tied to specific sources a reader could check?

Length is not a virtue; a longer answer wins only if the extra material is correct and \
relevant. Answer "tie" when neither is clearly better. Give the reason in one or two \
sentences that name the specific difference that decided it."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "reason": {"type": "string"},
    },
    "required": ["winner", "reason"],
    "additionalProperties": False,
}


class Meter:
    """Wraps a client and adds up everything spent through it."""

    def __init__(self, client: ModelClient) -> None:
        self._client = client
        self.calls = self.input_tokens = self.output_tokens = self.searches = 0

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> Completion:
        completion = await self._client.complete(
            prompt, system=system, schema=schema, web_search=web_search
        )
        self.calls += 1
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.searches += completion.searches
        return completion


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Token cost in dollars, or None for a model with no listed price."""
    if model not in PRICES:
        return None
    input_price, output_price = PRICES[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


@dataclass
class Answer:
    completed: bool
    text: str = ""
    error: str = ""
    seconds: float = 0.0
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0
    subgoals: int = 0
    interventions: list[str] = field(default_factory=list)

    def spend(self, meter: Meter, started: float) -> "Answer":
        self.seconds = round(time.monotonic() - started, 2)
        self.model_calls = meter.calls
        self.input_tokens = meter.input_tokens
        self.output_tokens = meter.output_tokens
        self.searches = meter.searches
        return self


async def answer_with_team(
    redis: Redis,
    client: ModelClient,
    question: str,
    corpus: Corpus | None,
    *,
    corpus_description: str,
    token_budget: int = 600_000,
    timeout: float = 900.0,
    lease_ttl: float = 30.0,
    handler_timeout: float = 400.0,
    drift_check: bool = False,
    replan: bool = False,
) -> Answer:
    """Plan the question, run the plan under the supervisor, return the writer's output.

    With `drift_check`, the supervisor reviews every output before the subgoals that
    depend on it may use it, at the cost of one more model call per subgoal. With `replan`,
    a subgoal that fails on every attempt is re-planned around once per run.
    """
    meter = Meter(client)
    started = time.monotonic()
    board = Blackboard(redis, f"eval-{uuid.uuid4().hex[:12]}")
    answer = Answer(completed=False)
    try:
        with workflow_span("research-question", board.run_id):
            planner = Planner(meter, corpus_description=corpus_description)
            plan = await planner.plan(question)
            answer.subgoals = len(plan.subgoals)
            await board.submit(
                plan.subgoals, supervised=True, review=drift_check, replans=int(replan)
            )
            checker = ModelDriftChecker(meter, question) if drift_check else None
            replanner = ModelReplanner(meter, question) if replan else None
            pool = WorkerPool(
                board,
                specialist_handler(meter, question, corpus),
                lease_ttl=lease_ttl,
                handler_timeout=handler_timeout,
            )
            supervisor = asyncio.create_task(
                Supervisor(
                    board,
                    checker=checker,
                    replanner=replanner,
                    token_budget=token_budget,
                    tick=0.2,
                ).run()
            )
            try:
                status = await pool.run(timeout=timeout)
            finally:
                # The supervisor stops by itself once the run has ended; if the run
                # hung or was interrupted, the timeout cancels it.
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(5):
                        await supervisor
        answer.interventions = [e["type"] for e in await board.interventions()]
        final = await _final_subgoal(board, plan.answer_id)
        if status.state == RunState.COMPLETED:
            answer.completed = True
            answer.text = final.output or ""
        else:
            cause = final.error
            answer.error = ": ".join(filter(None, [status.reason, cause])) or str(status.state)
    except Exception as exc:
        answer.error = f"{type(exc).__name__}: {exc}"
    finally:
        await board.destroy()
    return answer.spend(meter, started)


async def _final_subgoal(board: Blackboard, answer_id: str) -> Subgoal:
    """The planned answer subgoal, or the one a re-plan put in its place."""
    subgoal = await board.get(answer_id)
    while subgoal.state == State.REPLACED:
        # A replaced final subgoal always has exactly one stand-in (see check_patch).
        subgoal = await board.get(subgoal.replaced_by[0])
    return subgoal


async def answer_alone(
    client: ModelClient, question: str, corpus: Corpus | None, *, passages: int = 6
) -> Answer:
    """The baseline: one model call with the same web search and the same documents."""
    meter = Meter(client)
    started = time.monotonic()
    prompt = f"Question: {question}"
    if corpus is not None:
        found = corpus.search(question, k=passages)
        if found:
            shown = "\n\n".join(f"[{p.source}]\n{p.text}" for p in found)
            prompt += f"\n\nPassages from the project's own documents:\n\n{shown}"
    answer = Answer(completed=False)
    try:
        completion = await meter.complete(prompt, system=SINGLE_SYSTEM, web_search=True)
        answer.completed, answer.text = True, completion.text
    except Exception as exc:
        answer.error = f"{type(exc).__name__}: {exc}"
    return answer.spend(meter, started)


@dataclass
class Verdict:
    winner: str  # "team", "single" or "tie"
    reason: str
    input_tokens: int = 0
    output_tokens: int = 0


async def judge(client: ModelClient, question_id: str, question: str, team: str, single: str):
    """Blind comparison. Which answer is shown first is fixed per question, so a rerun
    judges the same pair in the same order."""
    team_first = random.Random(question_id).random() < 0.5
    first, second = (team, single) if team_first else (single, team)
    prompt = f"Question: {question}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}"
    completion = await client.complete(prompt, system=JUDGE_SYSTEM, schema=JUDGE_SCHEMA)
    reply = json.loads(completion.text)
    winner = {"A": "team" if team_first else "single", "B": "single" if team_first else "team"}
    return Verdict(
        winner.get(reply["winner"], "tie"),
        reply["reason"],
        completion.input_tokens,
        completion.output_tokens,
    )


@dataclass
class Result:
    id: str
    question: str
    team: Answer
    single: Answer
    verdict: Verdict | None


async def evaluate(
    redis: Redis,
    client: ModelClient,
    question_id: str,
    question: str,
    corpus: Corpus | None,
    *,
    corpus_description: str,
    **team_options: Any,
) -> Result:
    team = await answer_with_team(
        redis, client, question, corpus, corpus_description=corpus_description, **team_options
    )
    single = await answer_alone(client, question, corpus)
    verdict = None
    if team.completed and single.completed:
        try:
            verdict = await judge(client, question_id, question, team.text, single.text)
        except Exception as exc:
            verdict = Verdict("tie", f"judge failed: {type(exc).__name__}: {exc}")
    return Result(question_id, question, team, single, verdict)
