# Concord

A reliability layer for multi-agent runs. A planner splits a task into subgoals, workers run
them from a shared blackboard, and a supervisor catches stalls, repeated work and off-task
output.

**Status: early development.** The blackboard, scheduler, worker pool, supervisor and
fault-injection harness and tracing work and are measured against a mock model. The planner,
specialists and Claude client have run on 19 real questions with Claude Sonnet 5.5; see
[Results on real questions](#results-on-real-questions). The drift check is built and its
runtime behaviour is measured; how often its model reviewer is right has not been measured
yet (see [Drift check](#drift-check)). A run survives the process working it being killed
and resumed by a new one (see [Resuming a run](#resuming-a-run)), and can re-plan around a
subgoal that fails on every attempt (see [Re-planning](#re-planning)); the model re-planner
has not been run against the real API.
See [ROADMAP.md](ROADMAP.md) for the build plan.

## What works today

- **Graph validation**: a submitted subgoal graph must be acyclic, reference only known
  subgoals, and stay within size and depth limits.
- **Blackboard**: subgoal records live in Redis. Every state change is an atomic Lua script
  that also appends an event to a Redis Stream.
- **Scheduling**: a subgoal is released when all its dependencies are done.
- **Workers**: an asyncio pool leases ready subgoals, keeps the lease alive with heartbeats
  and writes the output back. A failed or timed-out attempt is retried up to a limit; after
  that the subgoal fails along with everything that depends on it.
- **Supervisor**: reads the event stream through a consumer group, so it resumes after a
  restart. It
  - requeues subgoals whose worker stopped sending heartbeats;
  - checks each subgoal before dispatch and reuses the output of an earlier duplicate
    (same normalized text and same inputs, or embedding similarity above a threshold);
  - reviews each finished output, when the run asks for it, and sends back one that did
    not do the work its subgoal asked for (see [Drift check](#drift-check));
  - re-plans around a subgoal that ran out of attempts, when the run allows it (see
    [Re-planning](#re-planning));
  - halts the run when it goes over its token budget;
  - logs every intervention as a `supervisor.*` event with its reason.
- **Resuming**: a run's state is all in Redis, so a new process can pick up a run whose
  process died, without redoing any output already recorded.

- **Tracing**: one run is one OpenTelemetry trace, with spans for the run, each
  pre-dispatch check, each attempt, model and tool calls, and blackboard writes, named after
  the GenAI semantic conventions. An attempt lost to a dead worker is recorded by the
  supervisor, so the failure, the requeue and the retry all show up. See
  [docs/tracing.md](docs/tracing.md).

- **Planner**: asks the model for a subgoal graph as structured output, checks it (no
  cycles, exactly one writer, no work that reaches no answer) and sends a rejected plan
  back once with the reason.
- **Specialists**: research (web search), retrieval (keyword search over local documents)
  and synthesis, the only one that writes the final answer.
- **Claude client**: retries, timeouts, token accounting, web search and structured output.
  Replies can be recorded to a file and replayed without the model.

Similarity matching currently ships with a word-overlap embedder for tests. It catches
reordered or lightly edited subgoals, not true paraphrases; those need a real embedding
model, planned for the real-model phase.

```python
import asyncio

from redis.asyncio import Redis

from concord.blackboard import Blackboard
from concord.llm import MockModelClient
from concord.models import SubgoalSpec
from concord.supervisor import Supervisor
from concord.worker import WorkerPool, model_handler

redis = Redis.from_url("redis://localhost:6379/0", decode_responses=True)
board = Blackboard(redis, run_id="demo")
await board.submit(
    [
        SubgoalSpec(id="search", description="find sources"),
        SubgoalSpec(id="answer", description="write the answer", depends_on=["search"]),
    ],
    supervised=True,
)
pool = WorkerPool(board, model_handler(MockModelClient(default="ok")))
status, _ = await asyncio.gather(pool.run(), Supervisor(board, token_budget=50_000).run())
print(await board.interventions())
```

## Scorecard

Measured by the fault-injection harness: 80 scripted scenarios, each with one injected
fault, played with and without the supervisor. Everything runs on a mock model, so the token
figures are word counts and only the relative difference means anything.

| Measure | Target | Result |
|---------|--------|--------|
| Exact duplicate subgoals caught | 100% | 8/8 (100%) |
| Reworded duplicate subgoals caught | ≥ 90% | 0/16 (0%) |
| False alarms on healthy runs | ≤ 5% | 0/16 (0%) |
| Time to detect a dead worker | ≤ 0.80s (2 × lease) | about 0.4s at worst |
| Off-task output caught | precision and recall ≥ 0.8 | reviewer accuracy not measured yet; 8/8 sent back given a reviewer that is never wrong |
| Runs that recover from a permanently failing subgoal | recover through a re-plan | 8/8 with re-planning on, given a re-planner whose patch works; 0/8 without |
| Tokens saved on runs with an exact duplicate | report the figure | 11.2% |
| Finished subgoals re-run after a crash | 0 | 0 in 10 kill-and-resume trials |

What the supervisor changes, by fault type:

| Fault | Correct result without supervisor | With supervisor |
|-------|-----------------------------------|-----------------|
| Dead worker | 0/8, the run hangs | 8/8 |
| Exact duplicate | 8/8, but the work is done twice | 8/8, duplicate never reaches a worker |
| Hung worker | 8/8 | 8/8 (caught by the worker's own deadline, not the supervisor) |
| Supervisor restart mid-run | not applicable | 8/8 |
| Off-task output | 0/8 | 0/8 by default; 8/8 with the drift check on and a reviewer that is never wrong |
| Permanently failing subgoal | 0/8 | 0/8 by default; 8/8 with re-planning on |

Two gaps are visible and left in on purpose:

- **Reworded duplicates are not caught by default.** Similarity matching is off unless an
  embedder is supplied. With the built-in word-overlap embedder switched on, light edits
  are caught (7/8 here, 38/40 in a larger run) and true paraphrases are not (0/40). In
  that larger run it also reused the wrong output once in 40 look-alike runs, where two
  subgoals differed only in their topic. Word overlap cannot tell which word changed, so
  it stays off until a real embedding model is plugged in.
- **Off-task output is only caught with the drift check on**, which is off by default
  because it costs a model call per subgoal. The 8/8 above shows what the runtime does
  with a correct verdict. It says nothing about how often a model reviewer gives one.

Reproduce it with `uv run python -m concord.harness`. CI runs the same command with
`--check`, which fails the build if a gate for an already-built detector regresses.

## Resuming a run

Everything about a run lives in Redis: the subgoal records, the queue, the leases and the
event stream. If the process running the workers and the supervisor dies, start new ones on
the same run id:

```python
board = Blackboard(redis, run_id)  # the run that was in progress; do not submit again
pool = WorkerPool(board, handler)
await asyncio.gather(pool.run(), Supervisor(board).run())
```

What the new process does with what the old one left behind:

- A subgoal that was being worked on is still leased to a worker that no longer exists.
  Once the lease runs out the supervisor puts it back in the queue, and that lost attempt
  counts toward the subgoal's attempt limit.
- Events the old supervisor received but had not finished handling are handed to the new
  one, whatever consumer name either used.
- An output already recorded is never produced again: a subgoal in `reviewing` is reviewed
  by the new supervisor, and one that is `done` stays done.
- Worker names carry a per-pool id, so the new workers can never be mistaken for the old
  ones that held the leases.

**What has been measured.** The harness runs ten trials. Each submits a run, starts a
separate process that works it, kills that process with SIGKILL at a random point once at
least one subgoal has finished, and starts a new process on the same run. Half the trials
review every output. All ten end with every expected output, and no subgoal was worked on
after its output had been recorded. One or two attempts that were in flight at the kill
are redone in each trial; that is the cost of a crash, and it is reported. A local run of
60 trials gave the same result.

**What has not.**

- Losing Redis. The trials keep Redis up. `docker-compose.yml` turns on Redis's
  append-only file, written to disk once a second, so a Redis crash can lose up to the
  last second of state changes; that case is not tested.
- More than one kill per run, and a process that is cut off rather than killed. Lease
  ownership is checked on every write, so a cut-off worker cannot overwrite a newer
  attempt, but no trial exercises it.
- Runs without a supervisor. Nothing but the supervisor takes back an expired lease, so
  an unsupervised run whose process dies stays stuck on the subgoals it had leased.

A resumed run waits up to one lease length (30 seconds by default) before redoing the work
that was in flight.

## Re-planning

A subgoal can fail on every attempt for a reason retrying will not fix: a source that
cannot be reached, or a request no specialist can satisfy as worded. Without re-planning,
it fails and takes everything that depends on it down with it.

With `replans=N` at submit, a subgoal that runs out of attempts moves to `replanning`
instead, and its dependents stay blocked. The supervisor shows a re-planner the question,
the whole graph with each subgoal's state, the outputs finished so far and the error, and
asks for a patch:

- subgoals to add, which may depend on finished subgoals and on each other;
- which subgoals take the failed one's place: everything that depended on it depends on
  those instead. Empty means the dependents go ahead without it. A failed final subgoal
  must name exactly one stand-in, which produces the run's result.

The re-planner proposes and the code decides. The supervisor checks every patch, whoever
wrote it, before one Lua script applies it: new ids must be new, nothing may depend on a
failed subgoal, and the graph as it would be must have no cycles and stay inside the size
and depth limits. The failed subgoal becomes `replaced` and no longer counts against the
run. What can go wrong is bounded:

- A run gets `N` re-plans in all. Once they are used, a subgoal that runs out of attempts
  fails as before, so a re-plan whose new subgoals fail too costs at most `N` re-plans.
- If the re-planner errors or its patch fails the checks, the subgoal fails as it would
  have without re-planning, and `supervisor.replan_failed` records why.
- A re-plan in flight when the supervisor stops is finished by the next one.

```python
from concord.replanner import ModelReplanner

await board.submit(specs, supervised=True, replans=1)
supervisor = Supervisor(board, replanner=ModelReplanner(client, question))
```

`ModelReplanner` asks the model for the patch and, if the checks reject it, gives the
model the reason and asks once more. The evaluation turns it on with `--replan`.

**What has been measured.** In the harness, a scenario where one subgoal fails on every
attempt ends with the run failed 8 times in 8 without re-planning. With one re-plan
allowed and a stand-in re-planner that asks for the same work from another source, all 8
recover with the right result, and the other 72 scenarios are unchanged with no false
alarms. This shows the runtime applies a working patch correctly.

**What has not.** Whether a model writes a patch that works. `ModelReplanner` is tested
against scripted replies only. It has not been run on the real questions.

## Drift check

A specialist can finish its subgoal with an output about something else. Nothing in the
scheduler notices: the output is text, and the subgoals that depend on it build on it.

With `review=True` a finished output is held in a `reviewing` state and its dependents stay
blocked. The supervisor puts five yes/no questions about it to a model:

| Question | Used for |
|----------|----------|
| Does the output do the work this subgoal describes? | the verdict |
| Is it about the subject of the original question? | the verdict |
| Does it keep to this subgoal rather than another part of the task? | recorded only |
| Are its main claims tied to a source or to its inputs? | recorded only |
| Is the subgoal specific enough to judge an output against? | choosing between "drifted" and "unclear" |

The model answers the questions; the code decides. An output that fails a verdict question
is sent back and the subgoal runs again, with the reviewer's reason added to the prompt.
What happens next is bounded on purpose, because a reviewer can be wrong:

- A subgoal's output is sent back at most once. A second failure is kept and recorded as
  `supervisor.flagged`, so a wrong reviewer costs one retry per subgoal and no more.
- If the subgoal itself is too vague to judge, the output is kept and the subgoal is
  recorded as `supervisor.needs_clarification`. Retrying a vague request gets another guess.
- If the reviewer fails, the output is kept and `supervisor.review_skipped` is recorded.
- Reviews run side by side, so a slow one does not delay dead-worker detection.

```python
from concord.drift import ModelDriftChecker

await board.submit(specs, supervised=True, review=True)
supervisor = Supervisor(board, checker=ModelDriftChecker(client, question))
```

**What has been measured.** In the harness, with a stand-in reviewer that is never wrong,
all 8 off-task outputs are sent back and all 8 runs end with the right result (0 of 8
without the check), with no false alarms on the other 64 scenarios.

**What has not.** How often the model reviewer is right. `evals/drift/cases.json` holds 100
cases for that, built from the recorded runs: 60 real outputs paired with their own subgoal
(labelled on-task), 20 paired with the output of a sibling subgoal from the same question
and 20 with an output from a different question (both labelled drifted). The labels follow
from how each case was built, not from a person reading every output, so the report lists
every disagreement for a person to check. The drifted cases are swaps of real outputs,
which is one kind of drift; a specialist that wanders off partway through is not covered.

```sh
uv run python -m concord.evals.drift run          # prints the estimated cost and stops
uv run python -m concord.evals.drift run --yes    # about $0.50 on Claude Sonnet 5.5
```

It prints precision and recall for two rules: the one in use (the two verdict questions)
and a strict one that also counts the two recorded-only questions.

## Results on real questions

`evals/questions.json` holds 20 research questions. Each was answered by the agent team and
by a single agent with the same tools (web search and the same local documents), on Claude
Sonnet 5.5 at medium effort. 19 of the 20 ran; the last one failed for both systems because
the API account ran out of credit, not because of anything in Concord.

| Across the 19 questions | Single agent | Team |
|-------------------------|--------------|------|
| Finished | 19 | 19 |
| Token cost, total | $2.13 | $6.16 |
| Median time per question | 26s | 48s |
| Web searches, total | 69 | 189 |
| Judged the better answer | 6 | 12 (1 tie) |

How to read this:

- **The team costs about 2.9 times as much and takes about 1.8 times as long.** It plans 3
  to 5 subgoals per question and each research subgoal runs its own searches.
- **The team's answer was preferred 12 times to 6.** That comparison is made blind by the
  same model, not by a person, and the team's answers are longer, which model judges are
  known to favour. Treat it as a signal that the extra searching finds more, not as proof.
- **The supervisor did nothing on any of these runs.** No worker died and the planner never
  asked for the same thing twice, so there was nothing to catch. The supervisor's value is
  what the [scorecard](#scorecard) measures under injected faults; on healthy runs it is
  only overhead.
- Costs are token spend only. Web search fees are billed separately.

The local documents the questions draw on are a frozen copy of this project's docs in
`evals/corpus/`, taken when the questions were run.

Every model reply from the run is saved in `evals/recording.json`, and the full answers in
`evals/results/sonnet-5-5.json`. Replaying the recording reproduces all 19 results exactly,
in about ten seconds and without an API key:

```sh
uv run python -m concord.evals --replay evals/recording.json --limit 19
```

To run it against the API yourself:

```sh
export ANTHROPIC_API_KEY=...
uv run python -m concord.evals --limit 1 --yes                      # try one question
uv run python -m concord.evals --yes --record evals/recording.json  # all 20
```

It refuses to start without `--yes` and stops starting new questions at `--max-cost`
(default $25 of token spend). `--model` and `--effort` change the model, and `--drift-check`
turns on the review of every output.

## Quick start

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/) and Docker.

```sh
uv sync                 # install dependencies
docker compose up -d    # start Redis, Postgres and Jaeger
uv run pytest           # run the tests
uv run ruff check .     # lint
uv run python -m concord.harness   # print the scorecard
```

## Layout

```
src/concord/   library code
examples/      runnable examples
evals/         the 20 evaluation questions and their results; the drift check's labelled cases
docs/          tracing guide
tests/         tests (mock model, no API key needed; blackboard tests need Redis running)
```
