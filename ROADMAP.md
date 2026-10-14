# Roadmap

Build in this order. Each phase ends with a test that must pass before the next one starts.

| # | Phase | Exit test |
|---|-------|-----------|
| 0 | Scope and setup | One command starts the services and the tests pass in CI |
| 1 | Blackboard and scheduler | A hand-written 8-subgoal graph completes on the mock model, in dependency order, with independent subgoals in parallel |
| 2 | Supervisor | Killing a worker mid-subgoal still ends in a completed run; a duplicated subgoal is served from the existing output |
| 3 | Fault-injection harness | One command prints the scorecard for at least 50 scenarios, and runs in CI |
| 4 | Tracing | A run with an injected dead worker shows the failed attempt, the requeue and the retry in one trace |
| 5 | Real planner and specialists | 20 real questions finish end to end, with cost and quality recorded beside a single-agent baseline |
| 6 | Task-drift check | Precision and recall measured against about 100 labelled outputs |
| 7 | Durability and recovery | Kill-and-resume passes ten times in a row with no finished subgoal re-run |
| 8 | Re-planning (stretch) | A run with one permanently failing subgoal recovers through a re-plan |
| 9 | Ship | A stranger can clone the repository, run one command and reproduce the scorecard |

## Design decisions

- **Redis Streams, not pub/sub**, for supervisor events. Pub/sub drops messages when the
  consumer is slow or restarting.
- **Leases and heartbeats for stall detection.** The planner's graph is acyclic by
  construction, so cycle checks only run on edges added during re-planning.
- **The supervisor checks each subgoal before dispatch.** In a supervised run a subgoal
  moves `pending → checking → ready → leased → done`, so a duplicate is caught before any
  worker spends tokens on it.
- **Duplicate detection by normalized key plus embedding similarity.** Repeated subgoals
  usually arrive reworded. Inputs are part of the comparison: the same instruction applied
  to different inputs is different work.
- **Task drift is checked with a yes/no rubric** tuned on a labelled set, not a raw score.
  The model answers the questions and the code turns the answers into a verdict.
- **An output is reviewed before anything depends on it.** A reviewed run adds a
  `reviewing` state between `leased` and `done`. Flagging after the fact would let the
  next subgoal build on a bad output.
- **A wrong reviewer is bounded.** One output per subgoal may be sent back; after that it
  is kept and flagged. A reviewer that fails is skipped, on record.
- **A run resumes from Redis alone.** A new process attaches to the run id; expired leases
  are requeued and unacknowledged supervisor events are claimed by the new supervisor, so
  no recorded output is redone. Redis is the one component that must survive.
- **Specialists are a plain model-and-tool loop.** A graph framework is added only where a
  specialist needs branching.
- **Scope: 5 to 20 subgoals, parallel readers, one writer.**

## Real questions

19 of 20 questions answered on Claude Sonnet 5.5 (the 20th failed when the API account ran
out of credit). The team cost $6.16 in tokens against $2.13 for a single agent, took a
median 48s against 26s, and was judged the better answer 12 times to 6 with 1 tie by a
same-model blind comparison. The supervisor made no interventions on these healthy runs.

## Scorecard

Measured by `uv run python -m concord.harness` on a mock model (72 scenarios). Targets are
proposals.

| Measure | Target | Result |
|---------|--------|--------|
| Exact duplicate subgoals caught | 100% | 8/8 (100%) |
| Reworded duplicate subgoals caught | ≥ 90% | 0/16 (0%), similarity matching is off by default |
| False alarms on healthy runs | ≤ 5% | 0/16 (0%) |
| Time to detect a dead worker | ≤ 2 × lease length | about 0.4s at worst against a 0.80s limit |
| Drift check precision and recall | ≥ 0.8 each | built; reviewer not yet measured on the 100 labelled cases. Given a reviewer that is never wrong, 8/8 off-task outputs are sent back |
| Tokens saved against a run with no supervisor | report the figure | 11.2% on runs with one exact duplicate |
| Finished subgoals re-run after a crash | 0 | 0 in 10 kill-and-resume trials (process killed with SIGKILL; Redis kept up) |
