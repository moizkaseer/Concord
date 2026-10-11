"""Scripted runs, each with at most one injected fault.

Every scenario is generated from a seed, so the same command always produces the same
set. A scenario is a small research-style graph: a search and a summary per topic, a
comparison of the summaries, and a final answer.
"""

import random
from dataclasses import dataclass
from enum import StrEnum

from concord.models import SubgoalSpec

TOPICS = [
    "solar power",
    "wind power",
    "nuclear power",
    "hydrogen fuel",
    "battery storage",
    "geothermal energy",
    "tidal energy",
    "carbon capture",
]
ASPECTS = ["cost", "history", "risks", "adoption", "regulation", "efficiency"]


class Kind(StrEnum):
    HEALTHY = "healthy"
    # Healthy, but with subgoals that differ by a single word. Reusing between them
    # would be wrong, so this is the control that exposes over-eager matching.
    LOOKALIKE = "healthy, look-alike subgoals"
    EXACT_DUPLICATE = "exact duplicate"
    LIGHT_REWORD = "reworded duplicate, light edit"
    PARAPHRASE = "reworded duplicate, paraphrase"
    DEAD_WORKER = "dead worker"
    HUNG_WORKER = "hung worker"
    SUPERVISOR_RESTART = "supervisor restart"
    OFF_TASK = "off-task output"


DUPLICATE_KINDS = (Kind.EXACT_DUPLICATE, Kind.LIGHT_REWORD, Kind.PARAPHRASE)
HEALTHY_KINDS = (Kind.HEALTHY, Kind.LOOKALIKE)


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: Kind
    specs: list[SubgoalSpec]
    expected: dict[str, str]  # the correct output of every subgoal
    target: str | None = None  # the subgoal the fault hits
    partner: str | None = None  # for duplicates, the subgoal it repeats


# Planners phrase the same kind of request at different lengths. Length matters to
# similarity matching: one changed word is a smaller share of a longer sentence.
_PHRASING = {
    "search": [
        "Find sources on the {aspect} of {topic}",
        "Find recent and reliable sources on the {aspect} of {topic} for the report",
    ],
    "summary": [
        "Summarize the sources on the {aspect} of {topic}",
        "Summarize the main findings from the sources on the {aspect} of {topic}",
    ],
}


def _describe(step: str, aspect: str, topic: str, style: int) -> str:
    return _PHRASING[step][style].format(aspect=aspect, topic=topic)


def _exact_variant(rng: random.Random, text: str) -> str:
    """The same words with different case, punctuation or spacing."""
    return rng.choice(
        [text.upper(), text + ".", text.replace(" ", "  "), text.lower() + "!", f"  {text}  "]
    )


def _light_edit(rng: random.Random, text: str) -> str:
    """One word changed or added; the rest identical."""
    return rng.choice(
        [
            text.replace(" on the ", " about the ", 1),
            f"Please {text[0].lower()}{text[1:]}",
            text.replace("sources", "relevant sources", 1),
        ]
    )


def _paraphrase(step: str, aspect: str, topic: str) -> str:
    """The same request in different words."""
    if step == "search":
        return f"Locate references covering {topic} {aspect}"
    return f"Condense what was gathered regarding {topic} {aspect}"


def build(kind: Kind, seed: int) -> Scenario:
    rng = random.Random(f"{kind}/{seed}")
    style = rng.choice([0, 1])
    count = rng.choice([2, 3])
    topics = rng.sample(TOPICS, count)
    if kind is Kind.LOOKALIKE:
        aspects = [rng.choice(ASPECTS)] * count
    else:
        aspects = rng.sample(ASPECTS, count)

    graph: dict[str, tuple[str, list[str]]] = {}
    expected: dict[str, str] = {}
    parts: dict[str, tuple[str, str, str]] = {}
    for n, (aspect, topic) in enumerate(zip(aspects, topics, strict=True)):
        for step, subgoal_id, deps in (("search", f"s{n}", []), ("summary", f"m{n}", [f"s{n}"])):
            graph[subgoal_id] = (_describe(step, aspect, topic, style), deps)
            expected[subgoal_id] = f"{step} result for the {aspect} of {topic}"
            parts[subgoal_id] = (step, aspect, topic)
    summaries = [f"m{n}" for n in range(count)]
    graph["compare"] = ("Compare the summaries and note where they disagree", summaries)
    expected["compare"] = f"comparison of {count} summaries"
    graph["answer"] = ("Write the final answer to the question", ["compare"])
    expected["answer"] = "final answer"

    target = partner = None
    if kind not in HEALTHY_KINDS and kind is not Kind.SUPERVISOR_RESTART:
        target = rng.choice(sorted(parts))
    if kind in DUPLICATE_KINDS:
        # The planner asked for the same work twice, and the answer waits on both.
        partner, target = target, f"{target}-dup"
        description, deps = graph[partner]
        if kind is Kind.EXACT_DUPLICATE:
            description = _exact_variant(rng, description)
        elif kind is Kind.LIGHT_REWORD:
            description = _light_edit(rng, description)
        else:
            description = _paraphrase(*parts[partner])
        graph[target] = (description, list(deps))
        expected[target] = expected[partner]
        graph["answer"][1].append(target)

    specs = [
        SubgoalSpec(id=subgoal_id, description=description, depends_on=deps)
        for subgoal_id, (description, deps) in graph.items()
    ]
    return Scenario(f"{kind} #{seed}", kind, specs, expected, target, partner)


def build_all(seeds: int) -> list[Scenario]:
    return [build(kind, seed) for kind in Kind for seed in range(seeds)]
