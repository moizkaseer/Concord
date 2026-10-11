import pytest

from concord.corpus import Corpus, Passage
from concord.llm import Completion
from concord.models import Subgoal
from concord.specialists import (
    RESEARCH_SYSTEM,
    RETRIEVAL_SYSTEM,
    SYNTHESIS_SYSTEM,
    specialist_handler,
)

QUESTION = "How do leases work?"
CORPUS = Corpus(
    [
        Passage("design.md § Leases", "A lease expires when heartbeats stop."),
        Passage("design.md § Tracing", "Spans go to Jaeger."),
    ]
)


class Recorder:
    def __init__(self, reply: Completion) -> None:
        self._reply = reply
        self.requests: list[dict] = []

    async def complete(self, prompt, *, system=None, schema=None, web_search=False):
        self.requests.append({"prompt": prompt, "system": system, "web_search": web_search})
        return self._reply


def subgoal(specialist: str) -> Subgoal:
    return Subgoal(id="part", description="Explain lease expiry", specialist=specialist)


async def test_research_searches_the_web_and_lists_its_sources():
    client = Recorder(Completion("Leases expire.", 100, 20, sources=("https://a.example",)))

    result = await specialist_handler(client, QUESTION)(subgoal("research"), {})

    (request,) = client.requests
    assert request["web_search"] is True
    assert request["system"] == RESEARCH_SYSTEM
    assert "Original question: How do leases work?" in request["prompt"]
    assert "Your part: Explain lease expiry" in request["prompt"]
    assert result.output == "Leases expire.\n\nSources consulted:\n- https://a.example"
    assert result.tokens == 120


async def test_retrieval_answers_from_matching_passages_only():
    client = Recorder(Completion("They expire. [design.md § Leases]", 50, 10))

    result = await specialist_handler(client, QUESTION, CORPUS)(subgoal("retrieval"), {})

    (request,) = client.requests
    assert request["web_search"] is False
    assert request["system"] == RETRIEVAL_SYSTEM
    assert "[design.md § Leases]\nA lease expires when heartbeats stop." in request["prompt"]
    assert "Jaeger" not in request["prompt"]
    assert result.output == "They expire. [design.md § Leases]"


async def test_retrieval_without_a_corpus_fails_the_attempt():
    client = Recorder(Completion("x", 1, 1))

    with pytest.raises(ValueError, match="no corpus"):
        await specialist_handler(client, QUESTION)(subgoal("retrieval"), {})


async def test_synthesis_writes_from_its_inputs_without_searching():
    client = Recorder(Completion("Final answer.", 200, 80))
    inputs = {"web": "Fact one. https://a.example", "docs": "Fact two. [design.md]"}

    result = await specialist_handler(client, QUESTION, CORPUS)(subgoal("synthesis"), inputs)

    (request,) = client.requests
    assert request["web_search"] is False
    assert request["system"] == SYNTHESIS_SYSTEM
    assert "Input from web:\nFact one. https://a.example" in request["prompt"]
    assert "Input from docs:\nFact two. [design.md]" in request["prompt"]
    assert result.output == "Final answer."


async def test_unknown_specialist_fails_the_attempt():
    client = Recorder(Completion("x", 1, 1))

    with pytest.raises(ValueError, match="no specialist named 'general'"):
        await specialist_handler(client, QUESTION)(subgoal("general"), {})
