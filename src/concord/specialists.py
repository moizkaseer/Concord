"""The three specialists a plan can assign a subgoal to.

Each is a plain function from a subgoal and its inputs to a result: gather what the
specialist is allowed to see, make one model call, return the text. Only the synthesis
specialist writes the final answer, so there is a single writer per run.
"""

from concord.corpus import Corpus
from concord.llm import Completion, ModelClient
from concord.models import Subgoal
from concord.tracing import tool_span
from concord.worker import Handler, WorkResult

RESEARCH_SYSTEM = """\
You are the research specialist on a team answering a question together. You have been \
given one part of the work. Other specialists are covering the other parts, and a writer \
will combine everything afterwards, so report only on your part and do not try to answer \
the whole question.

Search the web for what your part asks. Then report what you found as plain statements \
of fact, each followed by the URL it came from. Prefer primary sources such as official \
documentation. If sources disagree, say so and give both. If you could not find \
something, say that plainly instead of filling the gap from memory: the writer needs to \
know what is established and what is not."""

RETRIEVAL_SYSTEM = """\
You are the retrieval specialist on a team answering a question together. You have been \
given one part of the work and a set of passages from the project's own documents.

Report what the passages say about your part, as plain statements of fact, each followed \
by the name of the passage it came from in square brackets. Use only the passages. If \
they do not cover something your part asks for, say so plainly: a writer will combine \
your report with others and needs to know what the documents do and do not establish."""

SYNTHESIS_SYSTEM = """\
You are the writer on a team answering a question together. The other specialists have \
reported their findings, which you are given below as inputs. Write the final answer to \
the question for the person who asked it.

Base the answer on the inputs. Keep the source next to each claim it supports, as the \
inputs give it (a URL or a passage name). Where the inputs disagree or say something \
could not be found, say so in the answer rather than smoothing it over. Lead with the \
direct answer to the question, then the supporting detail. Do not describe the team or \
how the answer was produced."""


def _prompt(question: str, subgoal: Subgoal, inputs: dict[str, str], extra: str = "") -> str:
    parts = [f"Original question: {question}", f"Your part: {subgoal.description}"]
    parts += [f"Input from {dep}:\n{text}" for dep, text in inputs.items()]
    if extra:
        parts.append(extra)
    if subgoal.feedback:
        parts.append(
            "A reviewer sent back an earlier attempt at this part because it did not do "
            f"what the part asks: {subgoal.feedback}\nMake sure this attempt does."
        )
    return "\n\n".join(parts)


def _result(completion: Completion, text: str | None = None) -> WorkResult:
    return WorkResult(
        text if text is not None else completion.text,
        completion.input_tokens + completion.output_tokens,
    )


def specialist_handler(
    client: ModelClient, question: str, corpus: Corpus | None = None, *, passages: int = 4
) -> Handler:
    """A worker handler that sends each subgoal to the specialist its plan names."""

    async def research(subgoal: Subgoal, inputs: dict[str, str]) -> WorkResult:
        completion = await client.complete(
            _prompt(question, subgoal, inputs), system=RESEARCH_SYSTEM, web_search=True
        )
        if not completion.sources:
            return _result(completion)
        # Listed again in full so the writer can cite them even if the report shortened one.
        listed = "\n".join(f"- {url}" for url in completion.sources[:10])
        return _result(completion, f"{completion.text}\n\nSources consulted:\n{listed}")

    async def retrieval(subgoal: Subgoal, inputs: dict[str, str]) -> WorkResult:
        if corpus is None:
            raise ValueError("plan uses the retrieval specialist but no corpus was given")
        with tool_span("local_search"):
            found = corpus.search(f"{subgoal.description} {question}", k=passages)
        shown = "\n\n".join(f"[{p.source}]\n{p.text}" for p in found) or "(no passage matched)"
        completion = await client.complete(
            _prompt(question, subgoal, inputs, f"Passages:\n\n{shown}"), system=RETRIEVAL_SYSTEM
        )
        return _result(completion)

    async def synthesis(subgoal: Subgoal, inputs: dict[str, str]) -> WorkResult:
        completion = await client.complete(
            _prompt(question, subgoal, inputs), system=SYNTHESIS_SYSTEM
        )
        return _result(completion)

    handlers = {"research": research, "retrieval": retrieval, "synthesis": synthesis}

    async def handle(subgoal: Subgoal, inputs: dict[str, str]) -> WorkResult:
        if subgoal.specialist not in handlers:
            raise ValueError(f"no specialist named {subgoal.specialist!r}")
        return await handlers[subgoal.specialist](subgoal, inputs)

    return handle
