import pytest

from concord.llm import Completion, MockModelClient
from concord.recording import MissingRecording, RecordingClient, ReplayClient


class Searching:
    async def complete(self, prompt, *, system=None, schema=None, web_search=False):
        return Completion("found", 7, 3, sources=("https://a.example",), searches=2)


async def test_replay_returns_what_was_recorded(tmp_path):
    path = tmp_path / "nested" / "cassette.json"
    recorder = RecordingClient(Searching(), path)

    recorded = await recorder.complete("question", system="be brief", web_search=True)
    replayed = await ReplayClient(path).complete("question", system="be brief", web_search=True)

    assert replayed == recorded


async def test_replay_distinguishes_requests_by_everything_sent(tmp_path):
    path = tmp_path / "cassette.json"
    await RecordingClient(MockModelClient(default="ok"), path).complete("question", system="a")
    replay = ReplayClient(path)

    for changed in (
        {"prompt": "other question", "system": "a"},
        {"prompt": "question", "system": "b"},
        {"prompt": "question", "system": "a", "web_search": True},
        {"prompt": "question", "system": "a", "schema": {"type": "object"}},
    ):
        with pytest.raises(MissingRecording):
            await replay.complete(changed.pop("prompt"), **changed)


async def test_recording_adds_to_an_existing_file(tmp_path):
    path = tmp_path / "cassette.json"
    await RecordingClient(MockModelClient(default="first"), path).complete("one")
    await RecordingClient(MockModelClient(default="second"), path).complete("two")
    replay = ReplayClient(path)

    assert (await replay.complete("one")).text == "first"
    assert (await replay.complete("two")).text == "second"
