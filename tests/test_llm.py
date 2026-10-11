import pytest

from concord.llm import MockModelClient


async def test_returns_scripted_reply_for_matching_prompt():
    client = MockModelClient({"capital of France": "Paris"})

    result = await client.complete("What is the capital of France?")

    assert result.text == "Paris"


async def test_first_matching_entry_wins():
    client = MockModelClient({"France": "first", "capital": "second"})

    result = await client.complete("capital of France")

    assert result.text == "first"


async def test_falls_back_to_default():
    client = MockModelClient({"France": "Paris"}, default="unknown")

    result = await client.complete("capital of Peru")

    assert result.text == "unknown"


async def test_unscripted_prompt_raises_without_default():
    client = MockModelClient({"France": "Paris"})

    with pytest.raises(LookupError):
        await client.complete("capital of Peru")


async def test_counts_tokens_and_records_calls():
    client = MockModelClient(default="one two three")

    result = await client.complete("four word prompt here", system="be brief")

    assert result.input_tokens == 6
    assert result.output_tokens == 3
    assert client.calls == ["four word prompt here"]
