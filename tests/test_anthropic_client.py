from types import SimpleNamespace

import pytest

from concord.anthropic_client import AnthropicClient, ModelRefusal, ModelTruncated


def text(value: str):
    return SimpleNamespace(type="text", text=value)


def response(*content, stop_reason="end_turn", input_tokens=10, output_tokens=5, **extra):
    usage = SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=extra.pop("cache_write", 0),
        cache_read_input_tokens=extra.pop("cache_read", 0),
    )
    return SimpleNamespace(content=list(content), stop_reason=stop_reason, usage=usage, **extra)


class FakeSDK:
    """Stands in for AsyncAnthropic: records requests and plays back responses."""

    def __init__(self, *responses) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    async def _create(self, **request):
        self.requests.append(request)
        return self._responses.pop(0)


async def test_plain_request_shape_and_token_accounting():
    sdk = FakeSDK(response(text("Hello "), text("there"), cache_read=100, cache_write=20))
    client = AnthropicClient(client=sdk, effort="low")

    completion = await client.complete("Say hello", system="Be brief")

    (request,) = sdk.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["messages"] == [{"role": "user", "content": "Say hello"}]
    assert request["system"] == "Be brief"
    assert request["output_config"] == {"effort": "low"}
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tools" not in request and "thinking" not in request
    assert completion.text == "Hello there"
    assert (completion.input_tokens, completion.output_tokens) == (130, 5)


async def test_schema_is_sent_as_a_structured_output_format():
    sdk = FakeSDK(response(text('{"ok": true}')))
    schema = {"type": "object", "properties": {}, "additionalProperties": False}

    await AnthropicClient(client=sdk).complete("Reply in JSON", schema=schema)

    assert sdk.requests[0]["output_config"]["format"] == {"type": "json_schema", "schema": schema}


async def test_fallbacks_can_be_turned_off():
    sdk = FakeSDK(response(text("ok")))

    await AnthropicClient(client=sdk, fallbacks=False).complete("hi")

    assert "fallbacks" not in sdk.requests[0] and "betas" not in sdk.requests[0]


async def test_web_search_collects_sources_and_counts_searches():
    results = [SimpleNamespace(url="https://a.example"), SimpleNamespace(url="https://b.example")]
    sdk = FakeSDK(
        response(
            SimpleNamespace(type="server_tool_use", name="web_search"),
            SimpleNamespace(type="web_search_tool_result", content=results),
            SimpleNamespace(type="server_tool_use", name="web_search"),
            # A failed search carries an error object instead of a list of results.
            SimpleNamespace(
                type="web_search_tool_result",
                content=SimpleNamespace(error_code="max_uses_exceeded"),
            ),
            SimpleNamespace(type="web_search_tool_result", content=results[:1]),
            text("Found it."),
        )
    )

    completion = await AnthropicClient(client=sdk, max_searches=3).complete(
        "Look it up", web_search=True
    )

    assert sdk.requests[0]["tools"] == [
        {"type": "web_search_20260209", "name": "web_search", "max_uses": 3}
    ]
    assert completion.sources == ("https://a.example", "https://b.example")
    assert completion.searches == 2
    assert completion.text == "Found it."


async def test_paused_turn_is_resumed_and_usage_added_up():
    paused = [SimpleNamespace(type="server_tool_use", name="web_search"), text("Part one. ")]
    sdk = FakeSDK(
        response(*paused, stop_reason="pause_turn", input_tokens=10, output_tokens=5),
        response(text("Part two."), input_tokens=30, output_tokens=7),
    )

    completion = await AnthropicClient(client=sdk).complete("Research this", web_search=True)

    assert sdk.requests[1]["messages"] == [
        {"role": "user", "content": "Research this"},
        {"role": "assistant", "content": paused},
    ]
    assert completion.text == "Part one. Part two."
    assert (completion.input_tokens, completion.output_tokens) == (40, 12)
    assert completion.searches == 1


async def test_turn_that_never_stops_pausing_is_an_error():
    sdk = FakeSDK(*[response(text("x"), stop_reason="pause_turn") for _ in range(5)])

    with pytest.raises(ModelTruncated, match="still paused"):
        await AnthropicClient(client=sdk).complete("Research this", web_search=True)


async def test_refusal_and_truncation_raise():
    refused = response(
        stop_reason="refusal", stop_details=SimpleNamespace(category="cyber", explanation="")
    )
    with pytest.raises(ModelRefusal, match="cyber"):
        await AnthropicClient(client=FakeSDK(refused)).complete("hi")

    cut_off = response(text("partial"), stop_reason="max_tokens")
    with pytest.raises(ModelTruncated, match="cut off"):
        await AnthropicClient(client=FakeSDK(cut_off)).complete("hi")
