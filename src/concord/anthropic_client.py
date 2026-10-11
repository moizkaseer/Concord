"""A model client backed by the Claude API."""

from typing import Any

from anthropic import AsyncAnthropic

from concord.llm import Completion

DEFAULT_MODEL = "claude-sonnet-5-5"
PROVIDER = "anthropic"

# Lets the API re-run a request that a safety classifier declined on another model,
# instead of returning the refusal.
_FALLBACK_BETA = "server-side-fallback-2026-07-01"
_WEB_SEARCH_TOOL = "web_search_20260209"
# A turn that uses web search can pause when the server-side loop hits its iteration
# limit; resending it resumes the turn. This bounds how often we do that.
_MAX_RESUMES = 4


class ModelRefusal(RuntimeError):
    """The model declined the request."""


class ModelTruncated(RuntimeError):
    """The reply hit the output limit before it was finished."""


class AnthropicClient:
    """Calls Claude with retries, timeouts and token accounting.

    Rate limits, server errors and connection errors are retried by the SDK
    (`max_retries` times, with backoff). Credentials come from the environment
    (`ANTHROPIC_API_KEY`, or an `ant auth login` profile).
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        effort: str = "medium",
        max_tokens: int = 16000,
        timeout: float = 300.0,
        max_retries: int = 3,
        max_searches: int = 5,
        fallbacks: bool = True,
        client: AsyncAnthropic | None = None,
    ) -> None:
        self.model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._max_searches = max_searches
        self._fallbacks = fallbacks
        self._client = client or AsyncAnthropic(timeout=timeout, max_retries=max_retries)

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> Completion:
        output_config: dict[str, Any] = {"effort": self._effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self._max_tokens,
            "output_config": output_config,
        }
        if system:
            request["system"] = system
        if web_search:
            request["tools"] = [
                {"type": _WEB_SEARCH_TOOL, "name": "web_search", "max_uses": self._max_searches}
            ]
        if self._fallbacks:
            request["betas"] = [_FALLBACK_BETA]
            request["fallbacks"] = "default"

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        text: list[str] = []
        sources: list[str] = []
        input_tokens = output_tokens = searches = 0
        for _ in range(_MAX_RESUMES + 1):
            response = await self._client.beta.messages.create(messages=messages, **request)
            usage = response.usage
            # Cached input is still input the run paid for and read.
            input_tokens += (
                usage.input_tokens
                + (getattr(usage, "cache_creation_input_tokens", 0) or 0)
                + (getattr(usage, "cache_read_input_tokens", 0) or 0)
            )
            output_tokens += usage.output_tokens
            for block in response.content:
                if block.type == "text":
                    text.append(block.text)
                elif block.type == "server_tool_use" and block.name == "web_search":
                    searches += 1
                elif block.type == "web_search_tool_result" and isinstance(block.content, list):
                    # A failed search carries an error object here instead of a list.
                    sources.extend(result.url for result in block.content)
            if response.stop_reason != "pause_turn":
                break
            messages = [messages[0], {"role": "assistant", "content": response.content}]

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise ModelRefusal(f"the model declined the request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise ModelTruncated(f"reply was cut off at {self._max_tokens} output tokens")
        if response.stop_reason == "pause_turn":
            raise ModelTruncated(f"turn was still paused after {_MAX_RESUMES} resumes")

        return Completion(
            text="".join(text),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            sources=tuple(dict.fromkeys(sources)),
            searches=searches,
        )
