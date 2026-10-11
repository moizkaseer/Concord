"""Record a model client's replies to a file, and replay them later without the model.

Recording real replies once lets tests and demos run the whole pipeline for free and get
the same result every time. A reply is found again by a hash of everything that went
into the request, so a replayed run must send exactly the prompts the recorded run sent.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from concord.llm import Completion, ModelClient


class MissingRecording(LookupError):
    """A replayed run sent a request that was never recorded."""


def _key(prompt: str, system: str | None, schema: dict[str, Any] | None, web_search: bool) -> str:
    request = json.dumps([prompt, system, schema, web_search], sort_keys=True)
    return hashlib.sha256(request.encode()).hexdigest()


class RecordingClient:
    """Passes every call to `inner` and saves the reply to `path`."""

    def __init__(self, inner: ModelClient, path: str | Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._entries: dict[str, dict[str, Any]] = (
            json.loads(self._path.read_text()) if self._path.exists() else {}
        )

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> Completion:
        completion = await self._inner.complete(
            prompt, system=system, schema=schema, web_search=web_search
        )
        entry = dataclasses.asdict(completion)
        entry["prompt_preview"] = prompt[:120]
        self._entries[_key(prompt, system, schema, web_search)] = entry
        # Written after every call so an interrupted run keeps what it paid for.
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries, indent=2, sort_keys=True))
        return completion


class ReplayClient:
    """Answers from a file written by `RecordingClient`; never calls a model."""

    def __init__(self, path: str | Path) -> None:
        self._entries: dict[str, dict[str, Any]] = json.loads(Path(path).read_text())

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        schema: dict[str, Any] | None = None,
        web_search: bool = False,
    ) -> Completion:
        entry = self._entries.get(_key(prompt, system, schema, web_search))
        if entry is None:
            raise MissingRecording(f"no recorded reply for prompt: {prompt[:80]!r}")
        return Completion(
            text=entry["text"],
            input_tokens=entry["input_tokens"],
            output_tokens=entry["output_tokens"],
            sources=tuple(entry["sources"]),
            searches=entry["searches"],
        )
