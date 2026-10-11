"""Model client interface, plus a scripted mock so tests are deterministic and free."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int


class ModelClient(Protocol):
    async def complete(self, prompt: str, *, system: str | None = None) -> Completion: ...


class MockModelClient:
    """Returns scripted replies instead of calling a model.

    `script` maps a substring to the reply for any prompt containing it; the first
    matching entry wins. Prompts that match nothing get `default`, or raise if no
    default was given, so an unscripted call fails loudly in a test.
    """

    def __init__(self, script: dict[str, str] | None = None, default: str | None = None) -> None:
        self._script = dict(script or {})
        self._default = default
        self.calls: list[str] = []

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.calls.append(prompt)
        text = self._reply(prompt)
        # Word counts stand in for tokens so budget logic has something to add up.
        return Completion(
            text=text,
            input_tokens=len(prompt.split()) + len((system or "").split()),
            output_tokens=len(text.split()),
        )

    def _reply(self, prompt: str) -> str:
        for needle, reply in self._script.items():
            if needle in prompt:
                return reply
        if self._default is None:
            raise LookupError(f"no scripted reply for prompt: {prompt[:80]!r}")
        return self._default
