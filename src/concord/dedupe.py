"""Deciding whether two subgoals ask for the same work."""

import hashlib
import math
import re
from collections.abc import Sequence
from typing import Protocol

_NOT_WORD = re.compile(r"[^a-z0-9]+")


def normalize(description: str) -> str:
    """Lowercase and drop punctuation and extra spaces, so trivial rewording matches."""
    return _NOT_WORD.sub(" ", description.lower()).strip()


def duplicate_key(description: str, deps: Sequence[str]) -> str:
    """Two subgoals are exact duplicates when they share this key.

    Dependencies are part of the key: the same instruction applied to different inputs
    is different work.
    """
    return normalize(description) + "|" + ",".join(sorted(deps))


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    """Word-count vectors, hashed into a fixed number of dimensions.

    Deterministic and dependency-free, so it suits tests. It only sees shared words:
    it matches reordered or lightly edited text but not true paraphrases, which need a
    real embedding model behind the same interface.
    """

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for word in normalize(text).split():
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            vector[int.from_bytes(digest) % self._dimensions] += 1.0
        return vector
