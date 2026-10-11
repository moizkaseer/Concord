"""A small local document set with keyword search, for the retrieval specialist."""

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from concord.dedupe import normalize

# Standard BM25 constants.
_K1 = 1.5
_B = 0.75


@dataclass(frozen=True)
class Passage:
    source: str  # file name, and the section heading when there is one
    text: str


def _split(name: str, text: str) -> list[Passage]:
    """One passage per Markdown section, so a hit is small enough to quote."""
    passages: list[Passage] = []
    heading, lines = "", []

    def close() -> None:
        body = "\n".join(lines).strip()
        if body:
            passages.append(Passage(f"{name} § {heading}" if heading else name, body))

    for line in text.splitlines():
        if line.startswith("#"):
            close()
            heading, lines = line.lstrip("#").strip(), []
        else:
            lines.append(line)
    close()
    return passages


class Corpus:
    def __init__(self, passages: list[Passage]) -> None:
        self._passages = passages
        self._terms = [Counter(normalize(p.text + " " + p.source).split()) for p in passages]
        self._lengths = [sum(terms.values()) for terms in self._terms]
        self._mean_length = sum(self._lengths) / len(self._lengths) if passages else 0.0
        self._doc_frequency = Counter(term for terms in self._terms for term in terms)

    @classmethod
    def from_files(cls, paths: list[str | Path]) -> "Corpus":
        passages: list[Passage] = []
        for path in map(Path, paths):
            passages.extend(_split(path.name, path.read_text()))
        return cls(passages)

    def __len__(self) -> int:
        return len(self._passages)

    def search(self, query: str, k: int = 4) -> list[Passage]:
        """The `k` passages that best match the query's words (BM25), best first."""
        words = set(normalize(query).split())
        scored = []
        for index, terms in enumerate(self._terms):
            score = 0.0
            for word in words & terms.keys():
                frequency = terms[word]
                seen_in = self._doc_frequency[word]
                rarity = math.log(1 + (len(self._passages) - seen_in + 0.5) / (seen_in + 0.5))
                length = 1 - _B + _B * self._lengths[index] / self._mean_length
                score += rarity * frequency * (_K1 + 1) / (frequency + _K1 * length)
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [self._passages[index] for _, index in scored[:k]]
