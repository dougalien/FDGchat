from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Dict, List

from fdg_book_loader import BookChunk

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-\+\.]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "was",
    "what",
    "where",
    "who",
    "won",
}


@dataclass(frozen=True)
class RetrievalResult:
    chunk: BookChunk
    score: float


class LiteKeywordRetriever:
    def __init__(self, chunks: List[BookChunk]) -> None:
        self._chunks = chunks
        self._chunk_tokens: List[Counter[str]] = []
        self._doc_freq: Counter[str] = Counter()
        self._idf: Dict[str, float] = {}
        self._build_index()

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def _build_index(self) -> None:
        for chunk in self._chunks:
            tokens = _tokenize(f"{chunk.section_title} {chunk.content}")
            freq = Counter(tokens)
            self._chunk_tokens.append(freq)
            self._doc_freq.update(set(freq))

        n_docs = max(len(self._chunks), 1)
        self._idf = {
            term: math.log((n_docs + 1) / (df + 1)) + 1.0 for term, df in self._doc_freq.items()
        }

    def search(self, query: str, top_k: int = 6, min_score: float = 0.15) -> List[RetrievalResult]:
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        query_freq = Counter(query_tokens)
        query_terms = set(query_tokens)

        scored: List[RetrievalResult] = []
        for idx, chunk in enumerate(self._chunks):
            score = _score_chunk(
                query=query,
                query_freq=query_freq,
                query_terms=query_terms,
                chunk=chunk,
                chunk_freq=self._chunk_tokens[idx],
                idf=self._idf,
            )
            if score >= min_score:
                scored.append(RetrievalResult(chunk=chunk, score=score))

        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]


def _tokenize(text: str) -> List[str]:
    tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
    return [token for token in tokens if token not in _STOPWORDS and len(token) > 1]


def _score_chunk(
    query: str,
    query_freq: Counter[str],
    query_terms: set[str],
    chunk: BookChunk,
    chunk_freq: Counter[str],
    idf: Dict[str, float],
) -> float:
    if not chunk_freq:
        return 0.0

    base = 0.0
    for term, qtf in query_freq.items():
        ctf = chunk_freq.get(term, 0)
        if ctf == 0:
            continue
        base += (1.0 + math.log1p(ctf)) * (1.0 + math.log1p(qtf)) * idf.get(term, 1.0)

    overlap = len(query_terms.intersection(set(chunk_freq))) / max(len(query_terms), 1)

    query_lower = query.lower()
    text_lower = f"{chunk.section_title} {chunk.content}".lower()
    phrase_bonus = 0.35 if len(query_lower) > 8 and query_lower in text_lower else 0.0

    section_boost = 0.2 if any(term in chunk.section_title.lower() for term in query_terms) else 0.0

    file_boost = 0.0
    if "chapter 4" in query_lower and chunk.source_file == "chapter_4.html":
        file_boost += 0.6
    if "measurement register" in query_lower and chunk.source_file == "measurement_registers.html":
        file_boost += 5.0
    if "equation" in query_lower and "where" in query_lower and chunk.source_file == "quantitative_appendix.html":
        file_boost += 2.0

    symbol_terms = {"p-star", "g(d)", "chi", "eta", "register", "registers", "bao", "cmb", "jwst"}
    query_token_set = set(_tokenize(query_lower))
    symbol_bonus = 0.1 if query_token_set.intersection(symbol_terms) and chunk.content_type in {
        "equation",
        "table",
        "text",
    } else 0.0

    return base * (0.6 + overlap) + phrase_bonus + section_boost + symbol_bonus + file_boost
