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
CORE_PHRASES = {
    "finite deformable geometry",
    "finite geometry",
    "deformable geometry",
    "finite universe",
    "path response",
    "redshift as path response",
    "measurement registers",
    "p-star",
    "p★",
}


@dataclass(frozen=True)
class RetrievalResult:
    chunk: BookChunk
    score: float


@dataclass(frozen=True)
class RetrievalHit:
    source_file: str
    heading: str
    anchor_id: str
    score: float
    content: str


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
    if any(
        phrase in query_lower
        for phrase in ("finite deformable geometry", "finite geometry", "deformable geometry", "finite universe")
    ) and chunk.source_file in {"chapter_3.html", "chapter_4.html", "note_from_author.html", "chapter_1.html"}:
        file_boost += 1.2
    if "chapter 4" in query_lower and chunk.source_file == "chapter_4.html":
        file_boost += 0.6
    if "measurement register" in query_lower and chunk.source_file == "measurement_registers.html":
        file_boost += 5.0
    if "path response" in query_lower and chunk.source_file in {"chapter_1.html", "chapter_4.html"}:
        file_boost += 0.9
    if ("p-star" in query_lower or "p★" in query_lower) and chunk.source_file == "chapter_4.html":
        file_boost += 1.5
    if "equation" in query_lower and "where" in query_lower and chunk.source_file == "quantitative_appendix.html":
        file_boost += 2.0

    symbol_terms = {"p-star", "g(d)", "chi", "eta", "register", "registers", "bao", "cmb", "jwst", "p★"}
    query_token_set = set(_tokenize(query_lower))
    symbol_bonus = 0.1 if query_token_set.intersection(symbol_terms) and chunk.content_type in {
        "equation",
        "table",
        "text",
    } else 0.0

    return base * (0.6 + overlap) + phrase_bonus + section_boost + symbol_bonus + file_boost


def retrieve_chunks(query: str, chunks: List[BookChunk], top_k: int = 5) -> List[RetrievalHit]:
    retriever = LiteKeywordRetriever(chunks)
    results = retriever.search(query, top_k=max(top_k, 1), min_score=0.08)
    hits = [
        RetrievalHit(
            source_file=item.chunk.source_file,
            heading=item.chunk.section_title,
            anchor_id=item.chunk.anchor_id,
            score=item.score,
            content=item.chunk.content,
        )
        for item in results
    ]

    q = query.lower()
    has_core_phrase = any(phrase in q for phrase in CORE_PHRASES)
    if hits and has_core_phrase:
        preferred_files = {"chapter_3.html", "chapter_4.html", "note_from_author.html", "chapter_1.html"}
        preferred = [hit for hit in hits if hit.source_file in preferred_files]
        if preferred:
            ordered = preferred + [hit for hit in hits if hit not in preferred]
            return ordered[:top_k]

    if hits:
        return hits[:top_k]

    # Fallback: never return empty when chunks exist. Score by core phrase overlap.
    fallback_scores: List[tuple[float, BookChunk]] = []
    query_tokens = _tokenize(query)
    for chunk in chunks:
        text = f"{chunk.section_title} {chunk.content}".lower()
        score = 0.0
        for phrase in CORE_PHRASES:
            if phrase in q and phrase in text:
                score += 3.0
        for token in query_tokens:
            if token and token in text:
                score += 0.2
        if score > 0:
            fallback_scores.append((score, chunk))

    if not fallback_scores:
        fallback_scores = [(0.0, chunk) for chunk in chunks[: max(top_k, 1)]]

    fallback_scores.sort(key=lambda item: item[0], reverse=True)
    out: List[RetrievalHit] = []
    for score, chunk in fallback_scores[:top_k]:
        out.append(
            RetrievalHit(
                source_file=chunk.source_file,
                heading=chunk.section_title,
                anchor_id=chunk.anchor_id,
                score=score,
                content=chunk.content,
            )
        )
    return out
