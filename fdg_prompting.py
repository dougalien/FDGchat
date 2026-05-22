from __future__ import annotations

from typing import Any, List, Mapping, Optional

from fdg_ollama_client import OllamaConfig, call_ollama_chat, load_ollama_config
from fdg_retriever import RetrievalResult


ASSISTANT_NAME = "Finite Deformable Geometry AI"


def resolve_ollama_config(secrets: Optional[Mapping[str, Any]] = None) -> OllamaConfig:
    return load_ollama_config(secrets)


def build_citation(result: RetrievalResult) -> str:
    anchor = result.chunk.anchor_id or "top"
    return f"[{result.chunk.source_file}#{anchor}]"


def _format_context(results: List[RetrievalResult]) -> str:
    rows: List[str] = []
    for idx, item in enumerate(results, start=1):
        citation = build_citation(item)
        rows.append(
            "\n".join(
                [
                    f"Context {idx}: {citation}",
                    f"Section: {item.chunk.section_title}",
                    f"Page title: {item.chunk.page_title}",
                    f"Content type: {item.chunk.content_type}",
                    f"Text: {item.chunk.content}",
                ]
            )
        )
    return "\n\n".join(rows)


def generate_answer(question: str, retrieved: List[RetrievalResult], config: OllamaConfig) -> str:
    if not retrieved:
        return (
            "I cannot find that in the current book sources. "
            "Please ask about the listed pages (chapters, guidebook, appendix, registers, or references)."
        )

    system_message = {
        "role": "system",
        "content": (
            "You are the Finite Deformable Geometry AI, a book-specific assistant for A Finite Universe. "
            "Answer in plain language by default. Keep most answers to 2-4 short sentences. "
            "Use only the retrieved book context and clearly relevant cosmology context. "
            "Do not present the model as established cosmology. "
            "Do not invent equations, constants, references, or claims. "
            "If the answer is not in the book context, say so briefly. "
            "Avoid jargon unless the user asks for technical detail; if technical terms are needed, define them briefly. "
            "For complex questions, answer briefly and offer one follow-up path such as 'I can explain the equation version next.' "
            "Refuse unrelated questions briefly and redirect to the book. "
            "Always include citations in the form [file.html#anchor]."
        ),
    }

    user_message = {
        "role": "user",
        "content": (
            f"Question:\n{question}\n\n"
            "Use only the contexts below:\n\n"
            f"{_format_context(retrieved)}\n\n"
            "Answer style:\n"
            "- Use 2-4 short sentences for most answers.\n"
            "- Keep wording plain and simple.\n"
            "- Include 1-3 citations.\n"
            "- If answer support is missing, say that briefly.\n"
            "- If question is complex, give one short follow-up option."
        ),
    }

    output = call_ollama_chat([system_message, user_message], config).strip()
    if not output:
        return _retrieval_only_answer(question, retrieved)

    if output.startswith("Ollama chat request failed") or output.startswith("Ollama chat is not configured"):
        return _retrieval_only_answer(question, retrieved, prefix=output)

    if "[" not in output or ".html#" not in output:
        output = output + "\n\nSources: " + _fallback_citations(retrieved)
    return output


def _fallback_citations(retrieved: List[RetrievalResult], max_items: int = 4) -> str:
    unique: List[str] = []
    for item in retrieved:
        cite = build_citation(item)
        if cite not in unique:
            unique.append(cite)
        if len(unique) >= max_items:
            break
    return " ".join(unique)


def _retrieval_only_answer(question: str, retrieved: List[RetrievalResult], prefix: str = "") -> str:
    top = retrieved[:3]
    summary_lines: List[str] = []
    if prefix:
        summary_lines.append(prefix)
        summary_lines.append("")

    summary_lines.extend(
        [
            "Model generation is unavailable, so this is a retrieval-only draft based on book text.",
            "",
            f"Question: {question}",
            "",
            "Closest matching passages:",
        ]
    )

    for item in top:
        snippet = item.chunk.content[:420].strip()
        summary_lines.append(f"- {snippet} {build_citation(item)}")

    summary_lines.extend(
        [
            "",
            "To enable live generation, configure OLLAMA_MODEL and OLLAMA_BASE_URL (plus OLLAMA_API_KEY if required).",
        ]
    )
    return "\n".join(summary_lines)
