from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from fdg_book_loader import BOOK_SOURCE_FILES, BookChunk, load_book_chunks
from fdg_prompting import ASSISTANT_NAME, generate_answer, resolve_ollama_config
from fdg_retriever import LiteKeywordRetriever, RetrievalResult


STARTER_QUESTIONS = [
    "What is finite deformable geometry?",
    "Explain redshift as path response.",
    "What is P-star in Chapter 4?",
    "What are measurement registers?",
    "What does the model claim, and what does it not yet prove?",
    "Explain the model in plain language.",
    "Where are the equations defined?",
]


@st.cache_resource(show_spinner=False)
def _load_corpus() -> tuple[List[BookChunk], LiteKeywordRetriever]:
    app_dir = Path(__file__).resolve().parent
    project_root = app_dir.parent
    chunks = load_book_chunks(project_root, BOOK_SOURCE_FILES)
    retriever = LiteKeywordRetriever(chunks)
    return chunks, retriever


def _format_source_line(result: RetrievalResult) -> str:
    anchor = result.chunk.anchor_id or "top"
    return f"- `{result.chunk.source_file}#{anchor}` - {result.chunk.section_title}"


def _is_in_scope_query(query: str) -> bool:
    text = query.lower()
    in_scope_terms = {
        "finite",
        "deformable",
        "geometry",
        "model",
        "plain language",
        "claim",
        "prove",
        "cosmology",
        "redshift",
        "register",
        "chapter",
        "appendix",
        "reference",
        "guidebook",
        "p-star",
        "equation",
        "bao",
        "cmb",
        "jwst",
        "gravity",
    }
    return any(term in text for term in in_scope_terms)


def main() -> None:
    st.set_page_config(page_title="Finite Deformable Geometry AI", page_icon="", layout="wide")

    st.title("Finite Deformable Geometry AI")
    st.caption("Book-specific assistant for the finite deformable geometry web book.")

    st.info(
        "Scope: this assistant answers only from the local web-book sources "
        "(chapters, guidebook, appendix, measurement registers, and references)."
    )

    with st.expander("Knowledge-base pages", expanded=False):
        st.write("Loaded sources:")
        for file_name in BOOK_SOURCE_FILES:
            st.write(f"- `{file_name}`")

    chunks, retriever = _load_corpus()

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Starter questions**")
        starter_prompt = ""
        for question in STARTER_QUESTIONS:
            if st.button(question, use_container_width=True):
                starter_prompt = question

    with col2:
        st.metric("Indexed chunks", retriever.chunk_count)
        st.metric("Source pages", len(BOOK_SOURCE_FILES))

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask a question about the book. "
                    "I will cite the local source pages and sections when available."
                ),
            }
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.markdown("**Retrieved sources**")
                for line in message["sources"]:
                    st.markdown(line)

    user_input = st.chat_input("Ask about the finite deformable geometry web book")
    prompt = starter_prompt or user_input

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    retrieved = retriever.search(prompt, top_k=6, min_score=0.25)

    if not _is_in_scope_query(prompt) and not retrieved:
        refusal = (
            "I can only answer questions about the finite deformable geometry web book materials. "
            "Try asking about chapters, equations, registers, appendix sections, or references."
        )
        st.session_state.messages.append({"role": "assistant", "content": refusal})
        with st.chat_message("assistant"):
            st.markdown(refusal)
        return

    source_lines = [_format_source_line(item) for item in retrieved[:4]]

    config = resolve_ollama_config(st.secrets)
    answer = generate_answer(prompt, retrieved, config)

    assistant_message = {
        "role": "assistant",
        "content": answer,
        "sources": source_lines,
    }
    st.session_state.messages.append(assistant_message)

    with st.chat_message("assistant"):
        st.markdown(answer)
        if source_lines:
            st.markdown("**Retrieved sources**")
            for line in source_lines:
                st.markdown(line)
        elif chunks:
            st.markdown("No strong source match found in the book.")


if __name__ == "__main__":
    main()
