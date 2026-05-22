from __future__ import annotations

from typing import Dict, List

import streamlit as st

from fdg_book_loader import BOOK_SOURCE_FILES, BookChunk, get_last_load_status, load_book_chunks
from fdg_prompting import generate_answer, resolve_ollama_config
from fdg_retriever import LiteKeywordRetriever, RetrievalResult

STARTER_QUESTIONS = [
    "What is finite deformable geometry?",
    "Explain redshift as path response.",
    "What is P-star in Chapter 4?",
    "What are measurement registers?",
    "What does the model claim, and what does it not yet prove?",
    "Where are the equations defined?",
]


def apply_book_theme() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&display=swap');
:root {
  --ink: #182230;
  --paper: #fbf8f1;
  --panel: #ffffff;
  --blue: #09233f;
  --blue2: #123a63;
  --gold: #c89b3c;
  --line: #d9d0bd;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background: var(--paper);
  color: var(--ink);
  font-family: "Source Serif 4", Georgia, serif;
}
[data-testid="stHeader"] {
  background: linear-gradient(135deg, rgba(9,35,63,0.96), rgba(18,58,99,0.96));
  border-bottom: 3px solid var(--gold);
}
.fdg-title-band {
  background: linear-gradient(135deg, #06182c, #0d2e4f 60%, #06182c);
  color: #f8ecd0;
  border: 1px solid var(--line);
  border-left: 6px solid var(--gold);
  border-radius: 10px;
  padding: 0.9rem 1rem;
  margin: 0.25rem 0 0.8rem 0;
}
.fdg-title-main {
  margin: 0;
  font-size: clamp(1.4rem, 1.0rem + 1vw, 2rem);
  font-weight: 700;
  letter-spacing: 0.02em;
}
.fdg-title-sub {
  margin: 0.2rem 0 0;
  color: #dfe8f2;
  font-family: Inter, system-ui, sans-serif;
  font-size: 0.95rem;
}
.fdg-answer-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-top: 4px solid var(--gold);
  border-radius: 8px;
  padding: 1rem;
  margin-top: 0.8rem;
}
.fdg-answer-card p, .fdg-answer-card li {
  color: var(--ink);
}
[data-testid="stSidebar"] {
  background: #f6f1e6;
  border-right: 1px solid var(--line);
}
.stButton > button {
  background: #f6e2ae;
  border: 1px solid var(--gold);
  color: var(--blue);
  font-family: Inter, system-ui, sans-serif;
  font-weight: 700;
}
label, [data-testid="stMarkdownContainer"] p {
  font-family: "Source Serif 4", Georgia, serif;
}
</style>
""",
        unsafe_allow_html=True,
    )


def _load_corpus() -> tuple[List[BookChunk], LiteKeywordRetriever, Dict[str, object], str]:
    error_message = ""
    try:
        chunks = load_book_chunks()
    except Exception as exc:  # pragma: no cover
        chunks = []
        error_message = str(exc)

    status = get_last_load_status()
    retriever = LiteKeywordRetriever(chunks)
    return chunks, retriever, status, error_message


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
        "path response",
    }
    return any(term in text for term in in_scope_terms)


def _show_source_status(status: Dict[str, object], error_message: str) -> None:
    with st.sidebar.expander("Book source status", expanded=False):
        st.write(f"book_context path: `{status.get('book_context_dir', '')}`")
        st.write(f"source files found: `{status.get('source_files_found', 0)}`")
        st.write(f"chunks loaded: `{status.get('chunks_loaded', 0)}`")

        filenames = status.get("filenames_found", [])
        if isinstance(filenames, list) and filenames:
            st.write("filenames found:")
            for name in filenames:
                st.write(f"- `{name}`")

        cleaned = status.get("cleaned_char_count_by_file", {})
        if isinstance(cleaned, dict) and cleaned:
            st.write("cleaned character counts:")
            for name, count in cleaned.items():
                st.write(f"- `{name}`: {count}")

        by_file = status.get("chunks_created_by_file", {})
        if isinstance(by_file, dict) and by_file:
            st.write("chunks created by file:")
            for name, count in by_file.items():
                st.write(f"- `{name}`: {count}")

        if error_message:
            st.caption(error_message)


def main() -> None:
    st.set_page_config(page_title="Finite Deformable Geometry AI", page_icon="", layout="wide")
    apply_book_theme()

    chunks, retriever, load_status, load_error = _load_corpus()
    _show_source_status(load_status, load_error)

    source_files_found = int(load_status.get("source_files_found", 0) or 0)
    chunks_loaded = int(load_status.get("chunks_loaded", 0) or 0)

    if source_files_found == 0:
        st.error("Book source loading failed. The chatbot cannot answer until book_context files are available.")
        if load_error:
            st.caption(load_error)
        st.stop()

    if source_files_found > 0 and chunks_loaded == 0:
        st.error("Book files were found, but no text chunks were created. Check fdg_book_loader.py cleaning/chunking.")
        if load_error:
            st.caption(load_error)
        st.stop()

    st.markdown(
        """
<div class="fdg-title-band">
  <p class="fdg-title-main">Finite Deformable Geometry AI</p>
  <p class="fdg-title-sub">Book-specific assistant for A Finite Universe</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.caption(
        "Ask a short question about the model, chapters, guidebook, appendix, registers, or references. "
        "Answers stay plain-language by default."
    )

    if "fdg_question" not in st.session_state:
        st.session_state.fdg_question = ""
    if "fdg_answer" not in st.session_state:
        st.session_state.fdg_answer = ""
    if "fdg_sources" not in st.session_state:
        st.session_state.fdg_sources = []

    question = st.text_input("Your question", value=st.session_state.fdg_question, placeholder="What is finite deformable geometry?")
    ask_pressed = st.button("Ask", use_container_width=False)

    if ask_pressed and question.strip():
        prompt = question.strip()
        st.session_state.fdg_question = prompt

        retrieved = retriever.search(prompt, top_k=6, min_score=0.12)
        if not retrieved and chunks:
            retrieved = retriever.search(prompt, top_k=6, min_score=-1.0)

        if not _is_in_scope_query(prompt) and not retrieved:
            st.session_state.fdg_answer = (
                "I can only answer questions about this web book. "
                "Try asking about finite deformable geometry, path response, chapters, registers, or the appendix."
            )
            st.session_state.fdg_sources = []
        else:
            source_lines = [_format_source_line(item) for item in retrieved[:4]]
            config = resolve_ollama_config(st.secrets)
            st.session_state.fdg_answer = generate_answer(prompt, retrieved, config)
            st.session_state.fdg_sources = source_lines

    if st.session_state.fdg_answer:
        st.markdown(f"<div class='fdg-answer-card'>{st.session_state.fdg_answer}</div>", unsafe_allow_html=True)
        if st.session_state.fdg_sources:
            st.markdown("**Sources**")
            for line in st.session_state.fdg_sources:
                st.markdown(line)

    with st.expander("Starter questions", expanded=False):
        for starter in STARTER_QUESTIONS:
            if st.button(starter, key=f"starter_{starter}", use_container_width=True):
                st.session_state.fdg_question = starter
                retrieved = retriever.search(starter, top_k=6, min_score=0.12)
                if not retrieved and chunks:
                    retrieved = retriever.search(starter, top_k=6, min_score=-1.0)
                source_lines = [_format_source_line(item) for item in retrieved[:4]]
                config = resolve_ollama_config(st.secrets)
                st.session_state.fdg_answer = generate_answer(starter, retrieved, config)
                st.session_state.fdg_sources = source_lines
                st.rerun()


if __name__ == "__main__":
    main()
