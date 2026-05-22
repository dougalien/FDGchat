# Finite Deformable Geometry AI (POC)

This folder contains a separate Streamlit proof-of-concept chatbot app for the finite deformable geometry web book.

## Files
- `fdg_chatbot_app.py`: Streamlit UI and chat flow.
- `fdg_book_loader.py`: HTML loader and heading-based chunking.
- `fdg_retriever.py`: lightweight keyword/TF-IDF-style chunk retrieval.
- `fdg_prompting.py`: book-scoped prompting and response formatting.
- `fdg_ollama_client.py`: Ollama-only chat client and config loading.
- `requirements.txt`: Python dependencies.

## Scope and Safety Behavior
The assistant is named **Finite Deformable Geometry AI** and is restricted to the local book corpus:
- `index.html`
- `guidebook.html`
- `note_from_author.html`
- `measurement_registers.html`
- `chapter_1.html`
- `chapter_2.html`
- `chapter_3.html`
- `chapter_4.html`
- `quantitative_appendix.html`
- `references.html`

Behavior goals:
- answer only from the book corpus,
- cite local source file + section anchor when possible,
- clearly say when a claim is not found in the book,
- avoid presenting the model as established cosmology,
- avoid invented equations, citations, or claims.

## Setup
From `fdg_ai_chatbot/`:

```powershell
py -m pip install -r requirements.txt
```

## Configuration (Ollama only)
No API key is hard-coded.

The app reads settings from Streamlit secrets first, then environment variables.

Supported names:
- `OLLAMA_API_KEY`
- `OLLAMA_BASE_URL`
- `OLLAMA_MODEL`

Supported secrets layouts:
- Top-level: `st.secrets["OLLAMA_API_KEY"]`
- Sectioned: `st.secrets["ollama"]["OLLAMA_API_KEY"]` or `st.secrets["ollama"]["api_key"]`

Environment example:

```powershell
$env:OLLAMA_API_KEY = "<optional_key_here>"
$env:OLLAMA_BASE_URL = "https://your-ollama-endpoint/api"
$env:OLLAMA_MODEL = "llama3.1:8b"
```

## Run
From `fdg_ai_chatbot/`:

```powershell
streamlit run fdg_chatbot_app.py
```

If Ollama is not configured or unavailable, the app still runs in retrieval-only mode and returns matching passages with citations.

## Notes
- This is a separate prototype and does not modify existing web-book pages.
- Retrieval is intentionally simple (keyword/semantic-lite) for phase-1 prototyping.
