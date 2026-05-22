from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


@dataclass(frozen=True)
class OllamaConfig:
    api_key: str
    base_url: str
    model: str


def _value_from_secrets(secrets: Mapping[str, Any], key: str) -> str:
    if key in secrets and secrets[key]:
        return str(secrets[key]).strip()

    ollama_section = secrets.get("ollama")
    if isinstance(ollama_section, Mapping):
        section_value = ollama_section.get(key)
        if section_value:
            return str(section_value).strip()
        if key == "OLLAMA_API_KEY" and ollama_section.get("api_key"):
            return str(ollama_section.get("api_key")).strip()
        if key == "OLLAMA_BASE_URL" and ollama_section.get("base_url"):
            return str(ollama_section.get("base_url")).strip()
        if key == "OLLAMA_MODEL" and ollama_section.get("model"):
            return str(ollama_section.get("model")).strip()

    return ""


def _value_from_env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def load_ollama_config(secrets: Optional[Mapping[str, Any]] = None) -> OllamaConfig:
    secrets = secrets or {}

    api_key = _value_from_secrets(secrets, "OLLAMA_API_KEY") or _value_from_env("OLLAMA_API_KEY", "")
    base_url = _value_from_secrets(secrets, "OLLAMA_BASE_URL") or _value_from_env(
        "OLLAMA_BASE_URL", "http://localhost:11434/api"
    )
    model = _value_from_secrets(secrets, "OLLAMA_MODEL") or _value_from_env("OLLAMA_MODEL", "")

    return OllamaConfig(api_key=api_key, base_url=base_url.rstrip("/"), model=model)


def resolve_ollama_chat_url(base_url: str) -> str:
    normalized_base_url = base_url.rstrip("/")
    if normalized_base_url.endswith("/api"):
        return f"{normalized_base_url}/chat"
    return f"{normalized_base_url}/api/chat"


def call_ollama_chat(messages: list[dict[str, str]], config: OllamaConfig) -> str:
    if requests is None:
        return "The Python package 'requests' is not available, so Ollama chat cannot run."
    if not config.model:
        return (
            "Ollama chat is not configured. Add OLLAMA_MODEL in Streamlit secrets or the environment. "
            "OLLAMA_BASE_URL defaults to http://localhost:11434/api."
        )

    payload = {
        "model": config.model,
        "messages": messages,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    try:
        chat_url = resolve_ollama_chat_url(config.base_url)
        response = requests.post(
            chat_url,
            json=payload,
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"]
    except requests.RequestException as exc:
        return f"Ollama chat request failed: {exc}"
    except KeyError:
        return "Ollama returned JSON without message.content."
