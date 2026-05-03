"""Application startup configuration validation."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

try:  # pragma: no cover - optional dependency behavior
    from cryptography.fernet import Fernet
except Exception:  # pragma: no cover
    Fernet = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppConfig:
    """Normalized configuration required by the retrieval app."""

    openai_api_key: str
    conversation_key: Optional[str]
    chatbot_password: Optional[str]
    guardrail_config_path: Optional[Path]
    corpus_config_path: Path


def _read_env(name: str) -> str:
    return os.getenv(name, "").strip()


def load_and_validate() -> AppConfig:
    """Load env/file config, warning or raising on invalid startup state."""

    openai_api_key = _read_env("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is required at startup. Set a non-empty API key before launching the app."
        )

    conversation_key = _read_env("CONVERSATION_KEY") or None
    if conversation_key:
        if Fernet is None:
            raise EnvironmentError(
                "CONVERSATION_KEY is set but cryptography is unavailable. Install cryptography to validate Fernet keys."
            )
        try:
            Fernet(conversation_key)
        except Exception as exc:
            raise EnvironmentError(
                "CONVERSATION_KEY is set but invalid. Expected a valid base64 Fernet key."
            ) from exc
    else:
        log.warning(
            "CONVERSATION_KEY not set; conversation data will be stored unencrypted."
        )

    chatbot_password = _read_env("CHATBOT_PASSWORD") or None
    if not chatbot_password:
        log.warning("CHATBOT_PASSWORD not set; UI authentication is disabled.")

    guardrail_path_raw = _read_env("GUARDRAIL_CONFIG_PATH")
    guardrail_config_path: Optional[Path] = None
    if guardrail_path_raw:
        guardrail_config_path = Path(guardrail_path_raw)
        if not guardrail_config_path.exists():
            log.warning(
                "GUARDRAIL_CONFIG_PATH set but file not found: %s",
                guardrail_config_path,
            )

    corpus_path_raw = _read_env("CORPUS_CONFIG_PATH") or "Configs/corpus.yaml"
    corpus_config_path = Path(corpus_path_raw)
    if not corpus_config_path.exists():
        raise EnvironmentError(
            f"Corpus config file not found: {corpus_config_path}. "
            "Set CORPUS_CONFIG_PATH or provide Configs/corpus.yaml."
        )
    try:
        with corpus_config_path.open("r", encoding="utf-8") as source:
            parsed = yaml.safe_load(source)
    except Exception as exc:
        raise EnvironmentError(
            f"Failed to parse corpus config YAML at {corpus_config_path}: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise EnvironmentError(
            f"Corpus config at {corpus_config_path} must parse to a YAML mapping."
        )

    return AppConfig(
        openai_api_key=openai_api_key,
        conversation_key=conversation_key,
        chatbot_password=chatbot_password,
        guardrail_config_path=guardrail_config_path,
        corpus_config_path=corpus_config_path,
    )


__all__ = ["AppConfig", "load_and_validate"]
