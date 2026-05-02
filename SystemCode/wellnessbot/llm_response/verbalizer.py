from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from SystemCode.wellnessbot.llm_response.prompt_builder import (
    build_final_response_messages,
    build_final_response_payload,
    validate_final_response_text,
)

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


MODEL = os.getenv("OPENAI_FINAL_RESPONSE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def _env_openai_client() -> Any:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    return OpenAI(api_key=api_key)


def generate_final_response_text(
    result: Dict[str, Any],
    evidence_rows: Sequence[Dict[str, Any]] | None,
    *,
    timeout_s: float = 20.0,
) -> str:
    payload = build_final_response_payload(result, evidence_rows)
    messages = build_final_response_messages(result, evidence_rows)
    client = _env_openai_client()

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0.3,
        messages=messages,
        timeout=timeout_s,
    )

    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise RuntimeError("OpenAI returned empty final-response content")

    validation_errors = validate_final_response_text(content, payload)
    if validation_errors:
        joined = "; ".join(validation_errors)
        raise RuntimeError(f"Final-response validation failed: {joined}")

    return content