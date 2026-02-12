from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.nlu.mock_extractor import extract_mock

# OpenAI is optional dependency at runtime when MOCK_NLU=0.
# Install when needed: pip install openai
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


_SYSTEM_PROMPT = """You are an NLU extraction component.
You MUST output a single JSON object ONLY (no markdown, no prose).
Extract structured fields from the user's text.
Rules:
- If a field is unknown, use null (or "unknown" for enums).
- weeks_since_event: float weeks. Convert days to weeks (days/7).
- pain_score: integer 0..10 if present.
- swelling_level: one of ["none","mild","moderate","severe","unknown"].
- weight_bearing: one of ["none","partial","full","unknown"].
- event_type: one of ["acl_surgery","tkr","meniscus","sprain","unknown"].
- requested_exercise_text: short normalized phrase like "squats", "heel slides", "quad sets", "straight leg raise", etc.
- red_flag_terms: list of red flags mentioned (non-negated).
- negated_terms: list of terms that are explicitly negated (e.g., "no fever" => "fever" in negated_terms).
- missing_fields: list of missing field names from:
  ["weeks_since_event","event_type","requested_exercise_text","pain_score","swelling_level","weight_bearing"].
- nlu_source: must be "openai".
"""


def _build_response_schema() -> Dict[str, Any]:
    """
    JSON schema for structured output.
    Kept explicit to reduce model drift.
    """
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "weeks_since_event": {"type": ["number", "null"]},
            "event_type": {
                "type": "string",
                "enum": ["acl_surgery", "tkr", "meniscus", "sprain", "unknown"],
            },
            "requested_exercise_text": {"type": "string"},
            "pain_score": {"type": ["integer", "null"], "minimum": 0, "maximum": 10},
            "swelling_level": {
                "type": "string",
                "enum": ["none", "mild", "moderate", "severe", "unknown"],
            },
            "weight_bearing": {
                "type": "string",
                "enum": ["none", "partial", "full", "unknown"],
            },
            "red_flag_terms": {"type": "array", "items": {"type": "string"}},
            "negated_terms": {"type": "array", "items": {"type": "string"}},
            "missing_fields": {"type": "array", "items": {"type": "string"}},
            "nlu_source": {"type": "string", "enum": ["openai"]},
        },
        "required": [
            "weeks_since_event",
            "event_type",
            "requested_exercise_text",
            "pain_score",
            "swelling_level",
            "weight_bearing",
            "red_flag_terms",
            "negated_terms",
            "missing_fields",
            "nlu_source",
        ],
    }


def _env_openai_client() -> OpenAI:
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # Optional: if your corporate network requires proxy, set HTTP(S)_PROXY in env.
    return OpenAI(api_key=api_key)


def extract_openai(user_text: str, timeout_s: float = 12.0) -> NLUOutput:
    """
    OpenAI-based extractor.
    MUST return structured NLUOutput.
    """
    client = _env_openai_client()

    schema = _build_response_schema()

    # Use Structured Outputs via response_format json_schema when available.
    # If your installed openai SDK doesn't support this, it will throw and be caught by fallback wrapper.
    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "nlu_output", "schema": schema, "strict": True},
        },
        timeout=timeout_s,
    )

    content = resp.choices[0].message.content
    if not content:
        raise RuntimeError("OpenAI returned empty content")

    data = json.loads(content)
    # Ensure nlu_source is openai (model should do it, but we enforce)
    data["nlu_source"] = "openai"

    return NLUOutput.model_validate(data)


def extract_with_fallback(user_text: str, timeout_s: float = 12.0) -> NLUOutput:
    try:
        nlu = extract_openai(user_text=user_text, timeout_s=timeout_s)
        return nlu
    except Exception as e:
        print("OpenAI extraction failed:", type(e).__name__, repr(e))
        nlu = extract_mock(user_text)
        nlu.nlu_source = "mock_fallback"
        return nlu