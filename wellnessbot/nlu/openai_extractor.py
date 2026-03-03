from __future__ import annotations

import json
import os
from typing import Any, Dict

from wellnessbot.nlu.mock_extractor import extract_mock
from wellnessbot.nlu.schema import NLUOutput

# OpenAI is optional dependency at runtime when MOCK_NLU=0.
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


_SYSTEM_PROMPT = """
You are a structured NLU extraction component.

You MUST output a single JSON object ONLY (no markdown, no prose).

This is turn-level extraction:
- Extract information ONLY from the current user message.
- Do NOT use memory.
- Do NOT infer missing fields.
- If a field is not mentioned, return null (or "unknown" for enums).

Field definitions:

- weeks_since_event: float weeks (convert days to weeks if explicitly given).
- event_type: one of ["acl_surgery","tkr","meniscus","sprain","unknown"].
- requested_exercise_text: normalized short phrase.
- pain_score: integer 0–10 if present.
- swelling_level: one of ["none","mild","moderate","severe","unknown"].
- weight_bearing: one of ["none","partial","full","unknown"].
- red_flag_terms: list of explicitly mentioned red flag terms (non-negated).
- negated_terms: list of explicitly negated terms.
- nlu_source: must be "openai".

You are NOT allowed to:
- Make decisions.
- Suggest exercises.
- Ask clarifying questions.
- Explain anything.
"""


def apply_missing_fields_policy(nlu: NLUOutput) -> NLUOutput:
    """
    Deterministic missing-fields policy (do not trust the model for this).
    """
    missing = []
    if nlu.weeks_since_event is None:
        missing.append("weeks_since_event")
    if not (nlu.requested_exercise_text or "").strip():
        missing.append("requested_exercise_text")

    nlu.missing_fields = missing
    return nlu


def _build_response_schema() -> Dict[str, Any]:
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


def _env_openai_client() -> Any:
    """
    Return OpenAI client instance.
    Type is Any to avoid Pylance 'variable not allowed in type expression'
    because OpenAI may be None depending on optional import.
    """
    if OpenAI is None:
        raise RuntimeError("openai package not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    return OpenAI(api_key=api_key)


def extract_openai(user_text: str, timeout_s: float = 12.0) -> NLUOutput:
    client = _env_openai_client()
    schema = _build_response_schema()

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
    data["nlu_source"] = "openai"  # enforce

    nlu = NLUOutput.model_validate(data)
    nlu = apply_missing_fields_policy(nlu)
    return nlu


def extract_with_fallback(user_text: str, timeout_s: float = 12.0) -> NLUOutput:
    try:
        return extract_openai(user_text=user_text, timeout_s=timeout_s)
    except Exception as e:
        print("OpenAI extraction failed:", type(e).__name__, repr(e))
        nlu = extract_mock(user_text)
        nlu.nlu_source = "mock_fallback"
        nlu = apply_missing_fields_policy(nlu)
        return nlu