from __future__ import annotations

import json
import os
from typing import Any, Dict

from wellnessbot.nlu.mock_extractor import extract_mock
from wellnessbot.nlu.schema import NLUOutput

from datetime import datetime, timezone
import re

# OpenAI is optional dependency at runtime when MOCK_NLU=0.
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


_SYSTEM_PROMPT = """
You are a structured NLU extraction component.

You MUST output a single JSON object ONLY (no markdown, no prose).

This is turn-level extraction.

You will receive:
1. The current user message
2. The slot the system asked for (expected_slot)

Rules:

- Extract information ONLY from the current user message.
- Do NOT use conversation history.
- If expected_slot is provided, interpret short answers accordingly.

Examples:

expected_slot = pain_score
User: "1"
→ pain_score = 1

expected_slot = swelling_level
User: "mild"
→ swelling_level = "mild"

expected_slot = weeks_since_event
User: "2 weeks"
→ weeks_since_event = 2

If the answer does not match the expected slot, extract normally.

If a field is not mentioned, return null (or "unknown" for enums).

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
- surgery_date: date string in YYYY-MM-DD if explicitly provided.

You are NOT allowed to:
- Make decisions.
- Suggest exercises.
- Ask clarifying questions.
- Explain anything.
"""



def convert_date_to_weeks_if_needed(user_text: str, nlu: NLUOutput) -> NLUOutput:
    """
    If the user provided a date (YYYY-MM-DD), store it and convert it to weeks_since_event
    when weeks_since_event is not already present.
    """
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", user_text)
    if not match:
        return nlu

    date_str = match.group(1)

    if not (getattr(nlu, "surgery_date", "") or "").strip():
        nlu.surgery_date = date_str

    if nlu.weeks_since_event is not None:
        return nlu

    try:
        surgery_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta_days = (now - surgery_date).days
        weeks = round(delta_days / 7, 2)

        if weeks >= 0:
            nlu.weeks_since_event = weeks
    except Exception:
        pass

    return nlu


def apply_missing_fields_policy(nlu: NLUOutput) -> NLUOutput:
    """
    Deterministic missing-fields policy (do not trust the model for this).
    """
    missing = []
    if nlu.weeks_since_event is None and not (nlu.surgery_date or "").strip():
        missing.append("surgery_date")
    if (nlu.event_type or "unknown") == "unknown":
        missing.append("event_type")

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
            "surgery_date": {"type": "string"},
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
            "surgery_date",
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


def extract_openai(
    user_text: str,
    expected_slot: str | None = None,
    timeout_s: float = 12.0,
) -> NLUOutput:
    client = _env_openai_client()
    schema = _build_response_schema()

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"expected_slot: {expected_slot}\n\nuser_message:\n{user_text}",
            },
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
    nlu = convert_date_to_weeks_if_needed(user_text, nlu)
    nlu = apply_missing_fields_policy(nlu)
    return nlu


def extract_with_fallback(
    user_text: str,
    expected_slot: str | None = None,
    timeout_s: float = 12.0,
) -> NLUOutput:
    try:
        return extract_openai(
            user_text=user_text,
            expected_slot=expected_slot,
            timeout_s=timeout_s,
        )
    except Exception as e:
        print("OpenAI extraction failed:", type(e).__name__, repr(e))
        nlu = extract_mock(user_text)
        nlu.nlu_source = "mock_fallback"
        nlu = convert_date_to_weeks_if_needed(user_text, nlu)        
        nlu = apply_missing_fields_policy(nlu)
        return nlu