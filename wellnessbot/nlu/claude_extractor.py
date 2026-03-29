from __future__ import annotations

import os
from typing import Any, Dict

from wellnessbot.nlu.mock_extractor import extract_mock
from wellnessbot.nlu.openai_extractor import apply_missing_fields_policy
from wellnessbot.nlu.schema import NLUOutput

# Anthropic is optional dependency at runtime when NLU_PROVIDER=claude.
try:
    from anthropic import Anthropic
except Exception:  # pragma: no cover
    Anthropic = None  # type: ignore

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

_SYSTEM_PROMPT = """You are an NLU extraction component for a knee rehabilitation decision support system.
Extract structured fields from the user's text by calling the extract_nlu tool.
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
"""

# Tool definition that enforces structured output via Claude's tool use
_NLU_TOOL: Dict[str, Any] = {
    "name": "extract_nlu",
    "description": "Extract structured NLU fields from the user's rehabilitation query.",
    "input_schema": {
        "type": "object",
        "properties": {
            "weeks_since_event": {
                "type": ["number", "null"],
                "description": "Weeks since injury/surgery. Convert days to weeks (days/7). null if not mentioned.",
            },
            "event_type": {
                "type": "string",
                "enum": ["acl_surgery", "tkr", "meniscus", "sprain", "unknown"],
                "description": "Type of knee event/injury.",
            },
            "requested_exercise_text": {
                "type": "string",
                "description": "Normalized exercise name (e.g., 'squats', 'heel slides'). Empty string if not mentioned.",
            },
            "pain_score": {
                "type": ["integer", "null"],
                "description": "Pain score 0-10. null if not mentioned.",
            },
            "swelling_level": {
                "type": "string",
                "enum": ["none", "mild", "moderate", "severe", "unknown"],
                "description": "Swelling level.",
            },
            "weight_bearing": {
                "type": "string",
                "enum": ["none", "partial", "full", "unknown"],
                "description": "Weight bearing status.",
            },
            "red_flag_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Red flag symptoms mentioned (non-negated).",
            },
            "negated_terms": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Terms explicitly negated (e.g., 'no fever' => 'fever').",
            },
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
        ],
    },
}


def _env_anthropic_client() -> Any:
    """Return Anthropic client instance."""
    if Anthropic is None:
        raise RuntimeError("anthropic package not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    return Anthropic(api_key=api_key)


def extract_claude(user_text: str, timeout_s: float = 12.0) -> NLUOutput:
    client = _env_anthropic_client()

    resp = client.messages.create(
        model=MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
        tools=[_NLU_TOOL],
        tool_choice={"type": "tool", "name": "extract_nlu"},
        timeout=timeout_s,
    )

    # Extract the tool use block from the response
    tool_block = next(
        (block for block in resp.content if block.type == "tool_use"),
        None,
    )
    if not tool_block:
        raise RuntimeError("Claude did not return a tool_use block")

    data = tool_block.input
    data["nlu_source"] = "claude"
    data["missing_fields"] = []

    nlu = NLUOutput.model_validate(data)
    nlu = apply_missing_fields_policy(nlu)
    return nlu


def extract_claude_with_fallback(user_text: str, timeout_s: float = 12.0) -> NLUOutput:
    try:
        return extract_claude(user_text=user_text, timeout_s=timeout_s)
    except Exception as e:
        print("Claude extraction failed:", type(e).__name__, repr(e))
        nlu = extract_mock(user_text)
        nlu.nlu_source = "claude_fallback"
        nlu = apply_missing_fields_policy(nlu)
        return nlu
