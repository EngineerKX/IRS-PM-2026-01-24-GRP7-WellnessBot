from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
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

This is turn-level extraction.

You will receive:
1. The current user message
2. The slot the system asked for (expected_slot)

Rules:
- Extract information ONLY from the current user message.
- Do NOT use conversation history.
- If expected_slot is provided, interpret short answers accordingly.
- If a field is not mentioned, return null (or "unknown" for enums, false for booleans, [] for arrays).
- Do NOT make decisions.
- Do NOT suggest exercises.
- Do NOT ask clarifying questions.
- Do NOT explain anything.

Examples:

expected_slot = pain_score
User: "1"
→ pain_score = 1

expected_slot = pain_score
User: "2"
→ pain_score = 2

expected_slot = pain_score
User: "3"
→ pain_score = 3

expected_slot = pain_score
User: "0"
→ pain_score = 0

expected_slot = pain_score
User: "no pain"
→ pain_score = 0

expected_slot = pain_score
User: "none"
→ pain_score = 0

expected_slot = swelling_score
User: "1"
→ swelling_score = 1

expected_slot = swelling_score
User: "2"
→ swelling_score = 2

expected_slot = swelling_score
User: "3"
→ swelling_score = 3

expected_slot = swelling_score
User: "none"
→ swelling_score = 0

expected_slot = swelling_score
User: "no swelling"
→ swelling_score = 0

expected_slot = swelling_score
User: "0"
→ swelling_score = 0

expected_slot = symptom_screen
User: "none"
→ symptom_screen_done = true
→ symptom_flags = ["none"]
→ red_flag_terms = []
→ negated_terms = []
→ pain_score = null
→ swelling_score = null
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "pain"
→ symptom_screen_done = true
→ symptom_flags = ["pain"]
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "swelling"
→ symptom_screen_done = true
→ symptom_flags = ["swelling"]
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "pain and swelling"
→ symptom_screen_done = true
→ symptom_flags = ["pain", "swelling"]
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "fever"
→ symptom_screen_done = true
→ symptom_flags = ["fever"]
→ red_flag_terms = ["fever"]
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "excessive bleeding"
→ symptom_screen_done = true
→ symptom_flags = ["excessive_bleeding"]
→ red_flag_terms = ["excessive bleeding"]
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "fever, pain, swell"
→ symptom_screen_done = true
→ symptom_flags = ["fever", "pain", "swelling"]
→ red_flag_terms = ["fever"]
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "puffy"
→ symptom_screen_done = true
→ symptom_flags = ["swelling"]
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "2026-02-01"
→ symptom_screen_done = false
→ symptom_flags = []
→ requested_exercise_text = ""

expected_slot = weeks_since_event
User: "2 weeks"
→ weeks_since_event = 2

expected_slot = surgery_date
User: "2026-02-01"
→ surgery_date = "2026-02-01"
→ requested_exercise_text = ""

expected_slot = symptom_screen
User: "my knee feels puffy"
→ symptom_screen_done = true
→ symptom_flags = ["swelling"]

expected_slot = symptom_screen
User: "a bit swollen after walking"
→ symptom_screen_done = true
→ symptom_flags = ["swelling"]

expected_slot = symptom_screen
User: "sharp pain when bending"
→ symptom_screen_done = true
→ symptom_flags = ["pain"]

expected_slot = symptom_screen
User: "fever and swelling"
→ symptom_screen_done = true
→ symptom_flags = ["fever", "swelling"]
→ red_flag_terms = ["fever"]

expected_slot = symptom_screen
User: "no pain"
→ symptom_flags = []
→ red_flag_terms = []
→ negated_terms = ["pain"]

expected_slot = symptom_screen
User: "no swelling"
→ symptom_flags = []
→ red_flag_terms = []
→ negated_terms = ["swelling"]

expected_slot = symptom_screen
User: "no pain but swelling"
→ symptom_flags = ["swelling"]
→ red_flag_terms = []
→ negated_terms = ["pain"]

expected_slot = symptom_screen
User: "no fever"
→ symptom_flags = []
→ red_flag_terms = []
→ negated_terms = ["fever"]

expected_slot = symptom_screen
User: "my leg is swelling"
→ symptom_flags = ["swelling"]

expected_slot = symptom_screen
User: "pain and swell"
→ symptom_flags = ["pain", "swelling"]

expected_slot = symptom_screen
User: "I feel puffy"
→ symptom_flags = ["swelling"]

expected_slot = symptom_screen
User: "my knee hurts and is swollen"
→ symptom_flags = ["pain", "swelling"]

expected_slot = surgery_type
User: "arthroscopic knee surgery"
→ surgery_type = "arthroscopic_knee_surgery"

expected_slot = surgery_type
User: "post arthroscopic knee surgery"
→ surgery_type = "arthroscopic_knee_surgery"

expected_slot = surgery_type
User: "acl reconstruction"
→ surgery_type = "acl_reconstruction"

expected_slot = surgery_type
User: "tkr"
→ surgery_type = "tkr"

expected_slot = surgery_type
User: "sprain"
→ surgery_type = "sprain_non_surgical"

Field definitions:
- weeks_since_event: float weeks (convert days to weeks if explicitly given).
- surgery_type: one of ["arthroscopic_knee_surgery","acl_reconstruction","tkr","sprain_non_surgical","unknown"].
- requested_exercise_text: normalized short phrase ONLY if the user explicitly mentions an exercise they want to do. Otherwise return an empty string "".
- Never use "unknown" for requested_exercise_text. Use "" instead.
- pain_score: integer 0–3 if present.
- swelling_score: integer 0–3 if present.
- weight_bearing: one of ["none","partial","full","unknown"].
- symptom_screen_done: optional extraction hint only. The downstream policy layer decides conversation progress.
- symptom_flags: list from ["pain","swelling","fever","excessive_bleeding","none"] based only on the current message.
- red_flag_terms: list of explicitly mentioned red flag terms (non-negated), such as fever, excessive bleeding, wound drainage, pus, chest pain, shortness of breath, cannot bear weight, knee locking.
- negated_terms: list of explicitly negated terms.
- nlu_source: must be "openai".
- surgery_date: date string in YYYY-MM-DD if explicitly provided.

Important:
- A date-only answer like "2026-02-01" is NOT a symptom answer.
- "none" in symptom screening means no symptoms and should set symptom_screen_done=true and symptom_flags=["none"].
- When expected_slot = symptom_screen, do NOT fill pain_score or swelling_score unless the user explicitly gives those values.
- Normalize synonyms:
  - "swell", "swollen", "puffy" -> "swelling"
  - "bleed", "bleeding" -> "excessive bleeding"
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


def resolve_symptom_conflicts(nlu: NLUOutput) -> NLUOutput:
    flags = set(nlu.symptom_flags or [])
    red = set(nlu.red_flag_terms or [])
    neg = set(x.strip().lower() for x in (nlu.negated_terms or []))

    if "pain" in neg:
        flags.discard("pain")

    if "swelling" in neg:
        flags.discard("swelling")

    if "fever" in neg:
        flags.discard("fever")
        red = {x for x in red if x.strip().lower() != "fever"}

    if "excessive bleeding" in neg or "bleeding" in neg:
        flags.discard("excessive_bleeding")
        red = {
            x for x in red
            if x.strip().lower() not in {"excessive bleeding", "bleeding"}
        }

    if len(flags - {"none"}) > 0 and "none" in flags:
        flags.discard("none")

    nlu.symptom_flags = sorted(flags)
    nlu.red_flag_terms = sorted(red)
    return nlu


def normalize_expected_slot_numeric(
    user_text: str,
    expected_slot: str | None,
    nlu: NLUOutput,
) -> NLUOutput:
    text = (user_text or "").strip().lower()

    if expected_slot == "pain_score":
        if text in {"no pain", "none", "0"}:
            nlu.pain_score = 0
            return nlu

        if text.isdigit():
            value = int(text)
            if value in {0, 1, 2, 3}:
                nlu.pain_score = value

        return nlu

    if expected_slot == "swelling_score":
        if text in {"none", "no swelling", "0"}:
            nlu.swelling_score = 0
            return nlu

        if text.isdigit():
            value = int(text)
            if value in {0, 1, 2, 3}:
                nlu.swelling_score = value

        return nlu

    return nlu


def normalize_symptom_screen_if_needed(
    user_text: str,
    expected_slot: str | None,
    nlu: NLUOutput,
) -> NLUOutput:
    """
    Deterministic normalization for explicit symptom-screen replies.
    Critically, symptom-screen answers must not leak into pain/swelling slots.
    """
    if expected_slot != "symptom_screen":
        return nlu

    text = (user_text or "").strip().lower()

    # Hard reset slot-specific fields for symptom-screen turn.
    # Symptom screen should identify symptoms, not fill severity slots.
    nlu.pain_score = None
    nlu.swelling_score = None

    has_any_signal = bool(nlu.symptom_flags or nlu.red_flag_terms or nlu.negated_terms)
    if has_any_signal:
        if text in {"none", "no", "no symptoms"}:
            nlu.symptom_flags = ["none"]
            nlu.red_flag_terms = []
            nlu.negated_terms = []
        return nlu

    if text in {"none", "no", "no symptoms"}:
        nlu.symptom_flags = ["none"]
        nlu.red_flag_terms = []
        nlu.negated_terms = []

    return nlu


def normalize_requested_exercise_if_needed(nlu: NLUOutput) -> NLUOutput:
    text = (nlu.requested_exercise_text or "").strip().lower()
    if text in {"unknown", "none", "n/a", "na", "nil"}:
        nlu.requested_exercise_text = ""
    return nlu


def normalize_surgery_type(nlu: NLUOutput) -> NLUOutput:
    if nlu.surgery_type == "post_arthroscopic_knee_surgery":
        nlu.surgery_type = "arthroscopic_knee_surgery"
    return nlu


def _build_response_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "weeks_since_event": {"type": ["number", "null"]},
            "surgery_type": {
                "type": "string",
                "enum": [
                    "arthroscopic_knee_surgery",
                    "acl_reconstruction",
                    "tkr",
                    "sprain_non_surgical",
                    "unknown",
                ],
            },
            "surgery_date": {"type": "string"},
            "requested_exercise_text": {"type": "string"},
            "pain_score": {"type": ["integer", "null"], "minimum": 0, "maximum": 3},
            "swelling_score": {"type": ["integer", "null"], "minimum": 0, "maximum": 3},
            "weight_bearing": {
                "type": "string",
                "enum": ["none", "partial", "full", "unknown"],
            },
            "symptom_screen_done": {"type": "boolean"},
            "symptom_flags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["pain", "swelling", "fever", "excessive_bleeding", "none"],
                },
            },
            "red_flag_terms": {"type": "array", "items": {"type": "string"}},
            "negated_terms": {"type": "array", "items": {"type": "string"}},
            "missing_fields": {"type": "array", "items": {"type": "string"}},
            "nlu_source": {"type": "string", "enum": ["openai"]},
        },
        "required": [
            "weeks_since_event",
            "surgery_type",
            "surgery_date",
            "requested_exercise_text",
            "pain_score",
            "swelling_score",
            "weight_bearing",
            "symptom_screen_done",
            "symptom_flags",
            "red_flag_terms",
            "negated_terms",
            "missing_fields",
            "nlu_source",
        ],
    }


def _env_openai_client() -> Any:
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
    data["nlu_source"] = "openai"

    nlu = NLUOutput.model_validate(data)
    nlu = convert_date_to_weeks_if_needed(user_text, nlu)
    nlu = normalize_symptom_screen_if_needed(user_text, expected_slot, nlu)
    nlu = normalize_requested_exercise_if_needed(nlu)
    nlu = normalize_expected_slot_numeric(user_text, expected_slot, nlu)
    nlu = resolve_symptom_conflicts(nlu)
    nlu = normalize_surgery_type(nlu)
    return nlu


def extract_with_fallback(
    user_text: str,
    expected_slot: str | None = None,
    timeout_s: float = 12.0,
) -> NLUOutput:
    nlu = extract_openai(
        user_text=user_text,
        expected_slot=expected_slot,
        timeout_s=timeout_s,
    )

    nlu = convert_date_to_weeks_if_needed(user_text, nlu)
    nlu = normalize_symptom_screen_if_needed(user_text, expected_slot, nlu)
    nlu = normalize_requested_exercise_if_needed(nlu)
    nlu = normalize_expected_slot_numeric(user_text, expected_slot, nlu)
    nlu = resolve_symptom_conflicts(nlu)
    nlu = normalize_surgery_type(nlu)


    return nlu