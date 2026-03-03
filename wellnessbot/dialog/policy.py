from __future__ import annotations

from typing import List, Optional
from wellnessbot.dialog.state import ConversationState
from wellnessbot.dialog.questions import QUESTION_BANK, REQUIRED_ORDER


def compute_missing_slots(conv: ConversationState) -> List[str]:
    missing: List[str] = []

    if conv.weeks_since_event is None:
        missing.append("weeks_since_event")

    if (conv.event_type or "unknown") == "unknown":
        missing.append("event_type")

    if not (conv.requested_exercise_text or "").strip():
        missing.append("requested_exercise_text")

    if conv.pain_score is None:
        missing.append("pain_score")

    if (conv.swelling_level or "unknown") == "unknown":
        missing.append("swelling_level")

    if (conv.weight_bearing or "unknown") == "unknown":
        missing.append("weight_bearing")

    # Keep order stable
    ordered = [s for s in REQUIRED_ORDER if s in missing]
    return ordered


def next_question_for_missing(missing_slots: List[str]) -> Optional[dict]:
    if not missing_slots:
        return None
    slot = missing_slots[0]
    return {
        "slot_name": slot,
        "question": QUESTION_BANK.get(slot, f"Please provide: {slot}"),
    }