from __future__ import annotations

from typing import List, Optional
from wellnessbot.dialog.state import ConversationState
from wellnessbot.dialog.questions import QUESTION_BANK


def compute_missing_slots(conv: ConversationState) -> List[str]:
    # 1) Core profile first
    if not (conv.surgery_date or "").strip() and conv.weeks_since_event is None:
        return ["surgery_date"]

    if (conv.event_type or "unknown") == "unknown":
        return ["event_type"]

    # 2) Symptom screening must happen before recommendation
    if not getattr(conv, "symptom_screen_done", False):
        return ["symptom_screen"]

    symptom_flags = set(getattr(conv, "symptom_flags", []) or [])

    # 3) Follow-up only when symptoms were mentioned
    if "pain" in symptom_flags and conv.pain_score is None:
        return ["pain_score"]

    if "swelling" in symptom_flags and (conv.swelling_level or "unknown") == "unknown":
        return ["swelling_level"]

    return []


def next_question_for_missing(missing_slots: List[str]) -> Optional[dict]:
    if not missing_slots:
        return None
    slot = missing_slots[0]
    return {
        "slot_name": slot,
        "question": QUESTION_BANK.get(slot, f"Please provide: {slot}"),
    }