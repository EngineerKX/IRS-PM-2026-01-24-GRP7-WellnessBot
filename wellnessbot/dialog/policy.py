from __future__ import annotations

from typing import List, Optional
from wellnessbot.dialog.state import ConversationState
from wellnessbot.dialog.questions import QUESTION_BANK


def build_symptom_screen_question(conv: ConversationState) -> str:
    known = set(conv.symptom_flags or [])
    known.discard("none")

    if "swelling" in known:
        return (
            "I understand you have swelling. "
            "Do you also have fever, excessive bleeding, redness, or pain? "
            'If none, say "none".'
        )

    if "pain" in known:
        return (
            "I understand you have pain. "
            "Do you also have fever, excessive bleeding, redness, or swelling? "
            'If none, say "none".'
        )

    if known:
        known_text = ", ".join(sorted(known))
        return (
            f"I understand you mentioned {known_text}. "
            "Do you also have fever, excessive bleeding, redness, swelling, or pain? "
            'If none, say "none".'
        )

    return QUESTION_BANK.get(
        "symptom_screen",
        'Are you having any symptoms today, such as fever, excessive bleeding, unusual swelling, or pain? If none, just say "none".'
    )


def compute_missing_slots(conv: ConversationState) -> List[str]:
    conv.symptom_screen_done = compute_symptom_screen_done(conv)

    if (conv.event_type or "unknown") == "unknown":
        return ["event_type"]

    if not (conv.surgery_date or "").strip() and conv.weeks_since_event is None:
        return ["surgery_date"]

    if not conv.symptom_screen_done:
        return ["symptom_screen"]

    symptom_flags = set(conv.symptom_flags or [])

    if "pain" in symptom_flags and conv.pain_score is None:
        return ["pain_score"]

    if "swelling" in symptom_flags and (conv.swelling_level or "unknown") == "unknown":
        return ["swelling_level"]

    return []

def next_question_for_missing(
    conv: ConversationState,
    missing_slots: List[str]
) -> Optional[dict]:
    if not missing_slots:
        return None

    for slot in missing_slots:
        if slot not in (conv.asked_slots or []):
            conv.asked_slots.append(slot)
            return {
                "slot_name": slot,
                "question": QUESTION_BANK.get(slot, f"Please provide: {slot}"),
            }

    # fallback: ask first if everything was already asked
    slot = missing_slots[0]
    question = (
        build_symptom_screen_question(conv)
        if slot == "symptom_screen"
        else QUESTION_BANK.get(slot, f"Please provide: {slot}")
    )

    return {
        "slot_name": slot,
        "question": question,
    }



def compute_symptom_screen_done(conv: ConversationState) -> bool:
    # Case 1: explicit "none"
    if "none" in (conv.symptom_flags or []):
        return True

    # Case 2: symptoms were extracted AFTER symptom_screen was asked
    if conv.symptom_flags:
        # Only valid if symptom_screen was actually asked before
        if "symptom_screen" in (conv.asked_slots or []):
            return True

    # Case 3: red flag
    if conv.red_flag_terms:
        return True

    return False