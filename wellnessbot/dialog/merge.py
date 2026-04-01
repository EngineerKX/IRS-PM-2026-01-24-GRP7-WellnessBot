from __future__ import annotations

from wellnessbot.dialog.state import ConversationState
from wellnessbot.nlu.schema import NLUOutput


def merge_turn(
    conv: ConversationState,
    nlu_turn: NLUOutput,
    user_text: str,
    expected_slot: str | None = None,
) -> ConversationState:
    conv.turn_count += 1
    conv.last_user_text = user_text

    if (nlu_turn.surgery_type or "unknown") != "unknown":
        conv.surgery_type = nlu_turn.surgery_type

    if (getattr(nlu_turn, "surgery_date", "") or "").strip():
        conv.surgery_date = nlu_turn.surgery_date.strip()

    if nlu_turn.weeks_since_event is not None:
        conv.weeks_since_event = float(nlu_turn.weeks_since_event)

    if nlu_turn.pain_score is not None:
        conv.pain_score = int(nlu_turn.pain_score)

    if (nlu_turn.swelling_level or "unknown") != "unknown":
        conv.swelling_level = nlu_turn.swelling_level

    if (nlu_turn.weight_bearing or "unknown") != "unknown":
        conv.weight_bearing = nlu_turn.weight_bearing

    if (nlu_turn.requested_exercise_text or "").strip():
        conv.requested_exercise_text = nlu_turn.requested_exercise_text.strip()

    if getattr(nlu_turn, "red_flag_terms", None):
        existing = set(conv.red_flag_terms or [])
        existing.update(x.strip().lower() for x in nlu_turn.red_flag_terms if str(x).strip())
        conv.red_flag_terms = sorted(existing)

    if getattr(nlu_turn, "symptom_flags", None):
        existing_flags = set(x.strip().lower() for x in (conv.symptom_flags or []))
        new_flags = set(x.strip().lower() for x in (nlu_turn.symptom_flags or []))

        # "none" means no additional symptoms in this turn.
        # It should NOT erase earlier positive symptoms like swelling/pain.
        if new_flags == {"none"}:
            if not existing_flags:
                conv.symptom_flags = ["none"]
            else:
                conv.symptom_flags = sorted(existing_flags)
        else:
            merged_flags = (existing_flags | new_flags) - {"none"}
            conv.symptom_flags = sorted(merged_flags)

    conv.history.append(
        {
            "turn": conv.turn_count,
            "user_text": user_text,
            "expected_slot": expected_slot,
            "nlu_turn": nlu_turn.model_dump(),
        }
    )

    return conv