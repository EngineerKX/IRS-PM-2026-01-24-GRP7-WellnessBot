from __future__ import annotations

from SystemCode.wellnessbot.dialog.state import ConversationState
from SystemCode.wellnessbot.nlu.schema import NLUOutput


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

    if nlu_turn.swelling_score is not None:
        conv.swelling_score = int(nlu_turn.swelling_score)

    if (nlu_turn.weight_bearing or "unknown") != "unknown":
        conv.weight_bearing = nlu_turn.weight_bearing

    if (nlu_turn.requested_exercise_text or "").strip():
        conv.requested_exercise_text = nlu_turn.requested_exercise_text.strip()

    # Symptom-screen turns are authoritative.
    # If user says "none", clear prior symptom/red-flag state.
    if expected_slot == "symptom_screen":
        new_flags = set(x.strip().lower() for x in (nlu_turn.symptom_flags or []) if str(x).strip())
        new_red_flags = set(x.strip().lower() for x in (nlu_turn.red_flag_terms or []) if str(x).strip())

        if new_flags == {"none"}:
            conv.symptom_flags = ["none"]
            conv.red_flag_terms = []
        else:
            conv.symptom_flags = sorted(new_flags) if new_flags else []
            conv.red_flag_terms = sorted(new_red_flags) if new_red_flags else []

    else:
        # Non-symptom turns should not erase prior symptom-screen result,
        # but may add new symptom details if explicitly extracted.
        if getattr(nlu_turn, "red_flag_terms", None):
            existing = set(x.strip().lower() for x in (conv.red_flag_terms or []))
            existing.update(x.strip().lower() for x in nlu_turn.red_flag_terms if str(x).strip())
            conv.red_flag_terms = sorted(existing)

        if getattr(nlu_turn, "symptom_flags", None):
            existing_flags = set(x.strip().lower() for x in (conv.symptom_flags or []))
            new_flags = set(x.strip().lower() for x in (nlu_turn.symptom_flags or []))

            if new_flags == {"none"}:
                conv.symptom_flags = ["none"] if not existing_flags else sorted(existing_flags)
            else:
                merged_flags = (existing_flags | new_flags) - {"none"}
                conv.symptom_flags = sorted(merged_flags)

    if getattr(nlu_turn, "negated_terms", None):
        existing_neg = set(x.strip().lower() for x in (conv.negated_terms or []))
        new_neg = set(x.strip().lower() for x in (nlu_turn.negated_terms or []))
        conv.negated_terms = sorted(existing_neg | new_neg)

    conv.history.append(
        {
            "turn": conv.turn_count,
            "user_text": user_text,
            "expected_slot": expected_slot,
            "nlu_turn": nlu_turn.model_dump(),
        }
    )

    return conv