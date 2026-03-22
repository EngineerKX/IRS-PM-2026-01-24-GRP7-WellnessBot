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

    text = (user_text or "").strip().lower()

    if (nlu_turn.event_type or "unknown") != "unknown":
        conv.event_type = nlu_turn.event_type

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
        existing.update(nlu_turn.red_flag_terms)
        conv.red_flag_terms = sorted(existing)

    symptom_flags = set(conv.symptom_flags or [])

    if expected_slot == "symptom_screen":
        answered_symptom_screen = False

        if "pain" in text:
            symptom_flags.add("pain")
            answered_symptom_screen = True

        # swelling detection (more robust)
        if (
            "swelling" in text
            or "swell" in text
            or "swollen" in text
            or "puffy" in text
            or (nlu_turn.swelling_level or "unknown") != "unknown"
        ):
            symptom_flags.add("swelling")
            answered_symptom_screen = True

        if "fever" in text:
            symptom_flags.add("fever")
            answered_symptom_screen = True

        if any(k in text for k in ["bleeding", "bleed", "wound drainage", "pus"]):
            symptom_flags.add("excessive_bleeding")
            answered_symptom_screen = True

        if text in {"none", "no", "no symptoms", "i feel okay", "okay"}:
            symptom_flags.add("none")
            answered_symptom_screen = True

        if answered_symptom_screen:
            conv.symptom_screen_done = True

    if expected_slot == "pain_score" and nlu_turn.pain_score is not None:
        symptom_flags.add("pain")

    if expected_slot == "swelling_level" and (nlu_turn.swelling_level or "unknown") != "unknown":
        symptom_flags.add("swelling")

    conv.symptom_flags = sorted(symptom_flags)

    conv.history.append(
        {
            "turn": conv.turn_count,
            "user_text": user_text,
            "expected_slot": expected_slot,
            "nlu_turn": nlu_turn.model_dump(),
        }
    )

    return conv