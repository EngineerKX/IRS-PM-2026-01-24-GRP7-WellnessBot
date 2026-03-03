from __future__ import annotations

from wellnessbot.dialog.state import ConversationState
from wellnessbot.nlu.schema import NLUOutput


def merge_turn(conv: ConversationState, nlu_turn: NLUOutput, user_text: str) -> ConversationState:
    conv.turn_count += 1
    conv.last_user_text = user_text

    # Only overwrite if new value is present and meaningful
    if (nlu_turn.event_type or "unknown") != "unknown":
        conv.event_type = nlu_turn.event_type

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

    # Audit history (optional but useful)
    conv.history.append(
        {
            "turn": conv.turn_count,
            "user_text": user_text,
            "nlu_turn": nlu_turn.model_dump(),
        }
    )

    return conv