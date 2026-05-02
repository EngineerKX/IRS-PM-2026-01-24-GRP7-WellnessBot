from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from SystemCode.wellnessbot.dialog.state import ConversationState
from SystemCode.wellnessbot.dialog.questions import QUESTION_BANK
from SystemCode.wellnessbot.kg.kg import (
    phase_from_weeks,
    get_redflag_policies,
    symptom_matches_policy,
)


def _current_phase_from_conv(conv: ConversationState) -> Optional[str]:
    if (conv.surgery_type or "unknown") == "unknown":
        return None

    weeks_since_event = conv.weeks_since_event

    if weeks_since_event is None and (conv.surgery_date or "").strip():
        try:
            dt = datetime.strptime(conv.surgery_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta_days = (datetime.now(timezone.utc) - dt).days
            if delta_days >= 0:
                weeks_since_event = round(delta_days / 7, 2)
        except Exception:
            return None

    if weeks_since_event is None:
        return None

    return phase_from_weeks(weeks_since_event, conv.surgery_type)


def _matching_redflag_policies(conv: ConversationState, phase_id: str | None):
    if not phase_id:
        return []

    red_flags = set(x.strip().lower() for x in (conv.red_flag_terms or []))
    if not red_flags:
        return []

    policies = get_redflag_policies(conv.surgery_type, phase_id)
    if not policies:
        return []

    matched = []
    for p in policies:
        symptom = (p.symptom or "").strip().lower()
        if symptom_matches_policy(conv.surgery_type, symptom, list(red_flags)):
            matched.append(p)

    return matched


def _should_short_circuit_red_flags(conv: ConversationState, phase_id: str | None) -> bool:
    matched_policies = _matching_redflag_policies(conv, phase_id)
    if not matched_policies:
        return False

    for p in matched_policies:
        action = (p.action or "").strip().lower()
        if action == "escalate":
            return True

    return False


def _should_collect_severity_for_red_flags(conv: ConversationState, phase_id: str | None) -> bool:
    matched_policies = _matching_redflag_policies(conv, phase_id)
    if not matched_policies:
        return False

    for p in matched_policies:
        action = (p.action or "").strip().lower()
        steps = set(x.strip().lower() for x in (p.action_steps or []))

        if action == "supportive_sequence" and "stop_exercise" in steps:
            return True

    return False


def build_symptom_screen_question(conv: ConversationState) -> str:
    known = set(conv.symptom_flags or [])
    known.discard("none")

    if "fever" in known:
        return (
            "I understand you mentioned fever. "
            "Do you also have excessive bleeding or wound drainage? "
            'If none, say "none".'
        )

    if "excessive_bleeding" in known:
        return (
            "I understand you mentioned excessive bleeding. "
            "Do you also have fever or wound drainage? "
            'If none, say "none".'
        )

    if known:
        known_text = ", ".join(sorted(known))
        return (
            f"I understand you mentioned {known_text}. "
            "Do you also have fever, excessive bleeding, or wound drainage? "
            'If none, say "none".'
        )

    return QUESTION_BANK.get(
        "symptom_screen",
        'Are you having any symptoms today, such as fever or excessive bleeding? If none, just say "none".'
    )


def compute_missing_slots(conv: ConversationState) -> List[str]:
    conv.symptom_screen_done = compute_symptom_screen_done(conv)

    if not (conv.surgery_date or "").strip() and conv.weeks_since_event is None:
        return ["surgery_date"]

    phase_id = _current_phase_from_conv(conv)

    # terminal phase shortcut: bypass all further slot collection
    if phase_id and phase_id >= "P5":
        conv.pending_followup_slots = []
        return []

    if not conv.symptom_screen_done:
        return ["symptom_screen"]

    symptom_flags = set(x.strip().lower() for x in (conv.symptom_flags or []))
    red_flags = set(x.strip().lower() for x in (conv.red_flag_terms or []))

    # Short-circuit only when the KG policy for this phase says the red flag escalates
    if _should_short_circuit_red_flags(conv, phase_id):
        conv.pending_followup_slots = []
        return []

    # Handle pending follow-ups
    if conv.pending_followup_slots:
        remaining = []

        for slot in conv.pending_followup_slots:
            if slot == "pain_score" and conv.pain_score is None:
                remaining.append(slot)
            elif slot == "swelling_score" and conv.swelling_score is None:
                remaining.append(slot)

        conv.pending_followup_slots = remaining
        if remaining:
            return remaining

    # KG-driven supportive-sequence cases that still require severity collection
    if _should_collect_severity_for_red_flags(conv, phase_id):
        missing = []

        if conv.pain_score is None:
            missing.append("pain_score")

        if conv.swelling_score is None:
            missing.append("swelling_score")

        conv.exercise_blocked = True
        conv.block_reason = "redflag_supportive_sequence"
        conv.pending_followup_slots = missing.copy()

        if missing:
            return missing

    # NONE → still need severity collection
    if "none" in symptom_flags:
        missing = []

        if conv.pain_score is None:
            missing.append("pain_score")

        if conv.swelling_score is None:
            missing.append("swelling_score")

        conv.pending_followup_slots = missing.copy()

        if missing:
            return missing

    # Any symptom report (including non-escalating red-flag mentions) should still collect severity
    if (symptom_flags - {"none"}) or red_flags:
        missing = []

        if conv.pain_score is None:
            missing.append("pain_score")

        if conv.swelling_score is None:
            missing.append("swelling_score")

        conv.pending_followup_slots = missing.copy()

        if missing:
            return missing

    # Normal flow
    if "pain" in symptom_flags and conv.pain_score is None:
        return ["pain_score"]

    if "swelling" in symptom_flags and conv.swelling_score is None:
        return ["swelling_score"]

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
    symptom_flags = set(x.strip().lower() for x in (conv.symptom_flags or []))

    if "none" in symptom_flags:
        return True

    if symptom_flags - {"none"}:
        return True

    if conv.red_flag_terms:
        return True

    return False