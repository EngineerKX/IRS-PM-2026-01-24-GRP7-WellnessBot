from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from SystemCode.wellnessbot.kg.kg import (
    get_exercise,
    get_protocol_for_surgery_type,
    resolve_exercise_id,
    phase_from_weeks,
)
from SystemCode.wellnessbot.nlu.mock_extractor import extract_mock
from SystemCode.wellnessbot.nlu.openai_extractor import extract_with_fallback
from SystemCode.wellnessbot.nlu.schema import NLUOutput
from SystemCode.wellnessbot.planner.planner import plan, plan_alternatives
from SystemCode.wellnessbot.rules.engine import evaluate_rules
from SystemCode.wellnessbot.rules.rule_types import Action
from SystemCode.wellnessbot.state.infer import infer_state
from SystemCode.wellnessbot.logging.logger import log_interaction

from SystemCode.wellnessbot.dialog.state import ConversationState
from SystemCode.wellnessbot.dialog.merge import merge_turn
from SystemCode.wellnessbot.dialog.policy import compute_missing_slots, next_question_for_missing


def _make_interaction_id(user_text: str, ts: str) -> str:
    return hashlib.sha256(f"{ts}|{user_text}".encode("utf-8")).hexdigest()[:16]


def _phase_banner_from_conv(conv: ConversationState) -> str:
    surgery_type = conv.surgery_type
    if surgery_type in (None, "", "unknown"):
        return ""

    weeks_since_event = conv.weeks_since_event

    if weeks_since_event is None and (conv.surgery_date or "").strip():
        try:
            dt = datetime.strptime(conv.surgery_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta_days = (datetime.now(timezone.utc) - dt).days
            if delta_days >= 0:
                weeks_since_event = round(delta_days / 7, 2)
        except Exception:
            return ""

    if weeks_since_event is None:
        return ""

    phase_id = phase_from_weeks(weeks_since_event, surgery_type)
    if not phase_id:
        return ""

    protocol = get_protocol_for_surgery_type(surgery_type)
    if not protocol:
        return f"**Current phase:** {phase_id}\n\n"

    for ph in protocol.phases:
        if ph.phase_id == phase_id:
            return f"**Current phase:** {phase_id} ({ph.name})\n\n"

    return f"**Current phase:** {phase_id}\n\n"


def _extract_selfcare_routine(fired_rules: List[Any]) -> List[str]:
    out: List[str] = []

    for r in fired_rules:
        rule_id = getattr(r, "rule_id", "") or ""
        rationale = (getattr(r, "rationale", "") or "").strip()

        if rule_id.startswith("R_SELFCARE_") and rationale:
            out.append(rationale)

    # deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for item in out:
        if item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def run_pipeline(
    user_text: str,
    *,
    conv_state: Dict[str, Any] | None = None,
    force_mock_nlu: bool = False,
) -> Dict[str, Any]:
    """
    Loop-aware pipeline:
    1) NLU (turn-level)
    2) Merge into conversation state
    3) If missing slots -> CLARIFY mode
    4) Else -> run deterministic decision engine
    """

    conv = ConversationState.from_dict(conv_state or {})
    planner_history = (conv_state or {}).get("exercise_history", []) or []

    prior_missing_slots = compute_missing_slots(conv)
    expected_slot = prior_missing_slots[0] if prior_missing_slots else None

    mock_env = os.getenv("MOCK_NLU", "0").strip() == "1"
    use_mock = force_mock_nlu or mock_env
    conv.last_expected_slot = expected_slot or ""

    if use_mock:
        nlu_turn: NLUOutput = extract_mock(user_text)
    else:
        nlu_turn = extract_with_fallback(
            user_text=user_text,
            expected_slot=expected_slot,
            timeout_s=12.0,
        )

    conv = merge_turn(conv, nlu_turn, user_text, expected_slot=expected_slot)

    # --------------------------------------------
    # CLARIFY MODE
    # --------------------------------------------
    missing_slots = compute_missing_slots(conv)

    if missing_slots:
        next_q = next_question_for_missing(conv, missing_slots)

        phase_banner = ""
        if next_q["slot_name"] == "symptom_screen":
            phase_banner = _phase_banner_from_conv(conv)

        question_text = f"{phase_banner}{next_q['question']}"

        audit_trace = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "clarify",
            "missing_slots": missing_slots,
            "asked_slot": next_q["slot_name"],
            "notes": ["Clarify is system-driven (not LLM)."],
        }

        result = {
            "mode": "clarify",
            "question": question_text,
            "slot_name": next_q["slot_name"],
            "conv_state": conv.to_dict(),
            "nlu_turn": nlu_turn.model_dump(),
            "audit_trace": audit_trace,
            "user_text": user_text,
        }

        return result

    # --------------------------------------------
    # FINAL MODE
    # --------------------------------------------
    weeks_since_event = conv.weeks_since_event

    if weeks_since_event is None and (conv.surgery_date or "").strip():
        try:
            dt = datetime.strptime(conv.surgery_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            delta_days = (datetime.now(timezone.utc) - dt).days
            if delta_days >= 0:
                weeks_since_event = round(delta_days / 7, 2)
        except Exception:
            pass

    nlu_full = NLUOutput.model_validate(
        {
            "weeks_since_event": weeks_since_event,
            "surgery_type": conv.surgery_type,
            "surgery_date": conv.surgery_date,
            "requested_exercise_text": conv.requested_exercise_text,
            "pain_score": conv.pain_score,
            "swelling_score": conv.swelling_score,
            "weight_bearing": conv.weight_bearing,
            "symptom_screen_done": conv.symptom_screen_done,
            "symptom_flags": conv.symptom_flags,
            "red_flag_terms": conv.red_flag_terms,
            "negated_terms": conv.negated_terms or nlu_turn.negated_terms,
            "missing_fields": [],
            "nlu_source": nlu_turn.nlu_source,
        }
    )

    state = infer_state(nlu_full)

    from SystemCode.wellnessbot.rules.ruleset import rule_red_flags_escalate

    rr = rule_red_flags_escalate(nlu_full)
    if rr is not None and rr.action == Action.ESCALATE:
        protocol = get_protocol_for_surgery_type(nlu_full.surgery_type)
        protocol_id = protocol.protocol_id if protocol else None

        audit_context = {
            "user_text": user_text,
            "nlu_source": nlu_full.nlu_source,
            "surgery_type": nlu_full.surgery_type,
            "weeks_since_event": nlu_full.weeks_since_event,
            "requested_exercise_text": nlu_full.requested_exercise_text,
            "resolved_exercise_id": None,
            "exercise_name": None,
            "protocol_id": protocol_id,
            "phase_id": state.phase_id,
            "exercise_allowed_phases": None,
            "equipment_available": conv.equipment_available,
            "exercise_history_size": len(planner_history),
            "exercise_blocked": conv.exercise_blocked,
            "block_reason": conv.block_reason,
            "pending_followup_slots": conv.pending_followup_slots,
        }

        audit_trace = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "final",
            "audit_context": audit_context,
            "state": {
                "phase_id": state.phase_id,
                "risk_flags": state.risk_flags,
                "missing_fields": state.missing_fields,
            },
            "notes": ["Early escalation due to red flag before planning."],
        }

        result = {
            "mode": "final",
            "user_text": user_text,
            "nlu": nlu_full.model_dump(),
            "decision": {
                "action": rr.action.value,
                "confidence": 0.9,
                "rule_ids": [rr.rule_id],
                "citations": rr.citations,
            },
            "audit_trace": {
                **audit_trace,
                "rules_fired": [
                    {
                        "rule_id": rr.rule_id,
                        "action": rr.action.value,
                        "rationale": rr.rationale,
                        "citations": rr.citations,
                        "confidence_delta": rr.confidence_delta,
                    }
                ],
                "planner": None,
            },
            "conv_state": conv.to_dict(),
        }

        result["interaction_id"] = _make_interaction_id(
            user_text,
            audit_trace["timestamp_utc"],
        )

        log_interaction(result)

        return result

    protocol = get_protocol_for_surgery_type(nlu_full.surgery_type)
    protocol_id = protocol.protocol_id if protocol else None

    resolved_ex_id = resolve_exercise_id(
        nlu_full.requested_exercise_text,
        nlu_full.surgery_type,
    )

    ex = get_exercise(resolved_ex_id, nlu_full.surgery_type) if resolved_ex_id else None

    audit_context = {
        "user_text": user_text,
        "nlu_source": nlu_full.nlu_source,
        "surgery_type": nlu_full.surgery_type,
        "weeks_since_event": nlu_full.weeks_since_event,
        "requested_exercise_text": nlu_full.requested_exercise_text,
        "resolved_exercise_id": resolved_ex_id,
        "exercise_name": ex.name if ex else None,
        "protocol_id": protocol_id,
        "phase_id": state.phase_id,
        "exercise_allowed_phases": ex.allowed_phases if ex else None,
        "equipment_available": conv.equipment_available,
        "exercise_history_size": len(planner_history),
        "exercise_blocked": conv.exercise_blocked,
        "block_reason": conv.block_reason,
        "pending_followup_slots": conv.pending_followup_slots,
    }

    final_action, fired_rules = evaluate_rules(nlu_full)
    selfcare_routine = _extract_selfcare_routine(fired_rules)

    planner_out = None

    if final_action == Action.RECOMMEND:
        if state.phase_id:
            planner_out = plan(
                nlu_full,
                state.phase_id,
                equipment_available=conv.equipment_available,
                exercise_history=planner_history,
                selfcare_routine=selfcare_routine,
            )
        else:
            planner_out = {
                "plan": None,
                "notes": ["No phase available for planning."],
                "selfcare_routine": selfcare_routine,
            }

    elif final_action == Action.CLARIFY:
        if state.phase_id:
            planner_out = plan_alternatives(
                nlu_full.surgery_type,
                state.phase_id,
                top_k=5,
            )

    elif final_action in (Action.FORBID, Action.ESCALATE):
        planner_out = None

    conf = 0.5
    for r in fired_rules:
        conf += r.confidence_delta
    conf = max(0.0, min(1.0, conf))

    decision = {
        "action": final_action.value,
        "confidence": conf,
        "rule_ids": [
            r.rule_id for r in fired_rules if r.action.value == final_action.value
        ] or [r.rule_id for r in fired_rules],
        "citations": sorted({c for r in fired_rules for c in r.citations}),
    }

    audit_trace = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "final",
        "audit_context": audit_context,
        "state": {
            "phase_id": state.phase_id,
            "risk_flags": state.risk_flags,
            "missing_fields": state.missing_fields,
        },
        "rules_fired": [
            {
                "rule_id": r.rule_id,
                "action": r.action.value,
                "rationale": r.rationale,
                "citations": r.citations,
                "confidence_delta": r.confidence_delta,
            }
            for r in fired_rules
        ],
        "planner": planner_out,
        "notes": [
            "Decision brain = rules + constraints + planner. NLU provides structured fields only.",
        ],
    }

    result = {
        "mode": "final",
        "user_text": user_text,
        "nlu": nlu_full.model_dump(),
        "decision": decision,
        "audit_trace": audit_trace,
        "conv_state": conv.to_dict(),
    }

    result["interaction_id"] = _make_interaction_id(
        user_text,
        audit_trace["timestamp_utc"],
    )

    log_interaction(result)

    return result