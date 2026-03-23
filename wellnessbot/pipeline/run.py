from __future__ import annotations

import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict

from wellnessbot.kg.kg import get_exercise, get_protocol_for_event, resolve_exercise_id
from wellnessbot.nlu.mock_extractor import extract_mock
from wellnessbot.nlu.openai_extractor import extract_with_fallback
from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.planner.planner import plan, plan_alternatives
from wellnessbot.rules.engine import evaluate_rules
from wellnessbot.rules.rule_types import Action
from wellnessbot.state.infer import infer_state
from wellnessbot.logging.logger import log_interaction

# NEW: dialog imports
from wellnessbot.dialog.state import ConversationState
from wellnessbot.dialog.merge import merge_turn
from wellnessbot.dialog.policy import compute_missing_slots, next_question_for_missing


def _make_interaction_id(user_text: str, ts: str) -> str:
    return hashlib.sha256(f"{ts}|{user_text}".encode("utf-8")).hexdigest()[:16]


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
    3) If missing slots → CLARIFY mode
    4) Else → run deterministic decision engine
    """

    # --------------------------------------------
    # Conversation memory (load existing state first)
    # --------------------------------------------
    conv = ConversationState.from_dict(conv_state or {})

    # Figure out what slot the system was expecting BEFORE this turn
    # Peek only; do not mutate asked_slots here.
    prior_missing_slots = compute_missing_slots(conv)
    expected_slot = prior_missing_slots[0] if prior_missing_slots else None

    print("DEBUG expected_slot BEFORE extraction:", expected_slot)
    print("DEBUG user_text BEFORE extraction:", repr(user_text))

    # --------------------------------------------
    # Decide NLU mode
    # --------------------------------------------
    mock_env = os.getenv("MOCK_NLU", "0").strip() == "1"
    use_mock = force_mock_nlu or mock_env
    print("DEBUG PRIOR MISSING SLOTS:", prior_missing_slots)
    print("DEBUG EXPECTED SLOT CHOSEN:", expected_slot)
    conv.last_expected_slot = expected_slot or ""

    if use_mock:
        nlu_turn: NLUOutput = extract_mock(user_text)
    else:
        nlu_turn = extract_with_fallback(
            user_text=user_text,
            expected_slot=expected_slot,
            timeout_s=12.0,
        )

    # --------------------------------------------
    # Merge current turn into conversation state
    # --------------------------------------------
    conv = merge_turn(conv, nlu_turn, user_text, expected_slot=expected_slot)

    # --------------------------------------------
    # EARLY RED FLAG CHECK (before clarify)
    # --------------------------------------------
    from wellnessbot.rules.ruleset import rule_red_flags_escalate

    rr = rule_red_flags_escalate(nlu_turn)

    if rr is not None:
        audit_trace = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "final",
            "notes": ["Early escalation due to red flag before clarification."],
        }

        result = {
            "mode": "final",
            "user_text": user_text,
            "nlu": nlu_turn.model_dump(),
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
            },
            "conv_state": conv.to_dict(),
        }

        return result

    # --------------------------------------------
    # CLARIFY MODE (deterministic hard gate)
    # --------------------------------------------
    print("DEBUG STATE BEFORE POLICY:", conv.to_dict())
    print("DEBUG last_expected_slot BEFORE POLICY:", getattr(conv, "last_expected_slot", ""))
    print("DEBUG symptom_screen_done BEFORE POLICY:", conv.symptom_screen_done)
    missing_slots = compute_missing_slots(conv)
    print("DEBUG MISSING SLOTS AFTER POLICY:", missing_slots)
    print("DEBUG symptom_screen_done AFTER POLICY:", conv.symptom_screen_done)


    if missing_slots:
        next_q = next_question_for_missing(conv, missing_slots)

        audit_trace = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "clarify",
            "missing_slots": missing_slots,
            "asked_slot": next_q["slot_name"],
            "notes": ["Clarify is system-driven (not LLM)."],
        }

        result = {
            "mode": "clarify",
            "question": next_q["question"],
            "slot_name": next_q["slot_name"],
            "conv_state": conv.to_dict(),
            "nlu_turn": nlu_turn.model_dump(),
            "audit_trace": audit_trace,
            "user_text": user_text,
        }

        return result

    # --------------------------------------------
    # FINAL MODE (build full NLU from conv state)
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
            "event_type": conv.event_type,
            "requested_exercise_text": conv.requested_exercise_text,
            "pain_score": conv.pain_score,
            "swelling_level": conv.swelling_level,
            "weight_bearing": conv.weight_bearing,
            "red_flag_terms": conv.red_flag_terms,
            "negated_terms": nlu_turn.negated_terms,
            "missing_fields": [],
            "nlu_source": nlu_turn.nlu_source,
        }
    )

    # --------------------------------------------
    # State inference
    # --------------------------------------------
    state = infer_state(nlu_full)

    # --------------------------------------------
    # KG snapshot (audit)
    # --------------------------------------------
    protocol = get_protocol_for_event(nlu_full.event_type)
    protocol_id = protocol.protocol_id if protocol else None

    resolved_ex_id = resolve_exercise_id(
        nlu_full.requested_exercise_text, nlu_full.event_type
    )

    ex = get_exercise(resolved_ex_id, nlu_full.event_type) if resolved_ex_id else None

    audit_context = {
        "user_text": user_text,
        "nlu_source": nlu_full.nlu_source,
        "event_type": nlu_full.event_type,
        "weeks_since_event": nlu_full.weeks_since_event,
        "requested_exercise_text": nlu_full.requested_exercise_text,
        "resolved_exercise_id": resolved_ex_id,
        "exercise_name": ex.name if ex else None,
        "protocol_id": protocol_id,
        "phase_id": state.phase_id,
        "exercise_allowed_phases": ex.allowed_phases if ex else None,
        "equipment_available": conv.equipment_available,
        "exercise_history_size": len(conv.exercise_history),
    }

    # --------------------------------------------
    # Rule engine
    # --------------------------------------------
    final_action, fired_rules = evaluate_rules(nlu_full)

    # --------------------------------------------
    # Planner
    # --------------------------------------------
    planner_out = None

    if final_action == Action.RECOMMEND:
        if state.phase_id:
            planner_out = plan(
                nlu_full,
                state.phase_id,
                equipment_available=conv.equipment_available,
                exercise_history=conv.exercise_history,
            )
        else:
            planner_out = {"plan": None, "notes": ["No phase available for planning."]}

    elif final_action in (Action.FORBID, Action.CLARIFY):
        if state.phase_id:
            planner_out = plan_alternatives(
                nlu_full.event_type, state.phase_id, top_k=5
            )

    elif final_action == Action.SUPPORTIVE_CARE:
        supportive_text = " ".join(r.rationale.lower() for r in fired_rules)

        # If the rule says to stop exercise, do not show alternatives yet
        if "stop_exercise" in supportive_text:
            planner_out = None

        # If the rule says downgrade exercise, show safer options
        elif "exercise_downgrade" in supportive_text:
            if state.phase_id:
                planner_out = plan_alternatives(
                    nlu_full.event_type, state.phase_id, top_k=5
                )
    
    # --------------------------------------------
    # Update exercise history after recommendation
    # --------------------------------------------
    if final_action == Action.RECOMMEND and planner_out and planner_out.get("exercise_id"):
        conv.exercise_history.append(
            {
                "exercise_id": planner_out["exercise_id"],
                "turn": conv.turn_count,
            }
        )

        # keep only recent history to avoid unbounded growth
        conv.exercise_history = conv.exercise_history[-5:]

    # --------------------------------------------
    # Confidence (prototype)
    # --------------------------------------------
    conf = 0.5
    for r in fired_rules:
        conf += r.confidence_delta
    conf = max(0.0, min(1.0, conf))

    decision = {
        "action": final_action.value,
        "confidence": conf,
        "rule_ids": [
            r.rule_id for r in fired_rules if r.action.value == final_action.value
        ]
        or [r.rule_id for r in fired_rules],
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
        user_text, audit_trace["timestamp_utc"]
    )

    log_interaction(result)

    return result