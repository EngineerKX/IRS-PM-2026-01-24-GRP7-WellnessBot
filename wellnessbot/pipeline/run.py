from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict

from wellnessbot.kg.kg import get_exercise, get_protocol_for_event, resolve_exercise_id
from wellnessbot.nlu.mock_extractor import extract_mock
from wellnessbot.nlu.openai_extractor import extract_with_fallback
from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.planner.planner import plan
from wellnessbot.rules.engine import evaluate_rules
from wellnessbot.rules.rule_types import Action
from wellnessbot.state.infer import infer_state

from wellnessbot.logging.logger import log_interaction


def run_pipeline(user_text: str, force_mock_nlu: bool = False) -> Dict[str, Any]:
    """
    Runtime path (must follow):
    User Text
    -> NLU Extract (mock/OpenAI, structured output only)
    -> State Inference (phase + risk flags + missing info)
    -> KG Lookup (exercise constraints by phase)
    -> Rule Engine (RECOMMEND / FORBID / CLARIFY / ESCALATE)
    -> Planner (only if RECOMMEND)
    -> DECISION FINALIZED (with rule_ids, citations, confidence)
    -> Response Layer (explain + cite only; no decision changes)
    """

    # Debug prints only when explicitly enabled
    if os.getenv("DEBUG_PIPELINE", "0") == "1":
        print("OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))
        print("MOCK_NLU:", os.getenv("MOCK_NLU"))
        print("force_mock_nlu:", force_mock_nlu)

    # Default is OpenAI unless MOCK_NLU=1 or UI forces mock
    mock_env = os.getenv("MOCK_NLU", "0").strip() == "1"
    use_mock = force_mock_nlu or mock_env

    # --- NLU ---
    if use_mock:
        nlu: NLUOutput = extract_mock(user_text)
    else:
        nlu = extract_with_fallback(user_text=user_text, timeout_s=12.0)

    # --- State inference (no re-parsing text) ---
    state = infer_state(nlu)

    # --- KG context snapshot (audit) ---
    protocol = get_protocol_for_event(nlu.event_type)
    protocol_id = protocol.protocol_id if protocol else None

    resolved_ex_id = None
    exercise_name = None
    allowed_phases = None

    if (nlu.requested_exercise_text or "").strip():
        resolved_ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
        if resolved_ex_id:
            ex = get_exercise(resolved_ex_id, nlu.event_type)
            if ex:
                exercise_name = ex.name
                allowed_phases = ex.allowed_phases

    audit_context = {
        "user_text": user_text,
        "nlu_source": nlu.nlu_source,
        "event_type": nlu.event_type,
        "weeks_since_event": nlu.weeks_since_event,
        "requested_exercise_text": nlu.requested_exercise_text,
        "resolved_exercise_id": resolved_ex_id,
        "exercise_name": exercise_name,
        "protocol_id": protocol_id,
        "phase_id": state.phase_id,
        "exercise_allowed_phases": allowed_phases,
    }

    # --- Rules ---
    final_action, fired_rules = evaluate_rules(nlu)

    # --- Planner (only if RECOMMEND) ---
    planner_out = None
    if final_action == Action.RECOMMEND:
        planner_out = plan(nlu)
    elif final_action in (Action.FORBID, Action.CLARIFY):
        # Only suggest alternatives if we know phase_id
        if state.phase_id:
            from wellnessbot.planner.planner import plan_alternatives
            planner_out = plan_alternatives(nlu.event_type, state.phase_id, top_k=5)

    # --- Confidence (prototype bounded aggregation) ---
    conf = 0.5
    for r in fired_rules:
        conf += r.confidence_delta
    conf = max(0.0, min(1.0, conf))

    decision = {
        "action": final_action.value,
        "confidence": conf,
        "rule_ids": [r.rule_id for r in fired_rules if r.action.value == final_action.value]
        or [r.rule_id for r in fired_rules],
        "citations": sorted({c for r in fired_rules for c in r.citations}),
    }

    audit_trace = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
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
            "Citations are placeholders (PDF IDs/pages) until you wire real PDFs.",
        ],
    }

    result = {
        "user_text": user_text,
        "nlu": nlu.model_dump(),
        "decision": decision,
        "audit_trace": audit_trace,
    }

    log_interaction(result)

    return result