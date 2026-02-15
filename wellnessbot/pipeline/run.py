from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Dict, Any

from wellnessbot.nlu.mock_extractor import extract_mock
from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.state.infer import infer_state
from wellnessbot.rules.engine import evaluate_rules
from wellnessbot.rules.rule_types import Action
from wellnessbot.planner.planner import plan

from wellnessbot.nlu.openai_extractor import extract_with_fallback



def run_pipeline(user_text: str, force_mock_nlu: bool = False) -> Dict[str, Any]:
    """
    Runtime path (must follow):
    User Text -> NLU -> State -> KG -> Rules -> Planner(if RECOMMEND) -> Final decision -> Response layer
    Note: RAG not in this round.
    """
    print("OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))
    print("MOCK_NLU:", os.getenv("MOCK_NLU"))
    print("force_mock_nlu:", force_mock_nlu)

    mock_env = os.getenv("MOCK_NLU", "0").strip() == "1"
    use_mock = force_mock_nlu or mock_env

    if use_mock:
        nlu: NLUOutput = extract_mock(user_text)
    else:
        nlu = extract_with_fallback(user_text=user_text, timeout_s=12.0)


    # Early clarify if key fields missing (weeks is mandatory for safe phase)
    state = infer_state(nlu)
    final_action, fired_rules = evaluate_rules(nlu)

    planner_out = None
    if final_action == Action.RECOMMEND:
        planner_out = plan(nlu)

    # Confidence: simple bounded aggregation for prototype
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

    return {
        "user_text": user_text,
        "nlu": nlu.model_dump(),
        "decision": decision,
        "audit_trace": audit_trace,
    }