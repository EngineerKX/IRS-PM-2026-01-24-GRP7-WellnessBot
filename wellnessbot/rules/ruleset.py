from __future__ import annotations

from typing import Dict, List, Optional

from wellnessbot.kg.kg import get_exercise, phase_from_weeks, resolve_exercise_id
from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.rules.rule_types import Action, RuleResult


# Dominance: ESCALATE > FORBID > CLARIFY > RECOMMEND
DOMINANCE = {
    Action.ESCALATE: 4,
    Action.FORBID: 3,
    Action.CLARIFY: 2,
    Action.RECOMMEND: 1,
}

# Red flags for prototype
RED_FLAG_DOMINANT = {"fever", "locking", "cannot bear weight", "chest pain", "shortness of breath", "wound drainage", "pus"}


def rule_missing_weeks(nlu: NLUOutput) -> Optional[RuleResult]:
    if nlu.weeks_since_event is None:
        return RuleResult(
            action=Action.CLARIFY,
            rule_id="R_CLARIFY_WEEKS_001",
            rationale="Need time since event (weeks/days) to determine rehab phase safely.",
            citations=["SRC_RULEBOOK_001#weeks_required"],
            confidence_delta=-0.2,
        )
    return None


def rule_red_flags_escalate(nlu: NLUOutput) -> Optional[RuleResult]:
    hits = [t for t in nlu.red_flag_terms if t in RED_FLAG_DOMINANT]
    if hits:
        return RuleResult(
            action=Action.ESCALATE,
            rule_id="R_ESCALATE_REDFLAG_001",
            rationale=f"Red-flag symptoms present ({', '.join(hits)}). Escalate to clinician.",
            citations=["SRC_PDF_999#red_flags"],
            confidence_delta=-0.6,
        )
    return None


def rule_unknown_exercise_clarify(nlu: NLUOutput) -> Optional[RuleResult]:
    if not nlu.requested_exercise_text:
        return RuleResult(
            action=Action.CLARIFY,
            rule_id="R_CLARIFY_EXERCISE_001",
            rationale="Need the requested exercise name to check phase constraints.",
            citations=["SRC_RULEBOOK_001#exercise_required"],
            confidence_delta=-0.2,
        )
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    if ex_id is None:
        return RuleResult(
            action=Action.CLARIFY,
            rule_id="R_CLARIFY_EXERCISE_002",
            rationale="Requested exercise not recognized in current knowledge base. Ask user to rephrase or pick from known list.",
            citations=["SRC_KG_001#exercise_aliases"],
            confidence_delta=-0.1,
        )
    return None


def rule_phase_forbid(nlu: NLUOutput) -> Optional[RuleResult]:
    if nlu.weeks_since_event is None:
        return None

    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    if not ex_id:
        return None

    phase_id = phase_from_weeks(nlu.weeks_since_event, nlu.event_type)
    ex = get_exercise(ex_id, nlu.event_type)
    if ex and phase_id not in ex.allowed_phases:
        return RuleResult(
            action=Action.FORBID,
            rule_id="R_FORBID_PHASE_001",
            rationale=f"Exercise '{ex.name}' is not allowed in phase {phase_id}.",
            citations=ex.source_refs + ["SRC_RULEBOOK_001#phase_constraints"],
            confidence_delta=-0.3,
        )
    return None


def rule_pain_gate(nlu: NLUOutput) -> Optional[RuleResult]:
    if nlu.weeks_since_event is None:
        return None
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    if not ex_id:
        return None
    ex = get_exercise(ex_id, nlu.event_type)
    if not ex:
        return None

    if nlu.pain_score is not None and nlu.pain_score > ex.min_pain_max:
        return RuleResult(
            action=Action.FORBID,
            rule_id="R_FORBID_PAIN_001",
            rationale=f"Pain score {nlu.pain_score}/10 exceeds allowed threshold for '{ex.name}'.",
            citations=ex.source_refs + ["SRC_RULEBOOK_001#pain_gate"],
            confidence_delta=-0.2,
        )
    return None


def rule_swelling_gate(nlu: NLUOutput) -> Optional[RuleResult]:
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    if not ex_id:
        return None
    ex = get_exercise(ex_id, nlu.event_type)
    if not ex:
        return None

    if nlu.swelling_level != "unknown" and nlu.swelling_level not in ex.swelling_allowed:
        return RuleResult(
            action=Action.FORBID,
            rule_id="R_FORBID_SWELLING_001",
            rationale=f"Swelling level '{nlu.swelling_level}' not allowed for '{ex.name}'.",
            citations=ex.source_refs + ["SRC_RULEBOOK_001#swelling_gate"],
            confidence_delta=-0.2,
        )
    return None


def rule_recommend_if_all_clear(nlu: NLUOutput) -> Optional[RuleResult]:
    # This should fire only if nothing higher-risk blocks it; engine handles dominance.
    if nlu.weeks_since_event is None:
        return None
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    if not ex_id:
        return None
    ex = get_exercise(ex_id, nlu.event_type)
    if not ex:
        return None
    return RuleResult(
        action=Action.RECOMMEND,
        rule_id="R_RECOMMEND_CLEAR_001",
        rationale=f"No blocking constraints detected for '{ex.name}'.",
        citations=ex.source_refs + ["SRC_RULEBOOK_001#recommend_when_clear"],
        confidence_delta=+0.2,
    )

def rule_clarify_event_type_for_loaded(nlu: NLUOutput) -> Optional[RuleResult]:
    loaded = {"squats", "lunges", "leg press", "step ups", "wall sit"}
    if (nlu.requested_exercise_text or "").strip().lower() in loaded:
        if nlu.event_type == "unknown":
            return RuleResult(
                action=Action.CLARIFY,
                rule_id="R_CLARIFY_EVENT_001",
                rationale="To check safety for loaded exercises, I need to know the event type (e.g., ACL surgery vs TKR vs meniscus).",
                citations=["SRC_RULEBOOK_001#event_type_required_for_loaded"],
                confidence_delta=-0.1,
            )
    return None


RULES = [
    rule_red_flags_escalate,
    rule_missing_weeks,
    rule_unknown_exercise_clarify,
    rule_phase_forbid,
    rule_pain_gate,
    rule_swelling_gate,
    rule_recommend_if_all_clear,
    rule_clarify_event_type_for_loaded,
]