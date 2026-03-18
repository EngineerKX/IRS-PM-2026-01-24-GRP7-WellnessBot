from __future__ import annotations

from typing import Optional

from wellnessbot.kg.kg import (
    get_exercise,
    get_redflag_policies,
    get_selfcare_actions,
    phase_from_weeks,
    resolve_exercise_id,
)
from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.rules.rule_types import Action, RuleResult


# Dominance: ESCALATE > FORBID > SUPPORTIVE CARE > CLARIFY > RECOMMEND
DOMINANCE = {
    Action.ESCALATE: 5,
    Action.FORBID: 4,
    Action.SUPPORTIVE_CARE: 3,
    Action.CLARIFY: 2,
    Action.RECOMMEND: 1,
}


def _current_phase(nlu: NLUOutput) -> Optional[str]:
    if nlu.weeks_since_event is None:
        return None
    return phase_from_weeks(nlu.weeks_since_event, nlu.event_type)


def _match_redflag_policy(nlu: NLUOutput):
    phase_id = _current_phase(nlu)
    if not phase_id:
        return None

    policies = get_redflag_policies(nlu.event_type, phase_id)

    # fever
    if "fever" in (nlu.red_flag_terms or []):
        for p in policies:
            if p.symptom == "fever":
                return p

    # excessive bleeding / wound drainage
    if any(t in (nlu.red_flag_terms or []) for t in ["wound drainage", "pus"]):
        for p in policies:
            if p.symptom == "excessive_bleeding_or_wound_drainage":
                return p

    # pain severity
    if nlu.pain_score is not None:
        for p in policies:
            if p.symptom == "pain" and str(nlu.pain_score) == str(p.severity):
                return p

    # swelling severity
    if nlu.swelling_level in {"severe"}:
        # current schema is text-based, map severe -> 3
        for p in policies:
            if p.symptom == "swelling" and str(p.severity) == "3":
                return p

    return None


def _match_selfcare_actions(nlu: NLUOutput):
    phase_id = _current_phase(nlu)
    if not phase_id:
        return []

    actions = get_selfcare_actions(nlu.event_type, phase_id)

    matched = []

    # swelling mapping
    swelling_map = {
        "none": "none",
        "mild": "1",
        "moderate": "2",
        "severe": "3",
        "unknown": "unknown",
    }
    swell_level = swelling_map.get(nlu.swelling_level, "unknown")

    for a in actions:
        if a.swell_level == "any":
            matched.append(a)
        elif str(a.swell_level).lower() == str(swell_level).lower():
            matched.append(a)

    return matched


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
    policy = _match_redflag_policy(nlu)
    if not policy:
        return None

    if policy.action == "escalate":
        return RuleResult(
            action=Action.ESCALATE,
            rule_id=f"R_ESCALATE_{policy.redflag_id}",
            rationale=policy.message or "Red-flag symptoms detected. Escalate to clinician.",
            citations=["SRC_RULEBOOK_001#redflag_policy"],
            confidence_delta=-0.6,
        )

    # supportive sequence is kept as FORBID for now, so planner can still show alternatives
    if policy.action == "supportive_sequence":
        steps = ", ".join(policy.action_steps or [])
        return RuleResult(
            action=Action.SUPPORTIVE_CARE,
            rule_id=f"R_SUPPORTIVE_{policy.redflag_id}",
            rationale=f"Supportive care required before exercise: {steps}.",
            citations=["SRC_RULEBOOK_001#supportive_sequence"],
            confidence_delta=-0.3,
        )

    return None


def rule_selfcare_guidance(nlu: NLUOutput) -> Optional[RuleResult]:
    # only provide self-care if symptom information is actually present
    has_symptom_signal = (
        bool(nlu.red_flag_terms)
        or nlu.pain_score is not None
        or (nlu.swelling_level or "unknown") != "unknown"
    )

    if not has_symptom_signal:
        return None

    actions = _match_selfcare_actions(nlu)
    if not actions:
        return None

    texts = []
    for a in actions:
        texts.append(
            f"{a.care_type} for {a.duration_minutes} minutes ({a.frequency_condition})"
        )

    return RuleResult(
        action=Action.RECOMMEND,
        rule_id="R_SELFCARE_GUIDANCE_001",
        rationale="Self-care guidance: " + "; ".join(texts),
        citations=["SRC_RULEBOOK_001#selfcare_actions"],
        confidence_delta=+0.05,
    )


def rule_unknown_exercise_clarify(nlu: NLUOutput) -> Optional[RuleResult]:
    if not (nlu.requested_exercise_text or "").strip():
        return None

    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    if ex_id is None:
        return RuleResult(
            action=Action.CLARIFY,
            rule_id="R_CLARIFY_EXERCISE_002",
            rationale="Requested exercise not recognized in current knowledge base.",
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


def rule_weight_bearing_gate(nlu: NLUOutput) -> Optional[RuleResult]:
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    if not ex_id:
        return None

    ex = get_exercise(ex_id, nlu.event_type)
    if not ex:
        return None

    if nlu.weight_bearing != "unknown" and nlu.weight_bearing not in ex.weight_bearing_allowed:
        return RuleResult(
            action=Action.FORBID,
            rule_id="R_FORBID_WEIGHT_BEARING_001",
            rationale=f"Weight bearing '{nlu.weight_bearing}' not allowed for '{ex.name}'.",
            citations=ex.source_refs + ["SRC_RULEBOOK_001#weight_bearing_gate"],
            confidence_delta=-0.2,
        )
    return None


def rule_recommend_if_all_clear(nlu: NLUOutput) -> Optional[RuleResult]:
    if nlu.weeks_since_event is None:
        return None

    if not (nlu.requested_exercise_text or "").strip():
        return RuleResult(
            action=Action.RECOMMEND,
            rule_id="R_RECOMMEND_PHASE_001",
            rationale="Core recovery information is available. Recommend a suitable exercise from the current phase.",
            citations=["SRC_RULEBOOK_001#recommend_when_clear"],
            confidence_delta=+0.2,
        )

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
    req = (nlu.requested_exercise_text or "").strip().lower()
    if req and req in loaded and nlu.event_type == "unknown":
        return RuleResult(
            action=Action.CLARIFY,
            rule_id="R_CLARIFY_EVENT_001",
            rationale="To check safety for loaded exercises, I need to know the event type.",
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
    rule_weight_bearing_gate,
    rule_selfcare_guidance,
    rule_recommend_if_all_clear,
    rule_clarify_event_type_for_loaded,
]