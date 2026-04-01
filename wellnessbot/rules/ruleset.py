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


# Dominance: ESCALATE > FORBID > CLARIFY > RECOMMEND
DOMINANCE = {
    Action.ESCALATE: 4,
    Action.FORBID: 3,
    Action.CLARIFY: 2,
    Action.RECOMMEND: 1,
}


def _current_phase(nlu: NLUOutput) -> Optional[str]:
    if nlu.weeks_since_event is None:
        return None
    return phase_from_weeks(nlu.weeks_since_event, nlu.surgery_type)


def _has_non_recommend_blockers(nlu: NLUOutput) -> bool:
    """
    Internal helper for recommend rules only.
    Re-checks blocking conditions so recommend rules do not leak when
    FORBID / ESCALATE / CLARIFY should dominate.
    """
    blockers = [
        rule_red_flags_escalate(nlu),
        rule_missing_weeks(nlu),
        rule_unknown_exercise_clarify(nlu),
        rule_phase_forbid(nlu),
        rule_pain_gate(nlu),
        rule_swelling_gate(nlu),
        rule_weight_bearing_gate(nlu),
        rule_clarify_surgery_type_for_loaded(nlu),
    ]
    return any(rr is not None and rr.action != Action.RECOMMEND for rr in blockers)


def _match_redflag_policy(nlu: NLUOutput):
    phase_id = _current_phase(nlu)
    if not phase_id:
        return None

    policies = get_redflag_policies(nlu.surgery_type, phase_id)
    matches = []

    # pain severity
    if nlu.pain_score is not None:
        for p in policies:
            if p.symptom == "pain" and str(nlu.pain_score) == str(p.severity):
                matches.append(p)

    # swelling severity
    swelling_map = {
        "none": "0",
        "mild": "1",
        "moderate": "2",
        "severe": "3",
        "unknown": "unknown",
    }
    swell_value = swelling_map.get(nlu.swelling_level, "unknown")
    for p in policies:
        if p.symptom == "swelling" and str(swell_value) == str(p.severity):
            matches.append(p)

    # fever
    if "fever" in (nlu.red_flag_terms or []):
        for p in policies:
            if p.symptom == "fever":
                matches.append(p)

    # excessive bleeding / wound drainage
    if any(
        t in (nlu.red_flag_terms or [])
        for t in ["excessive bleeding", "wound drainage", "pus", "bleeding"]
    ):
        for p in policies:
            if p.symptom == "excessive_bleeding_or_wound_drainage":
                matches.append(p)

    if not matches:
        return None

    def policy_rank(p):
        if p.action == "escalate":
            return 2
        if p.action == "supportive_sequence":
            return 1
        return 0

    matches.sort(key=policy_rank, reverse=True)
    return matches[0]


def _match_selfcare_actions(nlu: NLUOutput):
    phase_id = _current_phase(nlu)
    if not phase_id:
        return []

    actions = get_selfcare_actions(nlu.surgery_type, phase_id)

    matched = []

    swelling_value = nlu.swelling_level

    if swelling_value in (None, "", "unknown"):
        symptom_flags = set(x.strip().lower() for x in (nlu.symptom_flags or []))

        if getattr(nlu, "symptom_screen_done", False) and "swelling" not in symptom_flags:
            swelling_value = "none"
        else:
            swelling_value = "unknown"

    swelling_map = {
        "none": "none",
        "mild": "1",
        "moderate": "2",
        "severe": "3",
        "unknown": "unknown",
    }
    swell_level = swelling_map.get(swelling_value, "unknown")

    for a in actions:
        if str(a.swell_level).lower() == "any":
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

    if policy.action == "supportive_sequence":
        steps = ", ".join(policy.action_steps or [])

        return RuleResult(
            action=Action.FORBID,
            rule_id=f"R_FORBID_SUPPORTIVE_{policy.redflag_id}",
            rationale=(
                "Do not proceed with exercise yet. "
                f"Supportive follow-up required first: {steps}."
            ),
            citations=["SRC_RULEBOOK_001#supportive_sequence"],
            confidence_delta=-0.3,
        )

    return None


def rule_selfcare_guidance(nlu: NLUOutput) -> Optional[RuleResult]:
    """
    Non-blocking supportive guidance.
    This may appear alongside FORBID, but should not appear with ESCALATE
    because engine stops on ESCALATE.
    """
    has_symptom_signal = (
        nlu.pain_score is not None
        or (nlu.swelling_level or "unknown") != "unknown"
    )

    if not has_symptom_signal:
        return None

    actions = _match_selfcare_actions(nlu)
    if not actions:
        return None

    phase_id = _current_phase(nlu)

    texts = []
    for a in actions:
        care_type = (a.care_type or "").strip().lower()

        if care_type in {"ice", "ice_elevate"}:
            texts.append(
                f"ice and elevate for {a.duration_minutes} minutes ({a.frequency_condition})"
            )
        elif care_type == "warm_hot_towel":
            texts.append(
                f"warm up with a hot towel for {a.duration_minutes} minutes ({a.frequency_condition})"
            )

        elif care_type == "warm_walk":
            texts.append(
                f"warm up by walking for {a.duration_minutes} minutes ({a.frequency_condition})"
            )

        elif care_type == "warm":
            texts.append(
                f"warm up for {a.duration_minutes} minutes ({a.frequency_condition})"
            )
        else:
            texts.append(
                f"{a.care_type} for {a.duration_minutes} minutes ({a.frequency_condition})"
            )

    if phase_id == "P1_1":
        rationale = "Early-phase supportive care: " + "; ".join(texts)
        rule_id = "R_SELFCARE_P1_SUPPORT_001"
        delta = +0.15
    else:
        rationale = "Supportive self-care guidance: " + "; ".join(texts)
        rule_id = "R_SELFCARE_GUIDANCE_001"
        delta = +0.05

    return RuleResult(
        action=Action.RECOMMEND,
        rule_id=rule_id,
        rationale=rationale,
        citations=["SRC_RULEBOOK_001#selfcare_actions"],
        confidence_delta=delta,
    )


def rule_unknown_exercise_clarify(nlu: NLUOutput) -> Optional[RuleResult]:
    if not (nlu.requested_exercise_text or "").strip():
        return None

    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.surgery_type)
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

    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.surgery_type)
    if not ex_id:
        return None

    phase_id = phase_from_weeks(nlu.weeks_since_event, nlu.surgery_type)
    ex = get_exercise(ex_id, nlu.surgery_type)
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

    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.surgery_type)
    if not ex_id:
        return None

    ex = get_exercise(ex_id, nlu.surgery_type)
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
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.surgery_type)
    if not ex_id:
        return None

    ex = get_exercise(ex_id, nlu.surgery_type)
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
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.surgery_type)
    if not ex_id:
        return None

    ex = get_exercise(ex_id, nlu.surgery_type)
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

    # Do not leak exercise recommendation when a blocker exists.
    if _has_non_recommend_blockers(nlu):
        return None

    if not (nlu.requested_exercise_text or "").strip():
        return RuleResult(
            action=Action.RECOMMEND,
            rule_id="R_RECOMMEND_PHASE_001",
            rationale="Core recovery information is available. Recommend a suitable exercise from the current phase.",
            citations=["SRC_RULEBOOK_001#recommend_when_clear"],
            confidence_delta=+0.2,
        )

    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.surgery_type)
    if not ex_id:
        return None

    ex = get_exercise(ex_id, nlu.surgery_type)
    if not ex:
        return None

    return RuleResult(
        action=Action.RECOMMEND,
        rule_id="R_RECOMMEND_CLEAR_001",
        rationale=f"No blocking constraints detected for '{ex.name}'.",
        citations=ex.source_refs + ["SRC_RULEBOOK_001#recommend_when_clear"],
        confidence_delta=+0.2,
    )


def rule_clarify_surgery_type_for_loaded(nlu: NLUOutput) -> Optional[RuleResult]:
    loaded = {"squats", "lunges", "leg press", "step ups", "wall sit"}
    req = (nlu.requested_exercise_text or "").strip().lower()
    if req and req in loaded and nlu.surgery_type == "unknown":
        return RuleResult(
            action=Action.CLARIFY,
            rule_id="R_CLARIFY_SURGERY_TYPE_001",
            rationale="To check safety for loaded exercises, I need to know the surgery type.",
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
    rule_clarify_surgery_type_for_loaded,
]