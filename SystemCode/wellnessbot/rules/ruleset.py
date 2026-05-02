from __future__ import annotations

from typing import Optional

from SystemCode.wellnessbot.kg.kg import (
    get_exercise,
    get_redflag_policies,
    get_selfcare_actions,
    phase_from_weeks,
    resolve_exercise_id,
)
from SystemCode.wellnessbot.nlu.schema import NLUOutput
from SystemCode.wellnessbot.rules.rule_types import Action, RuleResult
from SystemCode.wellnessbot.kg.kg import symptom_matches_policy
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


def _phase_major_number(phase_id: str | None) -> Optional[int]:
    if not phase_id:
        return None

    phase_id = str(phase_id).strip().upper()
    if not phase_id.startswith("P"):
        return None

    body = phase_id[1:]
    major = body.split("_")[0]

    try:
        return int(major)
    except ValueError:
        return None


def _is_return_to_sport_phase(nlu: NLUOutput) -> bool:
    phase_id = _current_phase(nlu)
    major = _phase_major_number(phase_id)
    return major is not None and major >= 5


def _has_non_recommend_blockers(nlu: NLUOutput) -> bool:
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

    # --- Pain-based matching ---
    if nlu.pain_score is not None:
        for p in policies:
            if p.symptom == "pain" and str(nlu.pain_score) == str(p.severity):
                matches.append(p)

    # --- Swelling-based matching ---
    if nlu.swelling_score is not None:
        for p in policies:
            if p.symptom == "swelling" and str(nlu.swelling_score) == str(p.severity):
                matches.append(p)

    # --- KG-driven red flag matching (CRITICAL FIX) ---
    red_flag_terms = list(nlu.red_flag_terms or [])

    for p in policies:
        symptom = (p.symptom or "").strip().lower()

        # skip already handled numeric symptoms
        if symptom in {"pain", "swelling"}:
            continue

        if symptom_matches_policy(nlu.surgery_type, symptom, red_flag_terms):
            matches.append(p)

    if not matches:
        return None

    # --- Ranking ---
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

    if nlu.swelling_score is None:
        symptom_flags = set(x.strip().lower() for x in (nlu.symptom_flags or []))
        if getattr(nlu, "symptom_screen_done", False) and "swelling" not in symptom_flags:
            swell_level = "0"
        else:
            swell_level = "unknown"
    else:
        swell_level = str(nlu.swelling_score)

    for a in actions:
        a_level = str(a.swell_level).lower()
        if a_level == "any":
            matched.append(a)
        elif a_level == swell_level.lower():
            matched.append(a)

    return matched


def _format_supportive_steps(steps: list[str] | None) -> str:
    if not steps:
        return "supportive follow-up"

    step_labels = {
        "prescribed_medicine_reminder": "follow prescribed medicine guidance",
        "pain_killer": "use pain relief as prescribed",
        "exercise_downgrade": "downgrade exercise intensity",
        "stop_exercise": "stop exercise for now",
        "pain_check": "monitor pain",
        "swell_check": "monitor swelling",
    }

    pretty = [step_labels.get(s, s.replace("_", " ")) for s in steps]
    return ", ".join(pretty)


def _supportive_sequence_is_blocking(policy) -> bool:
    steps = set(policy.action_steps or [])
    return "stop_exercise" in steps


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
            rationale=policy.message or "Exercise is not advised at this time. It is important to speak with a clinician before beginning any physical routine.",
            citations=[],
            confidence_delta=-0.6,
        )

    if policy.action == "supportive_sequence":
        steps_text = _format_supportive_steps(policy.action_steps or [])

        if _supportive_sequence_is_blocking(policy):
            return RuleResult(
                action=Action.FORBID,
                rule_id=f"R_FORBID_SUPPORTIVE_{policy.redflag_id}",
                rationale=(
                    "Do not proceed with exercise yet. "
                    f"Supportive follow-up required first: {steps_text}."
                ),
                citations=[],
                confidence_delta=-0.3,
            )

        return RuleResult(
            action=Action.RECOMMEND,
            rule_id=f"R_SUPPORTIVE_NONBLOCK_{policy.redflag_id}",
            rationale=(
                "Mild symptoms noted. Proceed conservatively with supportive follow-up: "
                f"{steps_text}."
            ),
            citations=[],
            confidence_delta=+0.02,
        )

    return None


def rule_selfcare_guidance(nlu: NLUOutput) -> Optional[RuleResult]:
    has_symptom_signal = (
        nlu.pain_score is not None
        or nlu.swelling_score is not None
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

        freq = (a.frequency_condition or "").replace("_", " ")
        if care_type in {"ice", "ice_elevate"}:
            texts.append(
                f"ice and elevate for {a.duration_minutes} minutes ({freq})"
            )
        elif care_type == "warm_hot_towel":
            texts.append(
                f"warm up with a hot towel for {a.duration_minutes} minutes ({freq})"
            )
        elif care_type == "warm_walk":
            texts.append(
                f"warm up by walking for {a.duration_minutes} minutes ({freq})"
            )
        elif care_type == "warm":
            texts.append(
                f"warm up for {a.duration_minutes} minutes ({freq})"
            )
        else:
            texts.append(
                f"{a.care_type} for {a.duration_minutes} minutes ({freq})"
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
        citations=[],
        confidence_delta=delta,
    )


def rule_return_to_sport(nlu: NLUOutput) -> Optional[RuleResult]:
    if nlu.weeks_since_event is None:
        return None

    if not _is_return_to_sport_phase(nlu):
        return None

    return RuleResult(
        action=Action.RECOMMEND,
        rule_id="R_RETURN_TO_SPORT_001",
        rationale=(
            "Congratulations, you are at the return-to-sport stage. "
            "You are encouraged to go for outdoor sports or sign up for a guided exercise package at a fitness centre."
        ),
        citations=[],
        confidence_delta=+0.3,
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
            rationale=f"Pain score {nlu.pain_score}/3 exceeds allowed threshold for '{ex.name}'.",
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

    if nlu.swelling_score is not None:
        score_to_label = {
            0: "none",
            1: "mild",
            2: "moderate",
            3: "severe",
        }
        swelling_label = score_to_label.get(nlu.swelling_score)

        if swelling_label is not None and swelling_label not in ex.swelling_allowed:
            return RuleResult(
                action=Action.FORBID,
                rule_id="R_FORBID_SWELLING_001",
                rationale=f"Swelling level {nlu.swelling_score}/3 not allowed for '{ex.name}'.",
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

    if _is_return_to_sport_phase(nlu):
        return None

    if _has_non_recommend_blockers(nlu):
        return None

    if not (nlu.requested_exercise_text or "").strip():
        return RuleResult(
            action=Action.RECOMMEND,
            rule_id="R_RECOMMEND_PHASE_001",
            rationale="Core recovery information is available. Recommend a suitable exercise from the current phase.",
            citations=[],
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
        citations=ex.source_refs + [],
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
    # --- ACTIVE RULES ---
    rule_red_flags_escalate,
    rule_return_to_sport,
    rule_selfcare_guidance,
    rule_recommend_if_all_clear,

    # --- DISABLED RULES (DEPRECATED / OUT OF CURRENT INTERACTION SCOPE) ---

    # rule_missing_weeks,
    # Reason: Responsibility moved to dialog manager (slot filling + next_question_for_missing)
    # Impact: Missing critical recovery fields are now handled earlier, instead of producing late-stage CLARIFY in the rule engine

    # rule_unknown_exercise_clarify,
    # Reason: Deprecated — user-specified exercise validation is no longer part of the current interaction design
    # Impact: The rule engine no longer generates CLARIFY decisions for unrecognized exercise requests

    # rule_phase_forbid,
    # Reason: Deprecated — phase-based exercise forbidding depended on explicit user exercise requests, which are no longer supported
    # Impact: The rule engine no longer performs FORBID checks on user-requested exercises

    # rule_pain_gate,
    # Reason: Deprecated — exercise-specific pain threshold checking depended on explicit user exercise requests, which are no longer supported
    # Impact: Pain is no longer evaluated against a user-requested exercise threshold in the rule engine

    # rule_swelling_gate,
    # Reason: Deprecated — exercise-specific swelling compatibility checking depended on explicit user exercise requests, which are no longer supported
    # Impact: Swelling is no longer checked against exercise-level allowance rules for user-requested exercises

    # rule_weight_bearing_gate,
    # Reason: Deprecated — exercise-specific weight-bearing compatibility checking depended on explicit user exercise requests, which are no longer supported
    # Impact: The rule engine no longer performs exercise-level weight-bearing FORBID checks on user-requested exercises

    # rule_clarify_surgery_type_for_loaded,
    # Reason: Deprecated — this rule existed only for safety checking of user-requested loaded exercises, which are no longer part of the interaction flow
    # Impact: The rule engine no longer asks for surgery type specifically to validate loaded exercise requests
]