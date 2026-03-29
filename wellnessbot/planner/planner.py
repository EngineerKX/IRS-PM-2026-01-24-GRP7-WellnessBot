"""
Intervention Planner --A* Informed Search for Exercise Selection

This module implements the Intervention Planner component for the WellnessBot
pipeline. It sits between the Rule & Inference Engine (upstream) and
RAG / Evidence Retrieval (downstream).

The Planner receives:
  1. A structured UserState dict (days_post_op, pain, swelling, etc.)
  2. A pre-filtered list of candidate Exercise dicts from the Rule Engine

It does NOT query the Knowledge Graph, implement rule logic, generate
user-facing text, or track multi-turn session state.

Design rationale:
  - A* informed search was chosen over ML ranking: zero labelled training data
    at cold-start; black-box decisions are unacceptable in a clinical domain.
  - Hard constraints execute before scoring --this is a structural safety
    requirement, not an optimisation preference.
  - g(n) and h(n) are kept separate in the audit log --required for
    explainability under EU AI Act for high-risk clinical AI.
  - effort_level is derived from E1–E5 position within phase (E1=0.2 easiest,
    E5=1.0 hardest) --conservative selection bias is built into h(n) via
    effort_inverse.
  - The audit log is the data foundation for future ML-based exercise ranking
    once interaction logs accumulate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWELLING_RANK: Dict[str, int] = {
    "none": 0,
    "mild": 1,
    "moderate": 2,
    "severe": 3,
}
"""Ordinal mapping for swelling severity. Used in both hard-constraint
pruning (ordinal comparison) and continuous scoring (normalised ratio)."""


# ---------------------------------------------------------------------------
# Step 1 --Hard Constraint Pruning
# ---------------------------------------------------------------------------

def _check_phase_eligibility(user_state: Dict, exercise: Dict) -> Optional[str]:
    """Return an elimination reason string if the user's rehab phase is not in
    the exercise's allowed_phases list, otherwise None.

    Phase eligibility is the most fundamental safety gate --an exercise
    designed for a later phase may impose loads the healing tissue cannot
    yet tolerate.
    """
    if user_state["rehab_phase"] not in exercise["allowed_phases"]:
        return (
            f"Phase mismatch: user is in {user_state['rehab_phase']}, "
            f"but exercise only allowed in {exercise['allowed_phases']}"
        )
    return None


def _check_pain_ceiling(user_state: Dict, exercise: Dict) -> Optional[str]:
    """Return an elimination reason if the user's current pain exceeds the
    exercise's maximum tolerable pain threshold, otherwise None.

    Pain ceiling is a hard gate because exercising beyond pain_max risks
    tissue damage and patient distress.
    """
    pain_max = exercise["limits"]["pain_max"]
    if user_state["pain_level"] > pain_max:
        return (
            f"Pain too high: user pain {user_state['pain_level']}/10 "
            f"exceeds exercise limit {pain_max}/10"
        )
    return None


def _check_swelling_ceiling(user_state: Dict, exercise: Dict) -> Optional[str]:
    """Return an elimination reason if the user's swelling severity exceeds
    the exercise's maximum tolerable swelling level, otherwise None.

    Swelling indicates active inflammation --exercising through excessive
    swelling can worsen joint effusion and delay recovery.
    """
    user_rank = SWELLING_RANK[user_state["swelling_level"]]
    limit_rank = SWELLING_RANK[exercise["limits"]["swelling_max"]]
    if user_rank > limit_rank:
        return (
            f"Swelling too high: user has '{user_state['swelling_level']}' "
            f"(rank {user_rank}), exercise limit is "
            f"'{exercise['limits']['swelling_max']}' (rank {limit_rank})"
        )
    return None


def _check_weight_bearing(user_state: Dict, exercise: Dict) -> Optional[str]:
    """Return an elimination reason if the user's weight-bearing status does
    not meet the exercise's requirement, otherwise None.

    Weight-bearing rules:
      - Exercise requires "full"    → user must be "full"
      - Exercise requires "partial" → user must be "partial" or "full"
      - Exercise requires "none"    → always passes (no WB needed)

    This protects surgical repairs from premature loading.
    """
    required = exercise["limits"]["weight_bearing_required"]
    user_wb = user_state["weight_bearing_status"]

    if required == "full" and user_wb != "full":
        return (
            f"Weight-bearing insufficient: exercise requires 'full', "
            f"user is '{user_wb}'"
        )
    if required == "partial" and user_wb not in ("partial", "full"):
        return (
            f"Weight-bearing insufficient: exercise requires 'partial', "
            f"user is '{user_wb}'"
        )
    return None


def prune_candidates(
    user_state: Dict, candidates: List[Dict]
) -> Tuple[List[Dict], List[Dict]]:
    """Apply all four hard constraints to each candidate.

    Returns:
        (survivors, eliminated) where each eliminated entry is
        {"exercise_id", "name", "reason"}.

    Hard constraints run before any scoring --this is a structural safety
    requirement. A single constraint failure eliminates the candidate.
    """
    checks = [
        _check_phase_eligibility,
        _check_pain_ceiling,
        _check_swelling_ceiling,
        _check_weight_bearing,
    ]

    survivors: List[Dict] = []
    eliminated: List[Dict] = []

    for ex in candidates:
        reasons = []
        for check in checks:
            reason = check(user_state, ex)
            if reason:
                reasons.append(reason)

        if reasons:
            eliminated.append({
                "exercise_id": ex["exercise_id"],
                "name": ex["name"],
                "reason": "; ".join(reasons),
            })
        else:
            survivors.append(ex)

    return survivors, eliminated


# ---------------------------------------------------------------------------
# Step 2 --A* Informed Search Scoring
# ---------------------------------------------------------------------------

def compute_g(user_state: Dict, exercise: Dict) -> float:
    """Compute g(n): the clinical risk cost of this exercise for the user.

    Higher g(n) means more risk --the exercise pushes the user closer to
    their safety limits. Three components are averaged:

      pain_risk     --how close user pain is to the exercise pain ceiling
      swelling_risk --how close user swelling is to the exercise swelling ceiling
      wb_burden     --1.0 if exercise requires full WB but user cannot provide it
                      (should be 0.0 after hard-constraint pruning, but kept for
                      robustness and auditability)
    """
    pain_max = exercise["limits"]["pain_max"]
    pain_risk = user_state["pain_level"] / pain_max if pain_max > 0 else 0.0

    swelling_max_rank = SWELLING_RANK[exercise["limits"]["swelling_max"]]
    if swelling_max_rank == 0:
        swelling_risk = 0.0
    else:
        swelling_risk = (
            SWELLING_RANK[user_state["swelling_level"]] / swelling_max_rank
        )

    required_wb = exercise["limits"]["weight_bearing_required"]
    user_wb = user_state["weight_bearing_status"]
    if required_wb == "full" and user_wb in ("non_partial", "partial"):
        wb_burden = 1.0
    else:
        wb_burden = 0.0

    return mean([pain_risk, swelling_risk, wb_burden])


def compute_h(user_state: Dict, exercise: Dict) -> float:
    """Compute h(n): the estimated rehabilitation benefit of this exercise.

    Higher h(n) means more clinical value. Five components are averaged:

      phase_match            --1.0 if the user's phase is in allowed_phases
      pain_safety_margin     --how far below the pain ceiling the user sits
      swelling_safety_margin --how far below the swelling ceiling the user sits
      effort_inverse         --preference for lower-effort (more conservative)
                               exercises via 1.0 - effort_level
      priority_score         --physiotherapist-assigned clinical priority
                               normalised to 0–1 (priority 1 = best = 1.0)
    """
    phase_match = (
        1.0 if user_state["rehab_phase"] in exercise["allowed_phases"] else 0.0
    )

    pain_max = exercise["limits"]["pain_max"]
    pain_safety_margin = (
        (pain_max - user_state["pain_level"]) / pain_max if pain_max > 0 else 0.0
    )

    swelling_max_rank = SWELLING_RANK[exercise["limits"]["swelling_max"]]
    user_swelling_rank = SWELLING_RANK[user_state["swelling_level"]]
    if swelling_max_rank == 0:
        swelling_safety_margin = 0.0
    else:
        swelling_safety_margin = (
            (swelling_max_rank - user_swelling_rank)
            / max(swelling_max_rank, 1)
        )

    effort_inverse = 1.0 - exercise["effort_level"]

    priority_score = 1.0 - (exercise["priority"] - 1) / 9

    return mean([
        phase_match,
        pain_safety_margin,
        swelling_safety_margin,
        effort_inverse,
        priority_score,
    ])


def score_candidates(
    user_state: Dict, survivors: List[Dict]
) -> List[Dict]:
    """Score every surviving candidate and return a list of score records
    sorted by f(n) descending (best first).

    Each record contains exercise_id, name, g_n, h_n, and f_n for the
    audit log.
    """
    scored: List[Dict] = []
    for ex in survivors:
        g = compute_g(user_state, ex)
        h = compute_h(user_state, ex)
        scored.append({
            "exercise_id": ex["exercise_id"],
            "name": ex["name"],
            "g_n": round(g, 6),
            "h_n": round(h, 6),
            "f_n": round(g + h, 6),
            "_exercise": ex,  # carry full object for selection; stripped before output
        })

    scored.sort(key=lambda s: s["f_n"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Step 3 --Selection with Deterministic Tie-Breaking
# ---------------------------------------------------------------------------

def select_best(
    scored: List[Dict], candidates: List[Dict]
) -> Tuple[Dict, str]:
    """Select the single best exercise from scored candidates.

    Tie-breaking rules (applied in order when f(n) scores are equal):
      1. Lower effort_level wins (more conservative choice)
      2. Lower priority number wins (priority 1 = highest clinical priority)
      3. Earlier position in the original candidate list from the Rule Engine

    Returns:
        (best_score_record, selection_rationale)
    """
    if len(scored) == 1:
        return scored[0], "Only one candidate survived pruning."

    # Build position index from the original candidate order
    position_index = {
        ex["exercise_id"]: i for i, ex in enumerate(candidates)
    }

    best_f = scored[0]["f_n"]
    tied = [s for s in scored if s["f_n"] == best_f]

    if len(tied) == 1:
        return tied[0], "Highest f(n) score."

    # Tie-break 1: lower effort_level
    min_effort = min(t["_exercise"]["effort_level"] for t in tied)
    tied = [t for t in tied if t["_exercise"]["effort_level"] == min_effort]
    if len(tied) == 1:
        return tied[0], "Tie-break applied: lower effort_level."

    # Tie-break 2: lower priority number
    min_priority = min(t["_exercise"]["priority"] for t in tied)
    tied = [t for t in tied if t["_exercise"]["priority"] == min_priority]
    if len(tied) == 1:
        return tied[0], "Tie-break applied: lower priority number."

    # Tie-break 3: earlier position in original candidate list
    tied.sort(key=lambda t: position_index.get(t["exercise_id"], 999))
    return tied[0], "Tie-break applied: position order in candidate list."


# ---------------------------------------------------------------------------
# Step 4 --Audit Log
# ---------------------------------------------------------------------------

def _build_audit_log(
    user_state: Dict,
    candidates_received: int,
    eliminated: List[Dict],
    scored: List[Dict],
    selected_exercise_id: Optional[str],
    selection_rationale: str,
) -> Dict[str, Any]:
    """Build a complete structured audit log for this planner invocation.

    The audit log is the primary explainability artefact --it records every
    pruning decision, every score, and the final selection rationale so that
    clinicians and regulators can inspect any recommendation.
    """
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "user_state": user_state,
        "candidates_received": candidates_received,
        "eliminated": eliminated,
        "scored": [
            {
                "exercise_id": s["exercise_id"],
                "name": s["name"],
                "g_n": s["g_n"],
                "h_n": s["h_n"],
                "f_n": s["f_n"],
            }
            for s in scored
        ],
        "selected_exercise_id": selected_exercise_id,
        "selection_rationale": selection_rationale,
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def run_planner(user_state: Dict, candidates: List[Dict]) -> Dict[str, Any]:
    """Run the Intervention Planner over a list of candidate exercises.

    This is the single entry point for the planner component. It executes
    four steps in sequence:
      1. Hard constraint pruning (phase, pain, swelling, weight-bearing)
      2. A* informed search scoring (g(n) risk cost + h(n) rehab benefit)
      3. Selection with deterministic tie-breaking
      4. Audit log generation

    Args:
        user_state: A UserState dict with keys: days_post_op, rehab_phase,
                    pain_level, swelling_level, weight_bearing_status.
        candidates: A list of Exercise dicts pre-filtered by the Rule Engine.

    Returns:
        A dict with "outcome" ("RECOMMEND" or "ESCALATE"), the selected
        exercise (if any), and a full audit log.
    """
    candidates_received = len(candidates)

    # Step 1 --Hard constraint pruning
    survivors, eliminated = prune_candidates(user_state, candidates)

    # ESCALATE if nothing survives
    if not survivors:
        audit_log = _build_audit_log(
            user_state=user_state,
            candidates_received=candidates_received,
            eliminated=eliminated,
            scored=[],
            selected_exercise_id=None,
            selection_rationale="All candidates eliminated by hard constraints.",
        )
        return {
            "outcome": "ESCALATE",
            "reason": "No safe exercise candidates available for current user state.",
            "audit_log": audit_log,
        }

    # Step 2 --A* scoring
    scored = score_candidates(user_state, survivors)

    # Step 3 --Selection
    best, rationale = select_best(scored, candidates)
    selected_exercise = best["_exercise"]

    # Step 4 --Audit log
    audit_log = _build_audit_log(
        user_state=user_state,
        candidates_received=candidates_received,
        eliminated=eliminated,
        scored=scored,
        selected_exercise_id=selected_exercise["exercise_id"],
        selection_rationale=rationale,
    )

    return {
        "outcome": "RECOMMEND",
        "selected_exercise": selected_exercise,
        "audit_log": audit_log,
    }


# ---------------------------------------------------------------------------
# Legacy wrappers --keep existing pipeline integration working until the
# team migrates to run_planner().  These will be removed once all callers
# are updated.
# ---------------------------------------------------------------------------

from wellnessbot.kg.kg import get_exercise, resolve_exercise_id, list_exercises_for_phase
from wellnessbot.nlu.schema import NLUOutput


def plan(nlu: NLUOutput) -> Dict:
    """Legacy planner entry point used by pipeline/run.py when action == RECOMMEND."""
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    ex = get_exercise(ex_id, nlu.event_type) if ex_id else None

    if not ex:
        return {"plan": None, "notes": ["No plan generated (exercise unknown)."]}

    return {
        "exercise_id": ex.exercise_id,
        "exercise_name": ex.name,
        "dose": {"sets": 2, "reps": 8, "frequency_per_day": 1},
        "stop_conditions": ["pain increases", "swelling increases", "new red flags"],
        "effort_score": 1,
        "citations": ex.source_refs,
    }


def plan_alternatives(event_type: str, phase_id: str, top_k: int = 5):
    """Legacy alternative suggestions used by pipeline/run.py when FORBID/CLARIFY."""
    cands = list_exercises_for_phase(event_type, phase_id)
    cands = cands[:top_k]

    return {
        "type": "alternatives",
        "phase_id": phase_id,
        "items": [
            {
                "exercise_id": e.exercise_id,
                "name": e.name,
                "citations": e.source_refs,
            }
            for e in cands
        ],
    }
