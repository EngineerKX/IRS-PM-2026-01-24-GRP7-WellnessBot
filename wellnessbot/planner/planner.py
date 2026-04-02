from __future__ import annotations

from typing import Dict, List

from wellnessbot.kg.kg import list_exercises_for_phase
from wellnessbot.nlu.schema import NLUOutput


def _equipment_compatible(ex, equipment_available: List[str]) -> bool:
    return all(req in equipment_available for req in ex.equipment_required)


def _score_exercise(ex, equipment_available: List[str], history: List[dict]) -> int:
    score = 100

    # priority weight: lower number = better
    score -= ex.priority * 10

    # soft penalty if equipment is missing
    for req in ex.equipment_required:
        if req not in equipment_available:
            score -= 30

    # history penalty: avoid recent repeats
    recent_ids = {h.get("exercise_id") for h in history[-3:]}
    if ex.exercise_id in recent_ids:
        score -= 25

    return score


def plan(
    nlu: NLUOutput,
    phase_id: str,
    equipment_available: List[str] | None = None,
    exercise_history: List[dict] | None = None,
) -> Dict:
    equipment_available = equipment_available or []
    exercise_history = exercise_history or []

    cands = list_exercises_for_phase(nlu.surgery_type, phase_id)

    if not cands:
        return {"plan": None, "notes": ["No safe exercises found for this phase."]}

    # Prefer exercises compatible with available equipment
    compatible = [ex for ex in cands if _equipment_compatible(ex, equipment_available)]
    if compatible:
        cands = compatible

    ranked = sorted(
        cands,
        key=lambda ex: _score_exercise(ex, equipment_available, exercise_history),
        reverse=True,
    )

    best = ranked[0]

    return {
        "exercise_id": best.exercise_id,
        "exercise_name": best.name,
        "citations": best.source_refs,
        "stop_conditions": ["pain increases"],
    }


def plan_alternatives(surgery_type: str, phase_id: str, top_k: int = 5) -> Dict:
    cands = list_exercises_for_phase(surgery_type, phase_id)

    return {
        "type": "alternatives",
        "phase_id": phase_id,
        "items": [
            {
                "exercise_id": e.exercise_id,
                "name": e.name,
                "citations": e.source_refs,
            }
            for e in cands[:top_k]
        ],
    }