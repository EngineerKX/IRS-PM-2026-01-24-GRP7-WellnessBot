from __future__ import annotations

from typing import Dict

from wellnessbot.kg.kg import get_exercise, resolve_exercise_id, list_exercises_for_phase
from wellnessbot.nlu.schema import NLUOutput


def plan(nlu: NLUOutput) -> Dict:
    """
    Minimal planner: returns an execution plan with conservative defaults.
    Only called when decision == RECOMMEND.
    """
    ex_id = resolve_exercise_id(nlu.requested_exercise_text, nlu.event_type)
    ex = get_exercise(ex_id, nlu.event_type) if ex_id else None

    if not ex:
        return {"plan": None, "notes": ["No plan generated (exercise unknown)."]}

    # Conservative dosing (placeholder)
    # effort_score: since CompatibleExercise doesn't have effort_score, use a simple default
    return {
        "exercise_id": ex.exercise_id,
        "exercise_name": ex.name,
        "dose": {"sets": 2, "reps": 8, "frequency_per_day": 1},
        "stop_conditions": ["pain increases", "swelling increases", "new red flags"],
        "effort_score": 1,
        "citations": ex.source_refs,
    }


def plan_alternatives(event_type: str, phase_id: str, top_k: int = 5):
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