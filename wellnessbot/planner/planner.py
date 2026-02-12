from __future__ import annotations

from typing import Dict, Optional

from wellnessbot.kg.mock_kg import get_exercise, resolve_exercise_id
from wellnessbot.nlu.schema import NLUOutput


def plan(nlu: NLUOutput) -> Dict:
    """
    Minimal planner: returns an execution plan with conservative defaults.
    Only called when decision == RECOMMEND.
    """
    ex_id = resolve_exercise_id(nlu.requested_exercise_text)
    ex = get_exercise(ex_id) if ex_id else None

    if not ex:
        return {"plan": None, "notes": ["No plan generated (exercise unknown)."]}

    # lowest-effort conservative dosing (placeholder)
    return {
        "exercise_id": ex.exercise_id,
        "exercise_name": ex.name,
        "dose": {"sets": 2, "reps": 8, "frequency_per_day": 1},
        "stop_conditions": ["pain increases", "swelling increases", "new red flags"],
        "effort_score": ex.effort_score,
    }