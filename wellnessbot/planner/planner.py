from __future__ import annotations

from typing import Dict

from wellnessbot.kg.kg import get_exercise, resolve_exercise_id, list_exercises_for_phase
from wellnessbot.nlu.schema import NLUOutput


def plan(nlu: NLUOutput, phase_id: str) -> Dict:
    cands = list_exercises_for_phase(nlu.event_type, phase_id)

    if not cands:
        return {"plan": None, "notes": ["No safe exercises found for this phase."]}

    ex = cands[0]  # simplest ranking

    return {
        "exercise_id": ex.exercise_id,
        "exercise_name": ex.name,
        "stop_conditions": ["pain increases", "swelling increases"],
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