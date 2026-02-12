from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from wellnessbot.kg.mock_kg import phase_from_weeks
from wellnessbot.nlu.schema import NLUOutput


@dataclass
class InferredState:
    phase_id: Optional[str]
    risk_flags: List[str]
    missing_fields: List[str]


def infer_state(nlu: NLUOutput) -> InferredState:
    phase_id = None
    if nlu.weeks_since_event is not None:
        phase_id = phase_from_weeks(nlu.weeks_since_event)

    risk = []
    if nlu.red_flag_terms:
        risk.append("red_flags_present")
    if nlu.pain_score is not None and nlu.pain_score >= 7:
        risk.append("high_pain")

    return InferredState(
        phase_id=phase_id,
        risk_flags=risk,
        missing_fields=list(nlu.missing_fields),
    )