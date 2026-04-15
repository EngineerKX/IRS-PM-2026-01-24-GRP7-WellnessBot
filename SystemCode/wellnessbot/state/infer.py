from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from SystemCode.wellnessbot.kg.kg import phase_from_weeks
from SystemCode.wellnessbot.nlu.schema import NLUOutput


@dataclass
class InferredState:
    phase_id: Optional[str]
    risk_flags: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)


def infer_state(nlu: NLUOutput) -> InferredState:
    risk_flags: List[str] = []
    missing_fields: List[str] = []

    phase_id = None
    if nlu.weeks_since_event is not None:
        phase_id = phase_from_weeks(nlu.weeks_since_event, nlu.surgery_type)
    else:
        missing_fields.append("weeks_since_event")

    if nlu.red_flag_terms:
        risk_flags.extend(nlu.red_flag_terms)

    return InferredState(
        phase_id=phase_id,
        risk_flags=sorted(set(risk_flags)),
        missing_fields=missing_fields,
    )