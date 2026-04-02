from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


WeightBearing = Literal["none", "partial", "full", "unknown"]
SurgeryType = Literal[
    "arthroscopic_knee_surgery",
    "acl_reconstruction",
    "tkr",
    "sprain_non_surgical",
    "unknown",
]
NLUSource = Literal["mock", "openai", "mock_fallback"]


class NLUOutput(BaseModel):
    weeks_since_event: Optional[float] = Field(
        default=None,
        description="Weeks since injury/surgery/event",
    )
    surgery_type: SurgeryType = "unknown"
    surgery_date: str = ""
    requested_exercise_text: str = ""

    #Standardized to 0–3 scale
    pain_score: Optional[int] = Field(default=None, ge=0, le=3)
    swelling_score: Optional[int] = Field(default=None, ge=0, le=3)

    weight_bearing: WeightBearing = "unknown"

    symptom_screen_done: bool = False
    symptom_flags: List[str] = Field(default_factory=list)

    red_flag_terms: List[str] = Field(default_factory=list)
    negated_terms: List[str] = Field(default_factory=list)

    missing_fields: List[str] = Field(default_factory=list)
    nlu_source: NLUSource = "mock"