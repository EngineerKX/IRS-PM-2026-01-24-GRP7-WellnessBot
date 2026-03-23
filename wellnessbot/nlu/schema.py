from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


SwellingLevel = Literal["none", "mild", "moderate", "severe", "unknown"]
WeightBearing = Literal["none", "partial", "full", "unknown"]
EventType = Literal["acl_surgery", "tkr", "meniscus", "sprain", "unknown"]
NLUSource = Literal["mock", "openai", "mock_fallback"]


class NLUOutput(BaseModel):
    weeks_since_event: Optional[float] = Field(default=None, description="Weeks since injury/surgery/event")
    event_type: EventType = "unknown"
    surgery_date: str = ""
    requested_exercise_text: str = ""
    pain_score: Optional[int] = Field(default=None, ge=0, le=10)
    swelling_level: SwellingLevel = "unknown"
    weight_bearing: WeightBearing = "unknown" 

    symptom_screen_done: bool = False
    symptom_flags: List[str] = Field(default_factory=list)

    red_flag_terms: List[str] = Field(default_factory=list)
    negated_terms: List[str] = Field(default_factory=list)

    missing_fields: List[str] = Field(default_factory=list)
    nlu_source: NLUSource = "mock"