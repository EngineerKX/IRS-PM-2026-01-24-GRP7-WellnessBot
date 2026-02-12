from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

PhaseId = Literal["P_1_ACUTE", "P_2_EARLY_MOB", "P_3_STRENGTH", "P_4_RETURN"]


@dataclass(frozen=True)
class Phase:
    phase_id: PhaseId
    name: str
    week_range: Tuple[float, float]  # inclusive start, exclusive end


@dataclass(frozen=True)
class Exercise:
    exercise_id: str  # E_*
    name: str
    allowed_phases: List[PhaseId]
    min_pain_max: int  # if pain_score > this, avoid
    swelling_allowed: List[str]  # none/mild/moderate/severe/unknown
    effort_score: int  # lower = easier
    source_refs: List[str]  # placeholder citations


PHASES: List[Phase] = [
    Phase("P_1_ACUTE", "Acute / Protection", (0.0, 2.0)),
    Phase("P_2_EARLY_MOB", "Early Mobility", (2.0, 6.0)),
    Phase("P_3_STRENGTH", "Strength", (6.0, 12.0)),
    Phase("P_4_RETURN", "Return to Activity", (12.0, 52.0)),
]

# Minimal exercise KB (not “medical truth”; just constraints for the prototype).
EXERCISES: Dict[str, Exercise] = {
    "E_HEEL_SLIDES": Exercise(
        "E_HEEL_SLIDES",
        "Heel Slides",
        ["P_1_ACUTE", "P_2_EARLY_MOB"],
        min_pain_max=6,
        swelling_allowed=["none", "mild", "moderate", "unknown"],
        effort_score=1,
        source_refs=["SRC_PDF_001#p12"],
    ),
    "E_QUAD_SETS": Exercise(
        "E_QUAD_SETS",
        "Quad Sets",
        ["P_1_ACUTE", "P_2_EARLY_MOB"],
        min_pain_max=6,
        swelling_allowed=["none", "mild", "moderate", "unknown"],
        effort_score=1,
        source_refs=["SRC_PDF_001#p10"],
    ),
    "E_SLR": Exercise(
        "E_SLR",
        "Straight Leg Raise",
        ["P_2_EARLY_MOB", "P_3_STRENGTH"],
        min_pain_max=5,
        swelling_allowed=["none", "mild", "unknown"],
        effort_score=2,
        source_refs=["SRC_PDF_002#p5"],
    ),
    "E_ANKLE_PUMPS": Exercise(
        "E_ANKLE_PUMPS",
        "Ankle Pumps",
        ["P_1_ACUTE", "P_2_EARLY_MOB", "P_3_STRENGTH", "P_4_RETURN"],
        min_pain_max=8,
        swelling_allowed=["none", "mild", "moderate", "severe", "unknown"],
        effort_score=1,
        source_refs=["SRC_PDF_001#p9"],
    ),
    "E_STATIONARY_BIKE": Exercise(
        "E_STATIONARY_BIKE",
        "Stationary Bike",
        ["P_2_EARLY_MOB", "P_3_STRENGTH", "P_4_RETURN"],
        min_pain_max=4,
        swelling_allowed=["none", "mild", "unknown"],
        effort_score=3,
        source_refs=["SRC_PDF_003#p18"],
    ),
    "E_WALL_SIT": Exercise(
        "E_WALL_SIT",
        "Wall Sit",
        ["P_3_STRENGTH", "P_4_RETURN"],
        min_pain_max=3,
        swelling_allowed=["none", "mild"],
        effort_score=4,
        source_refs=["SRC_PDF_003#p22"],
    ),
    "E_STEP_UPS": Exercise(
        "E_STEP_UPS",
        "Step Ups",
        ["P_3_STRENGTH", "P_4_RETURN"],
        min_pain_max=3,
        swelling_allowed=["none", "mild"],
        effort_score=4,
        source_refs=["SRC_PDF_003#p21"],
    ),
    "E_SQUATS": Exercise(
        "E_SQUATS",
        "Squats",
        ["P_3_STRENGTH", "P_4_RETURN"],
        min_pain_max=3,
        swelling_allowed=["none", "mild"],
        effort_score=5,
        source_refs=["SRC_PDF_003#p20"],
    ),
    "E_LUNGES": Exercise(
        "E_LUNGES",
        "Lunges",
        ["P_4_RETURN"],
        min_pain_max=2,
        swelling_allowed=["none"],
        effort_score=6,
        source_refs=["SRC_PDF_004#p7"],
    ),
    "E_LEG_PRESS": Exercise(
        "E_LEG_PRESS",
        "Leg Press",
        ["P_4_RETURN"],
        min_pain_max=2,
        swelling_allowed=["none"],
        effort_score=6,
        source_refs=["SRC_PDF_004#p9"],
    ),
    "E_HAMSTRING_CURLS": Exercise(
        "E_HAMSTRING_CURLS",
        "Hamstring Curls",
        ["P_3_STRENGTH", "P_4_RETURN"],
        min_pain_max=3,
        swelling_allowed=["none", "mild"],
        effort_score=4,
        source_refs=["SRC_PDF_003#p24"],
    ),
    "E_CALF_RAISES": Exercise(
        "E_CALF_RAISES",
        "Calf Raises",
        ["P_3_STRENGTH", "P_4_RETURN"],
        min_pain_max=4,
        swelling_allowed=["none", "mild", "unknown"],
        effort_score=3,
        source_refs=["SRC_PDF_003#p25"],
    ),
}

# Map from NLU requested_exercise_text to KB exercise_id
ALIASES = {
    "heel slides": "E_HEEL_SLIDES",
    "quad sets": "E_QUAD_SETS",
    "straight leg raise": "E_SLR",
    "ankle pumps": "E_ANKLE_PUMPS",
    "stationary bike": "E_STATIONARY_BIKE",
    "wall sit": "E_WALL_SIT",
    "step ups": "E_STEP_UPS",
    "squats": "E_SQUATS",
    "lunges": "E_LUNGES",
    "leg press": "E_LEG_PRESS",
    "hamstring curls": "E_HAMSTRING_CURLS",
    "calf raises": "E_CALF_RAISES",
}


def phase_from_weeks(weeks: float) -> PhaseId:
    for p in PHASES:
        if p.week_range[0] <= weeks < p.week_range[1]:
            return p.phase_id
    # clamp
    return "P_4_RETURN" if weeks >= 12 else "P_1_ACUTE"


def resolve_exercise_id(requested_exercise_text: str) -> Optional[str]:
    return ALIASES.get((requested_exercise_text or "").strip().lower())


def get_exercise(exercise_id: str):
    return EXERCISES.get(exercise_id)