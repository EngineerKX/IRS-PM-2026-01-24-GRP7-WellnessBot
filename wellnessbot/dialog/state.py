from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class ConversationState:
    # Core slots
    surgery_type: str = "unknown"
    surgery_date: str = ""
    weeks_since_event: Optional[float] = None
    pain_score: Optional[int] = None
    swelling_score: Optional[int] = None
    weight_bearing: str = "unknown"
    requested_exercise_text: str = ""

    # Symptom flow
    symptom_screen_done: bool = False
    red_flag_terms: List[str] = field(default_factory=list)
    symptom_flags: List[str] = field(default_factory=list)

    # Optional profile/preferences
    equipment_available: List[str] = field(default_factory=list)

    # Special-case decision/dialog control
    exercise_blocked: bool = False
    block_reason: str = ""
    pending_followup_slots: List[str] = field(default_factory=list)

    # Meta / dialog control
    negated_terms: List[str] = field(default_factory=list)
    asked_slots: List[str] = field(default_factory=list)
    last_user_text: str = ""
    turn_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_expected_slot: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "surgery_type": self.surgery_type,
            "surgery_date": self.surgery_date,
            "weeks_since_event": self.weeks_since_event,
            "pain_score": self.pain_score,
            "swelling_score": self.swelling_score,
            "weight_bearing": self.weight_bearing,
            "requested_exercise_text": self.requested_exercise_text,
            "symptom_screen_done": self.symptom_screen_done,
            "red_flag_terms": self.red_flag_terms,
            "symptom_flags": self.symptom_flags,
            "equipment_available": self.equipment_available,
            "exercise_blocked": self.exercise_blocked,
            "block_reason": self.block_reason,
            "pending_followup_slots": self.pending_followup_slots,
            "last_user_text": self.last_user_text,
            "turn_count": self.turn_count,
            "history": self.history,
            "negated_terms": self.negated_terms,
            "asked_slots": self.asked_slots,
            "last_expected_slot": self.last_expected_slot,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConversationState":
        if not d:
            return ConversationState()

        return ConversationState(
            surgery_type=d.get("surgery_type", d.get("event_type", "unknown")),
            surgery_date=d.get("surgery_date", "") or "",
            weeks_since_event=d.get("weeks_since_event"),
            pain_score=d.get("pain_score"),
            swelling_score=d.get("swelling_score"),
            weight_bearing=d.get("weight_bearing", "unknown"),
            requested_exercise_text=d.get("requested_exercise_text", "") or "",
            symptom_screen_done=bool(d.get("symptom_screen_done", False)),
            red_flag_terms=d.get("red_flag_terms", []) or [],
            symptom_flags=d.get("symptom_flags", []) or [],
            equipment_available=d.get("equipment_available", []) or [],
            exercise_blocked=bool(d.get("exercise_blocked", False)),
            block_reason=d.get("block_reason", "") or "",
            pending_followup_slots=d.get("pending_followup_slots", []) or [],
            last_user_text=d.get("last_user_text", "") or "",
            turn_count=int(d.get("turn_count", 0) or 0),
            history=d.get("history", []) or [],
            negated_terms=d.get("negated_terms", []) or [],
            asked_slots=d.get("asked_slots", []) or [],
            last_expected_slot=d.get("last_expected_slot", "") or "",
        )