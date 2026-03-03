from __future__ import annotations

QUESTION_BANK = {
    "weeks_since_event": "When was your surgery/injury? Tell me the date (YYYY-MM-DD) **or** how many weeks/days since it happened.",
    "event_type": "What was the event? (ACL surgery / TKR / meniscus / sprain). If unsure, say “unknown”.",
    "requested_exercise_text": "Which exercise are you trying to do? (e.g., heel slides, quad sets, straight leg raise, squats).",
    "pain_score": "On a scale of 0–10, what’s your pain right now?",
    "swelling_level": "How is the swelling: none / mild / moderate / severe?",
    "weight_bearing": "Are you weight-bearing: none / partial / full?",
    "red_flag_screen": "Any red flags today: fever, wound drainage, calf pain/swelling, chest pain, shortness of breath, cannot bear weight, knee locking?",
}

REQUIRED_ORDER = [
    "weeks_since_event",
    "event_type",
    "requested_exercise_text",
    "pain_score",
    "swelling_level",
    "weight_bearing",
]