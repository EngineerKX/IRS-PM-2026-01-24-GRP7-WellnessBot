from __future__ import annotations

QUESTION_BANK = {
    "weeks_since_event": "When was your surgery/injury? Tell me the date (YYYY-MM-DD) **or** how many weeks/days since it happened.",
    "event_type": "What was the event? (ACL surgery / TKR / meniscus / sprain). If unsure, say “unknown”.",
    "pain_score": "On a scale of 0–10, what’s your pain right now?",
    "swelling_level": "How is the swelling: none / mild / moderate / severe?",
    "weight_bearing": "Are you weight-bearing: none / partial / full?",
    "red_flag_screen": "Any red flags today: fever, wound drainage, calf pain/swelling, chest pain, shortness of breath, cannot bear weight, knee locking?",
    "surgery_date": "What was your surgery date? Please reply in YYYY-MM-DD format.",
}

REQUIRED_ORDER = [
    "surgery_date",
    "event_type",
    "pain_score",
    "swelling_level",
    "weight_bearing",
]