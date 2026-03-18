from __future__ import annotations

QUESTION_BANK = {
    "surgery_date": "When was your surgery or injury? Please reply in YYYY-MM-DD format, or say how many weeks/days ago.",
    "event_type": "What surgery or injury did you have? (ACL surgery / TKR / meniscus / sprain). If unsure, say “unknown”.",
    "symptom_screen": "Are you having any symptoms today, such as fever, excessive bleeding, unusual swelling, or pain? If none, just say “none”.",
    "pain_score": "How would you rate the pain from 1 to 3? (1 = mild, 2 = moderate, 3 = severe)",
    "swelling_level": "How would you rate the swelling? (1 = mild, 2 = moderate, 3 = severe, or say none)",
}

REQUIRED_ORDER = [
    "surgery_date",
    "event_type",
    "symptom_screen",
    "pain_score",
    "swelling_level",
]