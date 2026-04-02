from __future__ import annotations

QUESTION_BANK = {
    "surgery_type": "What surgery type did you have? (e.g. Arthroscopic knee surgery)",
    "surgery_date": "When was your surgery or injury? Please reply in YYYY-MM-DD format, or say how many weeks/days ago.",
    "symptom_screen": "Are you having any symptoms today, such as fever or excessive bleeding? If none, just say “none”.",
    "pain_score": "How would you rate the pain from 0 to 3? (0 = none, 1 = mild, 2 = moderate, 3 = severe)",
    "swelling_score": "How would you rate the swelling from 0 to 3? (0 = none, 1 = mild, 2 = moderate, 3 = severe)",
}

REQUIRED_ORDER = [
    "surgery_type",
    "surgery_date",
    "symptom_screen",
    "pain_score",
    "swelling_score",
]