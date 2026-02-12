from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from wellnessbot.nlu.schema import NLUOutput


RED_FLAGS = [
    "fever",
    "locking",
    "cannot bear weight",
    "can't bear weight",
    "calf pain",
    "shortness of breath",
    "chest pain",
    "redness",
    "hot knee",
    "wound drainage",
    "pus",
    "night sweats",
]

EXERCISE_KEYWORDS = [
    "squats",
    "heel slides",
    "quad sets",
    "straight leg raise",
    "slr",
    "lunges",
    "step ups",
    "leg press",
    "wall sit",
    "stationary bike",
    "cycling",
    "hamstring curls",
    "calf raises",
    "balance",
    "single leg balance",
]

EVENT_KEYWORDS: Dict[str, str] = {
    "acl": "acl_surgery",
    "tkr": "tkr",
    "knee replacement": "tkr",
    "meniscus": "meniscus",
    "sprain": "sprain",
}


def _extract_weeks(text: str) -> Optional[float]:
    t = text.lower()

    # "3 weeks" / "3 wk" / "3w"
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(weeks?|wks?|wk|w)\b", t)
    if m:
        return float(m.group(1))

    # "21 days" => 3 weeks
    m = re.search(r"\b(\d+(?:\.\d+)?)\s*(days?|d)\b", t)
    if m:
        days = float(m.group(1))
        return round(days / 7.0, 2)

    return None


def _extract_pain_score(text: str) -> Optional[int]:
    t = text.lower()
    # "pain 4/10", "pain: 4/10", "4 out of 10"
    m = re.search(r"\bpain\b[^0-9]{0,6}(\d{1,2})\s*(?:/|out of)\s*10\b", t)
    if m:
        v = int(m.group(1))
        return max(0, min(10, v))
    # "pain 4"
    m = re.search(r"\bpain\b[^0-9]{0,6}(\d{1,2})\b", t)
    if m:
        v = int(m.group(1))
        return max(0, min(10, v))
    return None


def _extract_swelling(text: str) -> str:
    t = text.lower()
    if re.search(r"\bno swelling\b|\bwithout swelling\b", t):
        return "none"
    for level in ["mild", "moderate", "severe"]:
        if re.search(rf"\b{level}\s+swelling\b|\bswelling\s+is\s+{level}\b|\b{level}\b", t):
            return level
    if "swelling" in t:
        return "unknown"
    return "unknown"


def _extract_weight_bearing(text: str) -> str:
    t = text.lower()
    if re.search(r"\bnon[-\s]?weight[-\s]?bearing\b|\bnwb\b", t):
        return "none"
    if re.search(r"\bpartial weight bearing\b|\bpwb\b", t):
        return "partial"
    if re.search(r"\bfull weight bearing\b|\bfwb\b", t):
        return "full"
    if "weight bearing" in t:
        return "unknown"
    return "unknown"


def _extract_event_type(text: str) -> str:
    t = text.lower()
    for k, v in EVENT_KEYWORDS.items():
        if k in t:
            return v
    if "surgery" in t or "operation" in t:
        # still unknown surgery type
        return "unknown"
    return "unknown"


def _find_negations(text: str) -> List[Tuple[str, int]]:
    """
    Return list of (negation_word, index). Basic scope used for nearby term matching.
    """
    t = text.lower()
    neg_words = ["no", "not", "without", "denies", "deny", "never"]
    out = []
    for w in neg_words:
        for m in re.finditer(rf"\b{w}\b", t):
            out.append((w, m.start()))
    return out


def _is_negated(term: str, text: str) -> bool:
    """
    Simple negation: if a negation word occurs within ~20 chars before the term occurrence.
    """
    t = text.lower()
    term_l = term.lower()

    for m in re.finditer(re.escape(term_l), t):
        start = m.start()
        window = t[max(0, start - 25) : start]
        if re.search(r"\b(no|not|without|denies|deny|never)\b", window):
            return True
    return False


def _extract_red_flags(text: str) -> Tuple[List[str], List[str]]:
    found: List[str] = []
    negated: List[str] = []
    t = text.lower()

    for rf in RED_FLAGS:
        if rf in t:
            if _is_negated(rf, t):
                negated.append(rf)
            else:
                found.append(rf)

    # normalize a few patterns
    if ("cannot bear weight" in t or "can't bear weight" in t) and not _is_negated("bear weight", t):
        if "cannot bear weight" not in found:
            found.append("cannot bear weight")

    return sorted(set(found)), sorted(set(negated))


def _extract_requested_exercise(text: str) -> str:
    t = text.lower()

    # prioritize explicit "want to do X" / "can I do X"
    m = re.search(r"\b(?:want to do|can i do|request(?:ed)?|plan to do)\b(.+)", t)
    if m:
        tail = m.group(1)
        # pick first matching exercise keyword from tail
        for ex in EXERCISE_KEYWORDS:
            if ex in tail:
                return ex

    # fallback: any exercise keyword anywhere
    for ex in EXERCISE_KEYWORDS:
        if ex in t:
            # unify some aliases
            if ex == "slr":
                return "straight leg raise"
            if ex == "cycling":
                return "stationary bike"
            return ex

    return ""


def extract_mock(user_text: str) -> NLUOutput:
    weeks = _extract_weeks(user_text)
    pain = _extract_pain_score(user_text)
    swelling = _extract_swelling(user_text)
    wb = _extract_weight_bearing(user_text)
    event_type = _extract_event_type(user_text)
    req_ex = _extract_requested_exercise(user_text)

    red_flags, negated = _extract_red_flags(user_text)

    missing = []
    if weeks is None:
        missing.append("weeks_since_event")
    if not req_ex:
        missing.append("requested_exercise_text")

    return NLUOutput(
        weeks_since_event=weeks,
        event_type=event_type,
        requested_exercise_text=req_ex,
        pain_score=pain,
        swelling_level=swelling,
        weight_bearing=wb,
        red_flag_terms=red_flags,
        negated_terms=negated,
        missing_fields=missing,
        nlu_source="mock",
    )