from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional


class Action(str, Enum):
    RECOMMEND = "RECOMMEND"
    FORBID = "FORBID"
    CLARIFY = "CLARIFY"
    ESCALATE = "ESCALATE"


@dataclass
class RuleResult:
    action: Action
    rule_id: str
    rationale: str
    citations: List[str]
    confidence_delta: float = 0.0