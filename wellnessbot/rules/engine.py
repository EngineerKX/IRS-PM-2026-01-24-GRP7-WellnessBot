from __future__ import annotations

from typing import Dict, List, Tuple

from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.rules.rule_types import Action, RuleResult
from wellnessbot.rules.ruleset import DOMINANCE, RULES


def evaluate_rules(nlu: NLUOutput) -> Tuple[Action, List[RuleResult]]:
    fired: List[RuleResult] = []

    for fn in RULES:
        rr = fn(nlu)
        if rr is not None:
            fired.append(rr)

    if not fired:
        # Safety default: CLARIFY (never silently RECOMMEND)
        fired.append(
            RuleResult(
                action=Action.CLARIFY,
                rule_id="R_CLARIFY_DEFAULT_001",
                rationale="Insufficient information to make a safe recommendation.",
                citations=["SRC_RULEBOOK_001#default_clarify"],
                confidence_delta=-0.2,
            )
        )

    # choose highest dominance action among fired
    final = sorted(fired, key=lambda r: DOMINANCE[r.action], reverse=True)[0].action
    return final, fired