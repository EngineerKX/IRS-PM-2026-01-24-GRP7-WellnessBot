from __future__ import annotations

from typing import List, Tuple

from wellnessbot.nlu.schema import NLUOutput
from wellnessbot.rules.rule_types import Action, RuleResult
from wellnessbot.rules.ruleset import DOMINANCE, RULES


def evaluate_rules(nlu: NLUOutput) -> Tuple[Action, List[RuleResult]]:
    fired: List[RuleResult] = []

    # Pass 1: collect everything except RECOMMEND
    for fn in RULES:
        rr = fn(nlu)
        if rr is None:
            continue
        if rr.action == Action.RECOMMEND:
            continue
        fired.append(rr)

    # If any ESCALATE/FORBID/SUPPORTIVE_CARE/CLARIFY fired, finalize without RECOMMEND noise
    if fired:
        final = sorted(fired, key=lambda r: DOMINANCE[r.action], reverse=True)[0].action
        return final, fired

    # Pass 2: only now allow RECOMMEND rules
    for fn in RULES:
        rr = fn(nlu)
        if rr is None:
            continue
        if rr.action == Action.RECOMMEND:
            fired.append(rr)

    if not fired:
        fired.append(
            RuleResult(
                action=Action.CLARIFY,
                rule_id="R_CLARIFY_DEFAULT_001",
                rationale="Insufficient information to make a safe recommendation.",
                citations=["SRC_RULEBOOK_001#default_clarify"],
                confidence_delta=-0.2,
            )
        )
        return Action.CLARIFY, fired

    # If we’re here, it's recommend-only
    return Action.RECOMMEND, fired