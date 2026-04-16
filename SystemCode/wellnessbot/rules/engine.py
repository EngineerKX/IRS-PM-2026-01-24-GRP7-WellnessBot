from __future__ import annotations

from typing import List, Tuple

from SystemCode.wellnessbot.nlu.schema import NLUOutput
from SystemCode.wellnessbot.rules.rule_types import Action, RuleResult
from SystemCode.wellnessbot.rules.ruleset import DOMINANCE, RULES


def evaluate_rules(nlu: NLUOutput) -> Tuple[Action, List[RuleResult]]:
    fired_non_recommend: List[RuleResult] = []
    fired_recommend: List[RuleResult] = []

    # Pass 1: collect everything except RECOMMEND
    for fn in RULES:
        rr = fn(nlu)
        if rr is None:
            continue
        if rr.action == Action.RECOMMEND:
            continue
        fired_non_recommend.append(rr)

    if fired_non_recommend:
        final = sorted(
            fired_non_recommend,
            key=lambda r: DOMINANCE[r.action],
            reverse=True,
        )[0].action

        # Hard stop on ESCALATE: do not add recommend/self-care noise
        if final == Action.ESCALATE:
            return final, fired_non_recommend

        # Soft stop on FORBID / CLARIFY:
        # still allow supportive RECOMMEND rules to append,
        # but final action remains non-recommend.
        for fn in RULES:
            rr = fn(nlu)
            if rr is None:
                continue
            if rr.action == Action.RECOMMEND:
                fired_recommend.append(rr)

        return final, fired_non_recommend + fired_recommend

    # Pass 2: only now allow RECOMMEND rules
    for fn in RULES:
        rr = fn(nlu)
        if rr is None:
            continue
        if rr.action == Action.RECOMMEND:
            fired_recommend.append(rr)

    if not fired_recommend:
        fallback = RuleResult(
            action=Action.CLARIFY,
            rule_id="R_CLARIFY_DEFAULT_001",
            rationale="Insufficient information to make a safe recommendation.",
            citations=["SRC_RULEBOOK_001#default_clarify"],
            confidence_delta=-0.2,
        )
        return Action.CLARIFY, [fallback]

    return Action.RECOMMEND, fired_recommend