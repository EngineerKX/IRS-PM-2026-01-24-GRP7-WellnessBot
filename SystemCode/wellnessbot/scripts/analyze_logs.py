from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_all(log_dir: Path, pattern: str) -> List[Dict[str, Any]]:
    files = sorted(log_dir.glob(pattern))
    rows: List[Dict[str, Any]] = []
    for fp in files:
        rows.extend(list(iter_jsonl(fp)))
    return rows


def _fmt_pct(x: int, total: int) -> str:
    return f"{(x / total):.1%}" if total > 0 else "n/a"


def main() -> None:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    if not log_dir.exists():
        print(f"No logs directory found at: {log_dir}")
        return

    interactions = load_all(log_dir, "interactions_*.jsonl")
    feedbacks = load_all(log_dir, "feedback_*.jsonl")

    if not interactions:
        print("No interaction logs found.")
        return

    # ---- Index interactions by interaction_id ----
    by_id: Dict[str, Dict[str, Any]] = {}
    missing_iid_interactions = 0
    for r in interactions:
        iid = r.get("interaction_id")
        if not iid:
            missing_iid_interactions += 1
            continue
        by_id[iid] = r

    # ---- Basic interaction stats ----
    n = len(interactions)
    action_ctr = Counter()
    nlu_source_ctr = Counter()
    rule_ctr = Counter()
    missing_ctr = Counter()

    for r in interactions:
        decision = r.get("decision") or {}
        action_ctr[decision.get("action") or "UNKNOWN"] += 1

        nlu = r.get("nlu") or {}
        nlu_source_ctr[nlu.get("nlu_source") or "UNKNOWN"] += 1

        audit = r.get("audit_trace") or {}
        state = audit.get("state") or {}
        for m in state.get("missing_fields") or []:
            missing_ctr[m] += 1

        for fired in audit.get("rules_fired") or []:
            rule_ctr[fired.get("rule_id") or "UNKNOWN_RULE"] += 1

    print(f"Total interactions: {n}")
    if missing_iid_interactions:
        print(f"WARNING: interactions missing interaction_id: {missing_iid_interactions}")
    print()

    print("Decision distribution:")
    for k, v in action_ctr.most_common():
        print(f"  {k:10s} {v:5d}  ({_fmt_pct(v, n)})")
    print()

    print("NLU source distribution:")
    for k, v in nlu_source_ctr.most_common():
        print(f"  {k:15s} {v:5d}  ({_fmt_pct(v, n)})")
    print()

    print("Top fired rules:")
    for k, v in rule_ctr.most_common(10):
        print(f"  {k:24s} {v:5d}")
    print()

    print("Most common missing fields:")
    for k, v in missing_ctr.most_common(10):
        print(f"  {k:24s} {v:5d}")
    print()

    # ---- Feedback stats (JOIN) ----
    if not feedbacks:
        print("No feedback logs found (yet).")
        return

    thumb_ctr = Counter()
    down_rule_ctr = Counter()
    disagreement_ctr = Counter()
    down_by_action_ctr = Counter()

    total_feedback = 0
    missing_iid_feedback = 0
    unjoinable_feedback = 0

    down_examples: List[Dict[str, Any]] = []

    for fb in feedbacks:
        total_feedback += 1

        iid = fb.get("interaction_id")
        if not iid:
            missing_iid_feedback += 1
            continue

        f = fb.get("feedback") or {}
        thumb = f.get("thumb") or "UNKNOWN"
        expected = f.get("expected_action")

        thumb_ctr[thumb] += 1

        # join to interaction
        inter = by_id.get(iid)
        if not inter:
            unjoinable_feedback += 1
            continue

        decision_action = (inter.get("decision") or {}).get("action") or "UNKNOWN"
        decision_rules = (inter.get("decision") or {}).get("rule_ids", []) or []
        user_text = inter.get("user_text") or ""

        # thumb-down by decision action
        if thumb == "down":
            down_by_action_ctr[decision_action] += 1
            for rid in decision_rules:
                down_rule_ctr[rid] += 1

            # store a few examples
            if len(down_examples) < 5:
                down_examples.append(
                    {
                        "interaction_id": iid,
                        "decision_action": decision_action,
                        "expected_action": expected,
                        "rule_ids": decision_rules,
                        "user_text": user_text,
                    }
                )

        # disagreement counts (expected vs actual)
        if expected and decision_action and expected != decision_action:
            disagreement_ctr[f"{decision_action}->{expected}"] += 1

    print(f"Total feedback entries: {total_feedback}")
    if missing_iid_feedback:
        print(f"WARNING: feedback entries missing interaction_id: {missing_iid_feedback}")
    if unjoinable_feedback:
        print(f"WARNING: feedback entries not joinable to interactions: {unjoinable_feedback}")
    print()

    print("Thumb distribution:")
    for k, v in thumb_ctr.most_common():
        print(f"  {k:8s} {v:5d}  ({_fmt_pct(v, total_feedback)})")
    print()

    if down_by_action_ctr:
        print("Thumb-down counts by decision action:")
        for k, v in down_by_action_ctr.most_common():
            print(f"  {k:10s} {v:5d}")
        print()

    print("Top rules associated with 👎:")
    for k, v in down_rule_ctr.most_common(10):
        print(f"  {k:24s} {v:5d}")
    print()

    if disagreement_ctr:
        print("Most common action disagreements (decision -> expected):")
        for k, v in disagreement_ctr.most_common(10):
            print(f"  {k:24s} {v:5d}")
        print()

    if down_examples:
        print("Sample 👎 examples (first 5):")
        for ex in down_examples:
            print("-" * 60)
            print(f"interaction_id: {ex['interaction_id']}")
            print(f"decision: {ex['decision_action']}  expected: {ex.get('expected_action')}")
            print(f"rule_ids: {ex['rule_ids']}")
            print(f"user_text: {ex['user_text']}")
        print()


if __name__ == "__main__":
    main()