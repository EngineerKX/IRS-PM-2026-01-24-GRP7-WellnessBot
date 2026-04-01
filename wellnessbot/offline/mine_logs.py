from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]


@dataclass
class CandidateCase:
    case_type: str
    priority: str
    title: str
    description: str
    evidence_count: int
    sample_interaction_ids: List[str]
    proposed_review: str


def read_jsonl(path: Path) -> List[JsonDict]:
    rows: List[JsonDict] = []
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {e}") from e
    return rows


def index_by_interaction_id(rows: Iterable[JsonDict]) -> Dict[str, JsonDict]:
    out: Dict[str, JsonDict] = {}
    for row in rows:
        iid = row.get("interaction_id")
        if not iid:
            continue
        out[iid] = row
    return out


def join_interactions_feedback(
    interactions: List[JsonDict],
    feedback_rows: List[JsonDict],
) -> Tuple[List[JsonDict], List[JsonDict]]:
    interaction_map = index_by_interaction_id(interactions)
    feedback_map = index_by_interaction_id(feedback_rows)

    merged: List[JsonDict] = []
    unmatched_feedback: List[JsonDict] = []

    for interaction in interactions:
        iid = interaction.get("interaction_id")
        merged_row = dict(interaction)
        merged_row["feedback"] = feedback_map.get(iid, {}).get("feedback")
        merged.append(merged_row)

    for fb in feedback_rows:
        iid = fb.get("interaction_id")
        if iid not in interaction_map:
            unmatched_feedback.append(fb)

    return merged, unmatched_feedback


def get_phase_id(row: JsonDict) -> Optional[str]:
    return row.get("audit_trace", {}).get("audit_context", {}).get("phase_id")


def get_rule_ids(row: JsonDict) -> List[str]:
    return [str(x) for x in (row.get("decision", {}) or {}).get("rule_ids", [])]


def get_winning_rule_id(row: JsonDict) -> Optional[str]:
    rule_ids = get_rule_ids(row)
    return rule_ids[0] if rule_ids else None


def get_action(row: JsonDict) -> Optional[str]:
    return (row.get("decision") or {}).get("action")


def get_feedback_thumb(row: JsonDict) -> Optional[str]:
    return (row.get("feedback") or {}).get("thumb")


def get_feedback_expected_action(row: JsonDict) -> Optional[str]:
    return (row.get("feedback") or {}).get("expected_action")


def get_feedback_comment(row: JsonDict) -> Optional[str]:
    return (row.get("feedback") or {}).get("comment")


def get_nlu(row: JsonDict) -> JsonDict:
    return row.get("nlu") or {}


def get_red_flags(row: JsonDict) -> List[str]:
    return list(get_nlu(row).get("red_flag_terms") or [])


def get_user_text(row: JsonDict) -> str:
    return str(row.get("user_text") or "")


def get_numeric_signature(row: JsonDict) -> Tuple[str, Optional[int], str, Optional[str], Optional[str]]:
    nlu = get_nlu(row)
    return (
        get_user_text(row),
        nlu.get("pain_score"),
        str(nlu.get("swelling_level") or "unknown"),
        get_action(row),
        get_winning_rule_id(row),
    )


def summarize_basic_stats(merged: List[JsonDict], unmatched_feedback: List[JsonDict]) -> JsonDict:
    action_counter = Counter()
    rule_counter = Counter()
    thumb_counter = Counter()

    for row in merged:
        action = get_action(row)
        if action:
            action_counter[action] += 1

        for rule_id in get_rule_ids(row):
            rule_counter[rule_id] += 1

        thumb = get_feedback_thumb(row)
        if thumb:
            thumb_counter[thumb] += 1

    return {
        "interaction_count": len(merged),
        "matched_feedback_count": sum(1 for r in merged if r.get("feedback") is not None),
        "unmatched_feedback_count": len(unmatched_feedback),
        "action_distribution": dict(action_counter),
        "rule_frequency": dict(rule_counter),
        "feedback_distribution": dict(thumb_counter),
    }


def mine_rule_disagreement(merged: List[JsonDict]) -> List[JsonDict]:
    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "thumbs_down": 0,
            "expected_action_counter": Counter(),
            "comments": [],
            "sample_interaction_ids": [],
        }
    )

    for row in merged:
        rule_id = get_winning_rule_id(row)
        if not rule_id:
            continue

        bucket = buckets[rule_id]
        bucket["count"] += 1

        iid = row.get("interaction_id")
        if iid and len(bucket["sample_interaction_ids"]) < 5:
            bucket["sample_interaction_ids"].append(iid)

        thumb = get_feedback_thumb(row)
        if thumb == "down":
            bucket["thumbs_down"] += 1

        exp = get_feedback_expected_action(row)
        if exp:
            bucket["expected_action_counter"][exp] += 1

        comment = get_feedback_comment(row)
        if comment and len(bucket["comments"]) < 5:
            bucket["comments"].append(comment)

    out: List[JsonDict] = []
    for rule_id, b in buckets.items():
        count = b["count"]
        down_rate = b["thumbs_down"] / count if count else 0.0
        out.append(
            {
                "rule_id": rule_id,
                "count": count,
                "thumbs_down": b["thumbs_down"],
                "thumbs_down_rate": round(down_rate, 3),
                "top_expected_actions": dict(b["expected_action_counter"].most_common(3)),
                "sample_comments": b["comments"],
                "sample_interaction_ids": b["sample_interaction_ids"],
            }
        )

    out.sort(key=lambda x: (-x["thumbs_down_rate"], -x["count"], x["rule_id"]))
    return out


def mine_numeric_ambiguity(merged: List[JsonDict]) -> List[JsonDict]:
    grouped: Dict[Tuple[str, Optional[str]], List[JsonDict]] = defaultdict(list)

    tracked_inputs = {
        "1", "2", "3", "none",
        "pain 1", "pain 2", "pain 3",
        "swelling 1", "swelling 2", "swelling 3",
    }

    for row in merged:
        user_text = get_user_text(row).strip().lower()
        if user_text not in tracked_inputs:
            continue
        grouped[(user_text, get_phase_id(row))].append(row)

    findings: List[JsonDict] = []

    for (user_text, phase_id), rows in grouped.items():
        signatures = Counter()
        sample_ids: List[str] = []

        for row in rows:
            signatures[get_numeric_signature(row)] += 1
            iid = row.get("interaction_id")
            if iid and len(sample_ids) < 5:
                sample_ids.append(iid)

        if len(signatures) > 1:
            findings.append(
                {
                    "user_text": user_text,
                    "phase_id": phase_id,
                    "variant_count": len(signatures),
                    "variants": [
                        {
                            "user_text": sig[0],
                            "pain_score": sig[1],
                            "swelling_level": sig[2],
                            "action": sig[3],
                            "winning_rule_id": sig[4],
                            "count": count,
                        }
                        for sig, count in signatures.items()
                    ],
                    "sample_interaction_ids": sample_ids,
                }
            )

    findings.sort(key=lambda x: (-x["variant_count"], x["user_text"]))
    return findings


def mine_threshold_consistency(merged: List[JsonDict]) -> List[JsonDict]:
    grouped: Dict[Tuple[Optional[str], Optional[int], str], Counter] = defaultdict(Counter)

    for row in merged:
        nlu = get_nlu(row)
        if str(nlu.get("requested_exercise_text") or ""):
            continue
        if nlu.get("red_flag_terms"):
            continue

        key = (
            get_phase_id(row),
            nlu.get("pain_score"),
            str(nlu.get("swelling_level") or "unknown"),
        )
        grouped[key][get_action(row) or "UNKNOWN"] += 1

    out: List[JsonDict] = []
    for key, counter in grouped.items():
        out.append(
            {
                "phase_id": key[0],
                "pain_score": key[1],
                "swelling_level": key[2],
                "action_distribution": dict(counter),
            }
        )

    out.sort(key=lambda x: (str(x["phase_id"]), str(x["pain_score"]), x["swelling_level"]))
    return out


def mine_red_flag_consistency(merged: List[JsonDict]) -> List[JsonDict]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    sample_ids: Dict[str, List[str]] = defaultdict(list)

    for row in merged:
        red_flags = get_red_flags(row)
        if not red_flags:
            continue

        action = get_action(row) or "UNKNOWN"
        iid = row.get("interaction_id")

        for rf in red_flags:
            grouped[rf][action] += 1
            if iid and len(sample_ids[rf]) < 5:
                sample_ids[rf].append(iid)

    out: List[JsonDict] = []
    for rf, counter in grouped.items():
        out.append(
            {
                "red_flag_term": rf,
                "action_distribution": dict(counter),
                "sample_interaction_ids": sample_ids[rf],
            }
        )

    out.sort(key=lambda x: x["red_flag_term"])
    return out


def generate_candidate_cases(
    rule_disagreement: List[JsonDict],
    numeric_ambiguity: List[JsonDict],
    red_flag_consistency: List[JsonDict],
) -> List[CandidateCase]:
    cases: List[CandidateCase] = []

    for item in rule_disagreement:
        if item["thumbs_down"] >= 1 and item["thumbs_down_rate"] >= 0.5:
            cases.append(
                CandidateCase(
                    case_type="rule_disagreement",
                    priority="HIGH",
                    title=f"Review rule {item['rule_id']}",
                    description=(
                        f"Rule {item['rule_id']} has repeated negative feedback. "
                        f"Thumbs down: {item['thumbs_down']} / {item['count']}."
                    ),
                    evidence_count=item["count"],
                    sample_interaction_ids=item["sample_interaction_ids"],
                    proposed_review="Review threshold and scope of this rule against nearby severity states.",
                )
            )

    for item in numeric_ambiguity:
        cases.append(
            CandidateCase(
                case_type="numeric_ambiguity",
                priority="HIGH",
                title=f"Ambiguous numeric input '{item['user_text']}'",
                description=(
                    f"Same raw input maps to {item['variant_count']} parsed/action variants "
                    f"in phase {item['phase_id']}."
                ),
                evidence_count=item["variant_count"],
                sample_interaction_ids=item["sample_interaction_ids"],
                proposed_review="Constrain UI or log asked_slot to stabilize interpretation.",
            )
        )

    for item in red_flag_consistency:
        actions = set(item["action_distribution"].keys())
        if "ESCALATE" not in actions or len(actions) > 1:
            cases.append(
                CandidateCase(
                    case_type="red_flag_consistency",
                    priority="HIGH",
                    title=f"Review red-flag handling for '{item['red_flag_term']}'",
                    description=f"Red flag '{item['red_flag_term']}' maps to actions {dict(item['action_distribution'])}.",
                    evidence_count=sum(item["action_distribution"].values()),
                    sample_interaction_ids=item["sample_interaction_ids"],
                    proposed_review="Confirm whether this red flag should always escalate and fix policy ordering if needed.",
                )
            )

    return cases


def run_mining(interactions_path: Path, feedback_path: Path) -> Dict[str, Any]:
    interactions = read_jsonl(interactions_path)
    feedback_rows = read_jsonl(feedback_path)

    merged, unmatched_feedback = join_interactions_feedback(interactions, feedback_rows)

    basic_stats = summarize_basic_stats(merged, unmatched_feedback)
    rule_disagreement = mine_rule_disagreement(merged)
    numeric_ambiguity = mine_numeric_ambiguity(merged)
    threshold_consistency = mine_threshold_consistency(merged)
    red_flag_consistency = mine_red_flag_consistency(merged)
    candidate_cases = generate_candidate_cases(
        rule_disagreement=rule_disagreement,
        numeric_ambiguity=numeric_ambiguity,
        red_flag_consistency=red_flag_consistency,
    )

    return {
        "basic_stats": basic_stats,
        "unmatched_feedback": unmatched_feedback,
        "rule_disagreement": rule_disagreement,
        "numeric_ambiguity": numeric_ambiguity,
        "threshold_consistency": threshold_consistency,
        "red_flag_consistency": red_flag_consistency,
        "candidate_cases": [asdict(c) for c in candidate_cases],
    }