from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonDict = Dict[str, Any]

# --------------------------------------------
# PLANNER MINING THRESHOLDS (TUNABLE)
# --------------------------------------------
PLANNER_THRESHOLDS = {
    "HIGH": {
        "min_count": 5,
        "min_thumbs_down": 4,
        "min_rate": 0.5,
    },
    "MEDIUM": {
        "min_count": 3,
        "min_thumbs_down": 2,
        "min_rate": 0.0,
    },
}


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


def read_jsonl_folder(folder: Path) -> List[JsonDict]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected a folder path, got file: {folder}")

    files = sorted(folder.glob("*.jsonl"))
    if not files:
        raise ValueError(f"No .jsonl files found in folder: {folder}")

    rows: List[JsonDict] = []
    for file in files:
        rows.extend(read_jsonl(file))

    return rows


def read_jsonl_source(path_or_folder: Path) -> List[JsonDict]:
    if path_or_folder.is_dir():
        return read_jsonl_folder(path_or_folder)
    return read_jsonl(path_or_folder)


def is_interaction_row(row: JsonDict) -> bool:
    return bool(row.get("interaction_id")) and "decision" in row


def is_feedback_row(row: JsonDict) -> bool:
    return bool(row.get("interaction_id")) and "feedback" in row


def read_interactions_source(path_or_folder: Path) -> List[JsonDict]:
    rows = read_jsonl_source(path_or_folder)
    out = [row for row in rows if is_interaction_row(row)]
    if not out:
        raise ValueError(f"No interaction rows found in source: {path_or_folder}")
    return out


def read_feedback_source(path_or_folder: Path) -> List[JsonDict]:
    rows = read_jsonl_source(path_or_folder)
    out = [row for row in rows if is_feedback_row(row)]
    return out


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
    rules_fired = row.get("audit_trace", {}).get("rules_fired", []) or []
    if not rules_fired:
        rule_ids = get_rule_ids(row)
        return rule_ids[0] if rule_ids else None

    dominance = {
        "ESCALATE": 4,
        "FORBID": 3,
        "CLARIFY": 2,
        "RECOMMEND": 1,
    }

    def sort_key(rule: JsonDict):
        action = str(rule.get("action") or "RECOMMEND").upper()
        delta = float(rule.get("confidence_delta") or 0.0)
        return (dominance.get(action, 0), delta)

    winner = sorted(rules_fired, key=sort_key, reverse=True)[0]
    return winner.get("rule_id")


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


def get_planner(row: JsonDict) -> JsonDict:
    return row.get("audit_trace", {}).get("planner") or {}


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

        thumb = get_feedback_thumb(row)
        iid = row.get("interaction_id")
        if thumb == "down":
            bucket["thumbs_down"] += 1
            if iid and iid not in bucket["sample_interaction_ids"] and len(bucket["sample_interaction_ids"]) < 5:
                bucket["sample_interaction_ids"].append(iid)

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


def mine_rule_combination_summary(merged: List[JsonDict]) -> List[JsonDict]:
    combo_counter: Counter = Counter()
    thumbs_down_counter: Counter = Counter()
    expected_action_counter: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    sample_ids: Dict[Tuple[str, ...], List[str]] = defaultdict(list)
    sample_comments: Dict[Tuple[str, ...], List[str]] = defaultdict(list)

    for row in merged:
        rule_ids = tuple(sorted(get_rule_ids(row)))
        if not rule_ids:
            continue

        combo_counter[rule_ids] += 1

        thumb = get_feedback_thumb(row)
        iid = row.get("interaction_id")
        if thumb == "down":
            thumbs_down_counter[rule_ids] += 1
            if iid and iid not in sample_ids[rule_ids] and len(sample_ids[rule_ids]) < 5:
                sample_ids[rule_ids].append(iid)

        exp = get_feedback_expected_action(row)
        if exp:
            expected_action_counter[rule_ids][exp] += 1

        comment = get_feedback_comment(row)
        if comment and len(sample_comments[rule_ids]) < 5:
            sample_comments[rule_ids].append(comment)

    out: List[JsonDict] = []
    for combo, count in combo_counter.items():
        thumbs_down = thumbs_down_counter[combo]
        down_rate = thumbs_down / count if count else 0.0
        out.append(
            {
                "rule_combination": list(combo),
                "count": count,
                "thumbs_down": thumbs_down,
                "thumbs_down_rate": round(down_rate, 3),
                "top_expected_actions": dict(expected_action_counter[combo].most_common(3)),
                "sample_comments": sample_comments[combo],
                "sample_interaction_ids": sample_ids[combo],
            }
        )

    out.sort(key=lambda x: (-x["thumbs_down_rate"], -x["count"], x["rule_combination"]))
    return out


def mine_rule_participation_summary(rule_combination_summary: List[JsonDict]) -> List[JsonDict]:
    rule_stats: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "negative_combination_count": 0,
            "total_combination_count": 0,
            "sample_combinations": [],
            "sample_interaction_ids": [],
        }
    )

    for item in rule_combination_summary:
        rules = item["rule_combination"]
        count = item["count"]
        thumbs_down = item["thumbs_down"]
        is_negative_combo = thumbs_down >= 1 and item["thumbs_down_rate"] >= 0.5

        for rule_id in rules:
            rule_stats[rule_id]["total_combination_count"] += count

            if is_negative_combo:
                rule_stats[rule_id]["negative_combination_count"] += 1

                if len(rule_stats[rule_id]["sample_combinations"]) < 5:
                    rule_stats[rule_id]["sample_combinations"].append(rules)

                for iid in item["sample_interaction_ids"]:
                    if len(rule_stats[rule_id]["sample_interaction_ids"]) >= 5:
                        break
                    if iid not in rule_stats[rule_id]["sample_interaction_ids"]:
                        rule_stats[rule_id]["sample_interaction_ids"].append(iid)

    out: List[JsonDict] = []
    for rule_id, stats in rule_stats.items():
        if stats["negative_combination_count"] == 0:
            continue

        out.append(
            {
                "rule_id": rule_id,
                "negative_combination_count": stats["negative_combination_count"],
                "total_combination_count": stats["total_combination_count"],
                "sample_combinations": stats["sample_combinations"],
                "sample_interaction_ids": stats["sample_interaction_ids"],
            }
        )

    out.sort(
        key=lambda x: (-x["negative_combination_count"], -x["total_combination_count"], x["rule_id"])
    )
    return out


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
            str(nlu.get("swelling_score") if nlu.get("swelling_score") is not None else "unknown"),
        )
        grouped[key][get_action(row) or "UNKNOWN"] += 1

    out: List[JsonDict] = []
    for key, counter in grouped.items():
        out.append(
            {
                "phase_id": key[0],
                "pain_score": key[1],
                "swelling_score": key[2],
                "action_distribution": dict(counter),
            }
        )

    out.sort(key=lambda x: (str(x["phase_id"]), str(x["pain_score"]), x["swelling_score"]))
    return out


def mine_red_flag_consistency(merged: List[JsonDict]) -> List[JsonDict]:
    grouped: Dict[Tuple[str, Optional[str]], Counter] = defaultdict(Counter)
    sample_ids: Dict[Tuple[str, Optional[str]], List[str]] = defaultdict(list)

    for row in merged:
        red_flags = get_red_flags(row)
        if not red_flags:
            continue

        action = get_action(row) or "UNKNOWN"
        iid = row.get("interaction_id")
        phase_id = get_phase_id(row)

        for rf in red_flags:
            key = (rf, phase_id)
            grouped[key][action] += 1
            if iid and len(sample_ids[key]) < 5:
                sample_ids[key].append(iid)

    out: List[JsonDict] = []
    for (rf, phase_id), counter in grouped.items():
        out.append(
            {
                "red_flag_term": rf,
                "phase_id": phase_id,
                "action_distribution": dict(counter),
                "sample_interaction_ids": sample_ids[(rf, phase_id)],
            }
        )

    out.sort(key=lambda x: (x["red_flag_term"], str(x["phase_id"])))
    return out


def mine_planner_selection_summary(merged: List[JsonDict]) -> List[JsonDict]:
    grouped: Dict[Tuple[str, str, str, int], Dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "thumbs_down": 0,
            "expected_action_counter": Counter(),
            "sample_interaction_ids": [],
            "sample_comments": [],
        }
    )

    for row in merged:
        planner = get_planner(row)

        exercise_id = planner.get("exercise_id")
        exercise_name = planner.get("exercise_name")
        phase_id = planner.get("phase_id")
        priority = planner.get("priority")

        if not exercise_id or not exercise_name or not phase_id or priority is None:
            continue

        key = (exercise_id, exercise_name, phase_id, priority)
        bucket = grouped[key]
        bucket["count"] += 1

        thumb = get_feedback_thumb(row)
        iid = row.get("interaction_id")
        if thumb == "down":
            bucket["thumbs_down"] += 1
            if iid and iid not in bucket["sample_interaction_ids"] and len(bucket["sample_interaction_ids"]) < 5:
                bucket["sample_interaction_ids"].append(iid)

        exp = get_feedback_expected_action(row)
        if exp:
            bucket["expected_action_counter"][exp] += 1

        comment = get_feedback_comment(row)
        if comment and len(bucket["sample_comments"]) < 5:
            bucket["sample_comments"].append(comment)

    out: List[JsonDict] = []
    for (exercise_id, exercise_name, phase_id, priority), bucket in grouped.items():
        count = bucket["count"]
        thumbs_down = bucket["thumbs_down"]
        down_rate = thumbs_down / count if count else 0.0

        out.append(
            {
                "exercise_id": exercise_id,
                "exercise_name": exercise_name,
                "phase_id": phase_id,
                "priority": priority,
                "count": count,
                "thumbs_down": thumbs_down,
                "thumbs_down_rate": round(down_rate, 3),
                "top_expected_actions": dict(bucket["expected_action_counter"].most_common(3)),
                "sample_comments": bucket["sample_comments"],
                "sample_interaction_ids": bucket["sample_interaction_ids"],
            }
        )

    out.sort(
        key=lambda x: (
            -x["thumbs_down_rate"],
            -x["thumbs_down"],
            -x["count"],
            x["phase_id"],
            x["priority"],
            x["exercise_id"],
        )
    )
    return out


def mine_planner_behavior_summary(merged: List[JsonDict]) -> JsonDict:
    priority_counter = Counter()
    painkiller_counter = Counter()
    no_plan_count = 0

    for row in merged:
        planner = get_planner(row)

        if not planner:
            continue

        priority = planner.get("priority")
        if priority is not None:
            priority_counter[str(priority)] += 1

        painkiller_counter[str(bool(planner.get("recommend_painkiller", False)))] += 1

        if not planner.get("exercise_id"):
            no_plan_count += 1

    return {
        "priority_distribution": dict(priority_counter),
        "recommend_painkiller_distribution": dict(painkiller_counter),
        "no_plan_count": no_plan_count,
    }


def generate_candidate_cases(
    rule_combination_summary: List[JsonDict],
    rule_participation_summary: List[JsonDict],
    red_flag_consistency: List[JsonDict],
    planner_selection_summary: List[JsonDict],
    planner_behavior_summary: JsonDict,
) -> List[CandidateCase]:
    cases: List[CandidateCase] = []

    for item in rule_combination_summary:
        if item["thumbs_down"] >= 1 and item["thumbs_down_rate"] >= 0.5:
            cases.append(
                CandidateCase(
                    case_type="rule_combination",
                    priority="HIGH",
                    title=f"Review rule combination {item['rule_combination']}",
                    description=(
                        f"Rule combination {item['rule_combination']} has repeated negative feedback. "
                        f"Thumbs down: {item['thumbs_down']} / {item['count']}."
                    ),
                    evidence_count=item["count"],
                    sample_interaction_ids=item["sample_interaction_ids"],
                    proposed_review="Review this rule set as a unit and confirm whether the co-firing behavior is intended.",
                )
            )

    for item in rule_participation_summary:
        if item["negative_combination_count"] >= 2:
            cases.append(
                CandidateCase(
                    case_type="rule_participation",
                    priority="HIGH",
                    title=f"Review repeated negative participation of {item['rule_id']}",
                    description=(
                        f"Rule {item['rule_id']} appears in {item['negative_combination_count']} "
                        f"negative rule combinations."
                    ),
                    evidence_count=item["negative_combination_count"],
                    sample_interaction_ids=item["sample_interaction_ids"],
                    proposed_review="Check whether this rule is the common contributing factor across different negative combinations.",
                )
            )

    for item in red_flag_consistency:
        actions = set(item["action_distribution"].keys())
        if len(actions) > 1:
            cases.append(
                CandidateCase(
                    case_type="red_flag_consistency",
                    priority="HIGH",
                    title=f"Review red-flag handling for '{item['red_flag_term']}' in {item['phase_id']}",
                    description=(
                        f"Red flag '{item['red_flag_term']}' in phase {item['phase_id']} "
                        f"maps to actions {dict(item['action_distribution'])}."
                    ),
                    evidence_count=sum(item["action_distribution"].values()),
                    sample_interaction_ids=item["sample_interaction_ids"],
                    proposed_review="Check whether mixed actions within the same phase are intentional; if not, refine the phase-specific red-flag policy or rule ordering.",
                )
            )

    for item in planner_selection_summary:
        count = item["count"]
        thumbs_down = item["thumbs_down"]
        rate = item["thumbs_down_rate"]

        high_cfg = PLANNER_THRESHOLDS["HIGH"]
        med_cfg = PLANNER_THRESHOLDS["MEDIUM"]

        if (
            count >= high_cfg["min_count"]
            and thumbs_down >= high_cfg["min_thumbs_down"]
            and rate >= high_cfg["min_rate"]
        ):
            priority = "HIGH"
        elif (
            count >= med_cfg["min_count"]
            and thumbs_down >= med_cfg["min_thumbs_down"]
            and rate >= med_cfg["min_rate"]
        ):
            priority = "MEDIUM"
        else:
            continue

        cases.append(
            CandidateCase(
                case_type="planner_selection_issue",
                priority=priority,
                title=f"Review planner selection {item['exercise_id']} in {item['phase_id']}",
                description=(
                    f"Planner selection {item['exercise_name']} ({item['exercise_id']}) in phase "
                    f"{item['phase_id']} received negative feedback. "
                    f"Thumbs down: {thumbs_down} / {count}."
                ),
                evidence_count=count,
                sample_interaction_ids=item["sample_interaction_ids"],
                proposed_review="Check whether planner priority, history handling, or candidate ranking is causing an unsuitable exercise choice.",
            )
        )

    no_plan_count = planner_behavior_summary.get("no_plan_count", 0)
    if no_plan_count > 0:
        cases.append(
            CandidateCase(
                case_type="planner_no_plan",
                priority="MEDIUM",
                title="Review planner no-plan outcomes",
                description=f"Planner returned no exercise selection in {no_plan_count} interaction(s).",
                evidence_count=no_plan_count,
                sample_interaction_ids=[],
                proposed_review="Check whether exercise pool exhaustion, history progression, or phase completion handling is behaving as intended.",
            )
        )

    return cases


def run_mining(interactions_path: Path, feedback_path: Path) -> Dict[str, Any]:
    interactions = read_interactions_source(interactions_path)
    feedback_rows = read_feedback_source(feedback_path)

    merged, unmatched_feedback = join_interactions_feedback(interactions, feedback_rows)

    basic_stats = summarize_basic_stats(merged, unmatched_feedback)
    rule_disagreement = mine_rule_disagreement(merged)
    rule_combination_summary = mine_rule_combination_summary(merged)
    rule_participation_summary = mine_rule_participation_summary(rule_combination_summary)
    threshold_consistency = mine_threshold_consistency(merged)
    red_flag_consistency = mine_red_flag_consistency(merged)
    planner_selection_summary = mine_planner_selection_summary(merged)
    planner_behavior_summary = mine_planner_behavior_summary(merged)

    candidate_cases = generate_candidate_cases(
        rule_combination_summary=rule_combination_summary,
        rule_participation_summary=rule_participation_summary,
        red_flag_consistency=red_flag_consistency,
        planner_selection_summary=planner_selection_summary,
        planner_behavior_summary=planner_behavior_summary,
    )

    return {
        "basic_stats": basic_stats,
        "unmatched_feedback": unmatched_feedback,
        "rule_disagreement": rule_disagreement,
        "rule_combination_summary": rule_combination_summary,
        "rule_participation_summary": rule_participation_summary,
        "threshold_consistency": threshold_consistency,
        "red_flag_consistency": red_flag_consistency,
        "planner_selection_summary": planner_selection_summary,
        "planner_behavior_summary": planner_behavior_summary,
        "candidate_cases": [asdict(c) for c in candidate_cases],
    }


if __name__ == "__main__":
    interactions_source = Path("SystemCode/logs/interactions")
    feedback_source = Path("SystemCode/logs/feedback")

    results = run_mining(interactions_source, feedback_source)

    print(json.dumps(results, indent=2, ensure_ascii=False))