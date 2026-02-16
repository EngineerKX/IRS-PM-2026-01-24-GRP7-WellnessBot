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


def load_records(log_dir: Path) -> List[Dict[str, Any]]:
    files = sorted(log_dir.glob("interactions_*.jsonl"))
    records: List[Dict[str, Any]] = []
    for fp in files:
        records.extend(list(iter_jsonl(fp)))
    return records


def main() -> None:
    log_dir = Path(__file__).resolve().parents[1] / "logs"
    if not log_dir.exists():
        print(f"No logs directory found at: {log_dir}")
        return

    records = load_records(log_dir)
    if not records:
        print(f"No log records found under: {log_dir}")
        return

    action_ctr = Counter()
    nlu_source_ctr = Counter()
    rule_ctr = Counter()
    missing_ctr = Counter()

    for r in records:
        decision = r.get("decision") or {}
        action = decision.get("action") or "UNKNOWN"
        action_ctr[action] += 1

        nlu = r.get("nlu") or {}
        nlu_source = nlu.get("nlu_source") or "UNKNOWN"
        nlu_source_ctr[nlu_source] += 1

        audit = r.get("audit_trace") or {}
        state = audit.get("state") or {}
        for m in state.get("missing_fields") or []:
            missing_ctr[m] += 1

        for fired in audit.get("rules_fired") or []:
            rid = fired.get("rule_id") or "UNKNOWN_RULE"
            rule_ctr[rid] += 1

    n = len(records)
    print(f"Total interactions: {n}\n")

    print("Decision distribution:")
    for k, v in action_ctr.most_common():
        print(f"  {k:10s} {v:5d}  ({v/n:.1%})")
    print()

    print("NLU source distribution:")
    for k, v in nlu_source_ctr.most_common():
        print(f"  {k:15s} {v:5d}  ({v/n:.1%})")
    print()

    print("Top fired rules:")
    for k, v in rule_ctr.most_common(10):
        print(f"  {k:24s} {v:5d}")
    print()

    print("Most common missing fields (from state):")
    for k, v in missing_ctr.most_common(10):
        print(f"  {k:24s} {v:5d}")
    print()


if __name__ == "__main__":
    main()