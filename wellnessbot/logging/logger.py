from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _default_logs_dir() -> Path:
    # repo_root/logs by default
    return Path(__file__).resolve().parents[2] / "logs"


def _safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def log_interaction(
    result: Dict[str, Any],
    *,
    feedback: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Path] = None,
) -> Path:
    """
    Append one interaction record to a JSONL file.

    - result is the return value from run_pipeline()
    - feedback is optional (e.g., {"thumb": "up"/"down", "comment": "..."}), can be added later
    - LOGGING_ENABLED=0 disables logging
    - LOG_FILE can override output filename
    - LOG_DIR can override output directory
    """
    if os.getenv("LOGGING_ENABLED", "1").strip() == "0":
        return Path("")

    out_dir = Path(os.getenv("LOG_DIR", "")).expanduser() if os.getenv("LOG_DIR") else None
    out_dir = out_dir if out_dir else (log_dir if log_dir else _default_logs_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    # Default file: logs/interactions_YYYY-MM-DD.jsonl (local UTC date)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    default_file = out_dir / f"interactions_{today}.jsonl"

    override_file = os.getenv("LOG_FILE")
    out_file = out_dir / _safe_filename(override_file) if override_file else default_file

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "user_text": result.get("user_text"),
        "nlu": result.get("nlu"),
        "decision": result.get("decision"),
        "audit_trace": result.get("audit_trace"),
        "feedback": feedback,
        "meta": {
            "app_version": os.getenv("APP_VERSION", "v2"),
            "logging_version": "1.0",
        },
    }

    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_file

def log_feedback(
    *,
    interaction_ref: Dict[str, Any],
    thumb: str,
    comment: Optional[str] = None,
    expected_action: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> Path:
    """
    Append user feedback to a separate JSONL file.
    This keeps logs append-only and avoids editing past interaction records.
    """
    if os.getenv("LOGGING_ENABLED", "1").strip() == "0":
        return Path("")

    out_dir = Path(os.getenv("LOG_DIR", "")).expanduser() if os.getenv("LOG_DIR") else None
    out_dir = out_dir if out_dir else (log_dir if log_dir else _default_logs_dir())
    out_dir.mkdir(parents=True, exist_ok=True)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_file = out_dir / f"feedback_{today}.jsonl"

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interaction_ref": interaction_ref,
        "feedback": {
            "thumb": thumb,  # "up" or "down"
            "comment": comment,
            "expected_action": expected_action,  # optional
        },
        "meta": {
            "app_version": os.getenv("APP_VERSION", "v2"),
            "feedback_version": "1.0",
        },
    }

    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_file