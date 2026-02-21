from __future__ import annotations

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# ------------------------------------------------------------
# Interaction ID
# ------------------------------------------------------------

def make_interaction_id(user_text: str, audit_ts: str) -> str:
    raw = f"{audit_ts}|{user_text}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# ------------------------------------------------------------
# Directory Helpers
# ------------------------------------------------------------

def _default_logs_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "logs"


def _safe_filename(name: str) -> str:
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def _resolve_log_dir(log_dir: Optional[Path]) -> Path:
    env_dir = os.getenv("LOG_DIR")
    out_dir = Path(env_dir).expanduser() if env_dir else None
    out_dir = out_dir if out_dir else (log_dir if log_dir else _default_logs_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


# ------------------------------------------------------------
# Interaction Logging
# ------------------------------------------------------------

def log_interaction(
    result: Dict[str, Any],
    *,
    log_dir: Optional[Path] = None,
) -> Path:
    """
    Append one interaction record to JSONL.

    Disabled if LOGGING_ENABLED=0
    """

    if os.getenv("LOGGING_ENABLED", "1").strip() == "0":
        return Path("")

    out_dir = _resolve_log_dir(log_dir)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    default_file = out_dir / f"interactions_{today}.jsonl"

    override_file = os.getenv("LOG_FILE")
    out_file = out_dir / _safe_filename(override_file) if override_file else default_file

    timestamp_utc = datetime.now(timezone.utc).isoformat()

    audit_trace = result.get("audit_trace") or {}
    audit_ts = audit_trace.get("timestamp_utc") or timestamp_utc
    user_text = result.get("user_text") or ""

    interaction_id = make_interaction_id(user_text, audit_ts)

    record = {
        "timestamp_utc": timestamp_utc,
        "interaction_id": interaction_id,
        "user_text": user_text,
        "nlu": result.get("nlu"),
        "decision": result.get("decision"),
        "audit_trace": audit_trace,
        "meta": {
            "app_version": os.getenv("APP_VERSION", "v2"),
            "logging_version": "1.1",
        },
    }

    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_file


# ------------------------------------------------------------
# Feedback Logging
# ------------------------------------------------------------

def log_feedback(
    *,
    interaction_id: str,
    thumb: str,
    comment: Optional[str] = None,
    expected_action: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> Path:
    """
    Append feedback to separate JSONL.
    """

    if os.getenv("LOGGING_ENABLED", "1").strip() == "0":
        return Path("")

    if not interaction_id:
        # Safety guard
        raise ValueError("interaction_id is required for feedback logging")

    out_dir = _resolve_log_dir(log_dir)

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_file = out_dir / f"feedback_{today}.jsonl"

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "interaction_id": interaction_id,
        "feedback": {
            "thumb": thumb,
            "comment": comment,
            "expected_action": expected_action,
        },
        "meta": {
            "app_version": os.getenv("APP_VERSION", "v2"),
            "feedback_version": "1.1",
        },
    }

    with out_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return out_file