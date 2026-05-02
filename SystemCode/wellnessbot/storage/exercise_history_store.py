from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

HISTORY_DIR = Path(__file__).resolve().parents[2] / "data" / "exercise_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _history_path(profile_id: str) -> Path:
    safe_id = profile_id.strip().lower().replace(" ", "_")
    return HISTORY_DIR / f"{safe_id}.json"


def load_exercise_history(profile_id: str) -> List[Dict]:
    if not profile_id:
        return []

    path = _history_path(profile_id)
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []

    return data


def save_exercise_history(profile_id: str, history: List[Dict]) -> None:
    if not profile_id:
        return

    path = _history_path(profile_id)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def append_exercise_history(profile_id: str, record: Dict, keep_last: int = 50) -> None:
    if not profile_id:
        return

    history = load_exercise_history(profile_id)
    history.append(record)

    if keep_last > 0:
        history = history[-keep_last:]

    save_exercise_history(profile_id, history)


def clear_exercise_history(profile_id: str) -> None:
    if not profile_id:
        return

    path = _history_path(profile_id)
    if path.exists():
        path.unlink()