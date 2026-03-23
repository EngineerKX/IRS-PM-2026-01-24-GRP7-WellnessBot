from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

PROFILE_DIR = Path("data/profiles")
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


def _profile_path(profile_id: str) -> Path:
    safe_id = profile_id.strip().lower().replace(" ", "_")
    return PROFILE_DIR / f"{safe_id}.json"


def default_profile(profile_id: str, display_name: str = "") -> Dict:
    return {
        "profile_id": profile_id,
        "display_name": display_name,
        "event_type": "unknown",
        "surgery_date": "",
        "equipment_available": [],
        "exercise_history": [],
    }


def save_profile(profile: Dict) -> None:
    profile_id = profile["profile_id"]
    path = _profile_path(profile_id)
    path.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def load_profile(profile_id: str) -> Dict:
    path = _profile_path(profile_id)
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {profile_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def create_profile(profile_id: str, display_name: str = "") -> Dict:
    path = _profile_path(profile_id)
    if path.exists():
        raise FileExistsError(f"Profile already exists: {profile_id}")
    profile = default_profile(profile_id, display_name)
    save_profile(profile)
    return profile


def list_profiles() -> List[str]:
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def delete_profile(profile_id: str) -> None:
    path = _profile_path(profile_id)
    if path.exists():
        path.unlink()