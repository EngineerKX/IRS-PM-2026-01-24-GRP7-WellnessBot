from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from wellnessbot.kg.loader import Protocol, load_protocols

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_PROTOCOLS: Dict[str, Protocol] = load_protocols(_DATA_DIR)

_EVENT_TO_PROTOCOL = {p.event_type: pid for pid, p in _PROTOCOLS.items()}


# ---- Mock-Compatible Exercise Wrapper ----
@dataclass
class CompatibleExercise:
    exercise_id: str
    name: str
    allowed_phases: List[str]
    min_pain_max: int
    swelling_allowed: List[str]
    source_refs: List[str]


def get_protocol_for_event(event_type: str) -> Optional[Protocol]:
    pid = _EVENT_TO_PROTOCOL.get(event_type)
    return _PROTOCOLS.get(pid) if pid else None


def phase_from_weeks(weeks: float, event_type: str) -> str:
    proto = get_protocol_for_event(event_type)
    if not proto:
        if not _PROTOCOLS:
            raise RuntimeError(
                f"No protocols loaded. Expected YAML files under: {_DATA_DIR / 'kg' / 'protocols'}"
            )
        proto = next(iter(_PROTOCOLS.values()))

    for ph in proto.phases:
        if ph.week_min <= weeks < ph.week_max:
            return ph.phase_id

    return proto.phases[-1].phase_id


def resolve_exercise_id(requested_exercise_text: str, event_type: str) -> Optional[str]:
    proto = get_protocol_for_event(event_type)
    if not proto:
        return None
    return proto.exercise_aliases.get((requested_exercise_text or "").strip().lower())


def _extract_constraint_from_constraint_list(constraints: List[Any], ctype: str) -> Tuple[Optional[Any], List[str]]:
    """
    Returns (value, evidence_refs) from a list of Constraint objects.
    Evidence refs are normalized as 'SRC_ID#pNN'.
    """
    for c in constraints or []:
        if getattr(c, "type", None) == ctype:
            ev = [f"{e.source_id}#p{e.page}" for e in getattr(c, "evidence", [])]
            return getattr(c, "value", None), ev
    return None, []


def _swelling_rank(level: str) -> int:
    order = ["none", "mild", "moderate", "severe", "unknown"]
    try:
        return order.index(level)
    except ValueError:
        return order.index("unknown")


def get_exercise(exercise_id: str, event_type: str) -> Optional[CompatibleExercise]:
    proto = get_protocol_for_event(event_type)
    if not proto:
        return None

    ex = proto.exercises.get(exercise_id)
    if not ex:
        return None

    # 1) Allowed phases: union base + variants
    base_allowed = list(getattr(ex, "allowed_in", []) or [])
    variant_allowed: List[str] = []
    for v in getattr(ex, "variants", []) or []:
        variant_allowed.extend(list(getattr(v, "allowed_in", []) or []))
    allowed_phases = sorted(set(base_allowed + variant_allowed))

    # 2) Constraints: consider both base and variants (compatibility, conservative)
    # Pain: choose the lowest pain_max (most conservative) among all definitions
    pain_vals: List[int] = []
    ev_pain_all: List[str] = []

    val, ev = _extract_constraint_from_constraint_list(getattr(ex, "constraints", []) or [], "pain_max")
    if val is not None:
        try:
            pain_vals.append(int(val))
        except Exception:
            pass
        ev_pain_all.extend(ev)

    for v in getattr(ex, "variants", []) or []:
        vval, vev = _extract_constraint_from_constraint_list(getattr(v, "constraints", []) or [], "pain_max")
        if vval is not None:
            try:
                pain_vals.append(int(vval))
            except Exception:
                pass
            ev_pain_all.extend(vev)

    min_pain_max = min(pain_vals) if pain_vals else 10  # safe fallback

    # Swelling: choose the most conservative swelling_max (lowest rank)
    swelling_vals: List[str] = []
    ev_sw_all: List[str] = []

    sval, sev = _extract_constraint_from_constraint_list(getattr(ex, "constraints", []) or [], "swelling_max")
    if sval is not None:
        swelling_vals.append(str(sval))
        ev_sw_all.extend(sev)

    for v in getattr(ex, "variants", []) or []:
        vsval, vsev = _extract_constraint_from_constraint_list(getattr(v, "constraints", []) or [], "swelling_max")
        if vsval is not None:
            swelling_vals.append(str(vsval))
            ev_sw_all.extend(vsev)

    swelling_levels = ["none", "mild", "moderate", "severe", "unknown"]
    if swelling_vals:
        # most conservative = smallest rank (none is strictest)
        most_conservative = min(swelling_vals, key=_swelling_rank)
        idx = _swelling_rank(most_conservative)
        swelling_allowed = swelling_levels[: idx + 1]
    else:
        swelling_allowed = swelling_levels  # no swelling constraint

    # 3) Evidence refs: union, dedup, stable order
    source_refs = sorted(set(ev_pain_all + ev_sw_all))

    return CompatibleExercise(
        exercise_id=getattr(ex, "exercise_id"),
        name=getattr(ex, "name"),
        allowed_phases=allowed_phases,
        min_pain_max=min_pain_max,
        swelling_allowed=swelling_allowed,
        source_refs=source_refs,
    )