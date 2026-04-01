from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from wellnessbot.kg.loader import (
    EvidenceChunk,
    Protocol,
    load_protocols,
    RedFlagPolicy,
    SelfCareAction,
)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_PROTOCOLS: Dict[str, Protocol] = load_protocols(_DATA_DIR)

_SURGERY_TO_PROTOCOL = {p.surgery_type: pid for pid, p in _PROTOCOLS.items()}


@dataclass
class CompatibleExercise:
    exercise_id: str
    name: str
    priority: int
    equipment_required: List[str]
    position: str
    caution: str
    allowed_phases: List[str]
    min_pain_max: int
    swelling_allowed: List[str]
    weight_bearing_allowed: List[str]
    source_refs: List[str]
    evidence_chunks: List[EvidenceChunk]


def get_protocol_for_surgery_type(surgery_type: str) -> Optional[Protocol]:
    pid = _SURGERY_TO_PROTOCOL.get(surgery_type)
    return _PROTOCOLS.get(pid) if pid else None


# Backward-compatible alias during migration
def get_protocol_for_event(event_type: str) -> Optional[Protocol]:
    return get_protocol_for_surgery_type(event_type)


def phase_from_weeks(weeks: float, surgery_type: str) -> str:
    proto = get_protocol_for_surgery_type(surgery_type)
    if not proto:
        if not _PROTOCOLS:
            raise RuntimeError(
                f"No protocols loaded. Expected YAML files under: {_DATA_DIR / 'kg' / 'protocols'}"
            )
        proto = next(iter(_PROTOCOLS.values()))

    for ph in proto.phases:
        if ph.week_min <= weeks < ph.week_max:
            return ph.phase_id

    # below first phase -> first phase
    if weeks < proto.phases[0].week_min:
        return proto.phases[0].phase_id

    # beyond last phase -> last phase
    if weeks >= proto.phases[-1].week_min:
        return proto.phases[-1].phase_id

    raise ValueError(
        f"Weeks value {weeks} did not match any phase range for protocol {proto.protocol_id}. "
        f"Check YAML phase ranges for gaps or overlaps."
    )


def resolve_exercise_id(requested_exercise_text: str, surgery_type: str) -> Optional[str]:
    proto = get_protocol_for_surgery_type(surgery_type)
    if not proto:
        return None
    return proto.exercise_aliases.get((requested_exercise_text or "").strip().lower())


def _extract_constraint_from_constraint_list(
    constraints: List[Any], ctype: str
) -> Tuple[Optional[Any], List[str]]:
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


def get_exercise(exercise_id: str, surgery_type: str) -> Optional[CompatibleExercise]:
    proto = get_protocol_for_surgery_type(surgery_type)
    if not proto:
        return None

    ex = proto.exercises.get(exercise_id)
    if not ex:
        return None

    # Allowed phases: union base + variants
    base_allowed = list(getattr(ex, "allowed_in", []) or [])
    variant_allowed: List[str] = []
    for v in getattr(ex, "variants", []) or []:
        variant_allowed.extend(list(getattr(v, "allowed_in", []) or []))
    allowed_phases = sorted(set(base_allowed + variant_allowed))

    # Pain constraints
    pain_vals: List[int] = []
    ev_pain_all: List[str] = []

    val, ev = _extract_constraint_from_constraint_list(
        getattr(ex, "constraints", []) or [], "pain_max"
    )
    if val is not None:
        try:
            pain_vals.append(int(val))
        except Exception:
            pass
        ev_pain_all.extend(ev)

    for v in getattr(ex, "variants", []) or []:
        vval, vev = _extract_constraint_from_constraint_list(
            getattr(v, "constraints", []) or [], "pain_max"
        )
        if vval is not None:
            try:
                pain_vals.append(int(vval))
            except Exception:
                pass
            ev_pain_all.extend(vev)

    min_pain_max = min(pain_vals) if pain_vals else 10

    # Swelling constraints
    swelling_vals: List[str] = []
    ev_sw_all: List[str] = []

    sval, sev = _extract_constraint_from_constraint_list(
        getattr(ex, "constraints", []) or [], "swelling_max"
    )
    if sval is not None:
        swelling_vals.append(str(sval))
        ev_sw_all.extend(sev)

    for v in getattr(ex, "variants", []) or []:
        vsval, vsev = _extract_constraint_from_constraint_list(
            getattr(v, "constraints", []) or [], "swelling_max"
        )
        if vsval is not None:
            swelling_vals.append(str(vsval))
            ev_sw_all.extend(vsev)

    swelling_levels = ["none", "mild", "moderate", "severe", "unknown"]
    if swelling_vals:
        most_conservative = min(swelling_vals, key=_swelling_rank)
        idx = _swelling_rank(most_conservative)
        swelling_allowed = swelling_levels[: idx + 1]
    else:
        swelling_allowed = swelling_levels

    # Weight-bearing constraints
    wb_vals: List[str] = []
    ev_wb_all: List[str] = []

    wval, wev = _extract_constraint_from_constraint_list(
        getattr(ex, "constraints", []) or [], "weight_bearing"
    )
    if wval is not None:
        wb_vals.append(str(wval))
        ev_wb_all.extend(wev)

    for v in getattr(ex, "variants", []) or []:
        vwval, vwev = _extract_constraint_from_constraint_list(
            getattr(v, "constraints", []) or [], "weight_bearing"
        )
        if vwval is not None:
            wb_vals.append(str(vwval))
            ev_wb_all.extend(vwev)

    if wb_vals:
        weight_bearing_allowed = sorted(set(wb_vals))
    else:
        weight_bearing_allowed = ["none", "partial", "full"]

    # Evidence refs: constraints + evidence chunks
    raw_chunks: List[EvidenceChunk] = list(getattr(ex, "evidence_chunks", []) or [])
    chunk_refs = [f"{ec.source_id}#{ec.chunk_id}" for ec in raw_chunks]
    source_refs = sorted(set(ev_pain_all + ev_sw_all + ev_wb_all + chunk_refs))

    return CompatibleExercise(
        exercise_id=ex.exercise_id,
        name=ex.name,
        priority=ex.priority,
        equipment_required=ex.equipment_required,
        position=ex.position,
        caution=ex.caution,
        allowed_phases=allowed_phases,
        min_pain_max=min_pain_max,
        swelling_allowed=swelling_allowed,
        weight_bearing_allowed=weight_bearing_allowed,
        source_refs=source_refs,
        evidence_chunks=raw_chunks,
    )


def list_exercises_for_phase(surgery_type: str, phase_id: str) -> List[CompatibleExercise]:
    proto = get_protocol_for_surgery_type(surgery_type)
    if not proto:
        return []

    out: List[CompatibleExercise] = []
    for ex_id in proto.exercises.keys():
        ex = get_exercise(ex_id, surgery_type)
        if not ex:
            continue
        if phase_id in (ex.allowed_phases or []):
            out.append(ex)

    out.sort(key=lambda e: (e.priority, e.name.lower(), e.exercise_id))
    return out


def get_redflag_policies(surgery_type: str, phase_id: str) -> List[RedFlagPolicy]:
    proto = get_protocol_for_surgery_type(surgery_type)
    if not proto:
        return []
    return [p for p in proto.redflag_policies if phase_id in p.phase_ids]


def get_selfcare_actions(surgery_type: str, phase_id: str) -> List[SelfCareAction]:
    proto = get_protocol_for_surgery_type(surgery_type)
    if not proto:
        return []
    return [a for a in proto.selfcare_actions if a.phase_id == phase_id]