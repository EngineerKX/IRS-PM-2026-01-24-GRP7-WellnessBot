from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


_WEIGHT_MAP = {
    "non_weight": "none",
    "none": "none",
    "partial": "partial",
    "full": "full",
}


@dataclass(frozen=True)
class Evidence:
    source_id: str
    page: int


@dataclass(frozen=True)
class Constraint:
    type: str
    value: Any
    evidence: List[Evidence]


@dataclass(frozen=True)
class Phase:
    phase_id: str
    name: str
    week_min: float
    week_max: float
    source_id: str = ""
    phase_goal: List[str] | None = None


@dataclass(frozen=True)
class ExerciseVariant:
    variant_id: str
    name: str
    allowed_in: List[str]
    constraints: List[Constraint]


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    source_id: str


@dataclass(frozen=True)
class Exercise:
    exercise_id: str
    name: str
    priority: int
    equipment_required: List[str]
    position: str
    caution: str
    allowed_in: List[str]
    constraints: List[Constraint]
    variants: List[ExerciseVariant]
    evidence_chunks: List[EvidenceChunk]


@dataclass(frozen=True)
class RedFlagPolicy:
    redflag_id: str
    phase_ids: List[str]
    symptom: str
    severity: str
    action: str
    action_steps: List[str]
    message: str


@dataclass(frozen=True)
class SelfCareAction:
    selfcare_id: str
    phase_id: str
    swell_level: str
    care_type: str
    duration_minutes: int
    frequency_condition: str


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    event_type: str
    version: str
    procedure: Dict[str, Any]
    phases: List[Phase]
    exercise_aliases: Dict[str, str]
    exercises: Dict[str, Exercise]
    sources: Dict[str, str]
    redflag_policies: List[RedFlagPolicy]
    selfcare_actions: List[SelfCareAction]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _normalize_equipment_list(items: List[Any] | None) -> List[str]:
    if not items:
        return []

    normalized: List[str] = []
    for item in items:
        text = str(item).strip().lower()
        if not text:
            continue
        normalized.append(text.replace(" ", "_"))
    return normalized


def _normalize_weight_value(value: Any) -> Any:
    if value is None:
        return value
    text = str(value).strip().lower()
    return _WEIGHT_MAP.get(text, text)


def load_protocols(data_dir: Path) -> Dict[str, Protocol]:
    protocols_dir = data_dir / "kg" / "protocols"
    protocol_files = sorted(protocols_dir.glob("*.yaml"))

    out: Dict[str, Protocol] = {}

    for fp in protocol_files:
        raw = _load_yaml(fp)

        phases = [
            Phase(
                phase_id=p["phase_id"],
                name=p["name"],
                week_min=float(p["week_min"]),
                week_max=float(p["week_max"]),
                source_id=str(p.get("source_id", "") or ""),
                phase_goal=list(p.get("phase_goal", []) or []),
            )
            for p in raw.get("phases", [])
        ]

        def parse_evidence(evs: List[Dict[str, Any]]) -> List[Evidence]:
            return [
                Evidence(
                    source_id=str(e["source_id"]),
                    page=int(e.get("page", 0)),
                )
                for e in (evs or [])
            ]

        def parse_constraints(cs: List[Dict[str, Any]]) -> List[Constraint]:
            parsed: List[Constraint] = []
            for c in cs or []:
                ctype = str(c["type"])
                value = c.get("value")
                if ctype == "weight_bearing":
                    value = _normalize_weight_value(value)

                parsed.append(
                    Constraint(
                        type=ctype,
                        value=value,
                        evidence=parse_evidence(c.get("evidence", [])),
                    )
                )
            return parsed

        def parse_evidence_chunks(items: List[Dict[str, Any]]) -> List[EvidenceChunk]:
            return [
                EvidenceChunk(
                    chunk_id=str(item["chunk_id"]),
                    source_id=str(item["source_id"]),
                )
                for item in (items or [])
            ]

        exercises: Dict[str, Exercise] = {}
        for e in raw.get("exercises", []):
            variants: List[ExerciseVariant] = []
            for v in e.get("variants", []) or []:
                variants.append(
                    ExerciseVariant(
                        variant_id=v["variant_id"],
                        name=v["name"],
                        allowed_in=list(v.get("allowed_in", [])),
                        constraints=parse_constraints(v.get("constraints", [])),
                    )
                )

            ex = Exercise(
                exercise_id=str(e["exercise_id"]),
                name=str(e["name"]),
                priority=int(e.get("priority", 5)),
                equipment_required=_normalize_equipment_list(
                    e.get("equipment_required", [])
                ),
                position=str(e.get("position", "") or ""),
                caution=str(e.get("caution", "") or ""),
                allowed_in=list(e.get("allowed_in", [])),
                constraints=parse_constraints(e.get("constraints", [])),
                variants=variants,
                evidence_chunks=parse_evidence_chunks(e.get("evidence_chunks", [])),
            )
            exercises[ex.exercise_id] = ex

        redflag_policies: List[RedFlagPolicy] = []
        for r in raw.get("redflag_policies", []) or []:
            redflag_policies.append(
                RedFlagPolicy(
                    redflag_id=str(r["redflag_id"]),
                    phase_ids=list(r.get("phase_ids", []) or []),
                    symptom=str(r["symptom"]),
                    severity=str(r.get("severity", "any")),
                    action=str(r["action"]),
                    action_steps=list(r.get("action_steps", []) or []),
                    message=str(r.get("message", "") or ""),
                )
            )

        selfcare_actions: List[SelfCareAction] = []
        for s in raw.get("selfcare_actions", []) or []:
            selfcare_actions.append(
                SelfCareAction(
                    selfcare_id=str(s["selfcare_id"]),
                    phase_id=str(s["phase_id"]),
                    swell_level=str(s.get("swell_level", "any")),
                    care_type=str(s["care_type"]),
                    duration_minutes=int(s.get("duration_minutes", 0)),
                    frequency_condition=str(s.get("frequency_condition", "") or ""),
                )
            )

        proto = Protocol(
            protocol_id=str(raw["protocol_id"]),
            event_type=str(raw["event_type"]),
            version=str(raw.get("version", "1.0")),
            procedure=dict(raw.get("procedure", {}) or {}),
            phases=phases,
            exercise_aliases={
                str(k).lower(): str(v)
                for k, v in (raw.get("exercise_aliases") or {}).items()
            },
            exercises=exercises,
            sources={
                str(k): str(v) for k, v in (raw.get("sources", {}) or {}).items()
            },
            redflag_policies=redflag_policies,
            selfcare_actions=selfcare_actions,
        )
        out[proto.protocol_id] = proto

    return out