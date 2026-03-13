from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@dataclass(frozen=True)
class ExerciseVariant:
    variant_id: str
    name: str
    allowed_in: List[str]
    constraints: List[Constraint]


@dataclass(frozen=True)
class Exercise:
    exercise_id: str
    name: str
    allowed_in: List[str]
    constraints: List[Constraint]
    variants: List[ExerciseVariant]


@dataclass(frozen=True)
class Protocol:
    protocol_id: str
    event_type: str
    version: str
    phases: List[Phase]
    exercise_aliases: Dict[str, str]
    exercises: Dict[str, Exercise]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
            )
            for p in raw.get("phases", [])
        ]

        def parse_evidence(evs: List[Dict[str, Any]]) -> List[Evidence]:
            return [Evidence(source_id=e["source_id"], page=int(e["page"])) for e in evs]

        def parse_constraints(cs: List[Dict[str, Any]]) -> List[Constraint]:
            out_cs: List[Constraint] = []
            for c in cs or []:
                out_cs.append(
                    Constraint(
                        type=c["type"],
                        value=c.get("value"),
                        evidence=parse_evidence(c.get("evidence", [])),
                    )
                )
            return out_cs

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
                exercise_id=e["exercise_id"],
                name=e["name"],
                allowed_in=list(e.get("allowed_in", [])),
                constraints=parse_constraints(e.get("constraints", [])),
                variants=variants,
            )
            exercises[ex.exercise_id] = ex

        proto = Protocol(
            protocol_id=raw["protocol_id"],
            event_type=raw["event_type"],
            version=str(raw.get("version", "1.0")),
            phases=phases,
            exercise_aliases={k.lower(): v for k, v in (raw.get("exercise_aliases") or {}).items()},
            exercises=exercises,
        )
        out[proto.protocol_id] = proto

    return out