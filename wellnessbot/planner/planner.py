from __future__ import annotations

import logging
from typing import Dict, List

from wellnessbot.kg.kg import CompatibleExercise, list_exercises_for_phase
from wellnessbot.nlu.schema import NLUOutput

logger = logging.getLogger(__name__)


def _equipment_compatible(ex: CompatibleExercise, equipment_available: List[str]) -> bool:
    """
    An exercise is compatible if:
    - It requires no equipment, OR
    - ALL of its required equipment is in equipment_available.
    """
    if not ex.equipment_required:
        return True
    return all(req in equipment_available for req in ex.equipment_required)



def _working_priority_and_seen(
    history: List[dict],
    phase_id: str,
    candidates: List[CompatibleExercise],
) -> tuple[int, set]:
    """
    Determine the current working priority tier and the set of exercise IDs
    already seen within the current tier run for this phase.

    Algorithm — scan backwards through phase history:
      1. The last entry tells us the current working priority.
      2. Keep scanning back as long as entries are at the same priority —
         these are exercises already done in this tier run.
      3. Stop when we hit an entry at a LOWER priority number — that is the
         tier boundary (we just advanced to this tier from the one below).
      4. The seen set contains only IDs collected since that boundary.

    This means after completing a full cycle (e.g. P3 priority-1 → 2 → 3 → done),
    and re-entering the phase, the seen set resets correctly for the new cycle.

    If no phase history exists, returns (lowest priority tier, empty set).
    If all tiers are exhausted (last tier completed with nothing left), returns
    (highest priority tier, full seen set) — caller handles PHASE_COMPLETE.
    """
    sorted_priorities = sorted({ex.priority for ex in candidates})
    valid_ids = {ex.exercise_id for ex in candidates}

    # Only keep history entries whose exercise_id actually belongs to this phase's pool.
    # This guards against stale cross-phase exercise IDs that were recorded under the
    # wrong phase_id (e.g. P1_1_E2 stored with phase_id=P3 from a pain-downgrade session).
    phase_entries = [
        e for e in history
        if e.get("phase_id") == phase_id
        and e.get("exercise_id") in valid_ids
    ]

    if not phase_entries:
        return sorted_priorities[0], set()

    # Scan backwards to find the current tier run
    current_priority = phase_entries[-1].get("priority", sorted_priorities[0])
    seen_in_tier: set = set()

    for entry in reversed(phase_entries):
        entry_priority = entry.get("priority", current_priority)
        if entry_priority != current_priority:
            # Hit an entry from a different tier — this is the tier boundary; stop here
            break
        seen_in_tier.add(entry["exercise_id"])

    logger.debug(
        "_working_priority_and_seen | phase=%s current_priority=%d seen_in_tier=%s",
        phase_id, current_priority, sorted(seen_in_tier),
    )

    # Check if there are unseen exercises at the current priority tier
    tier_ids = {ex.exercise_id for ex in candidates if ex.priority == current_priority}
    unseen_in_tier = tier_ids - seen_in_tier

    if unseen_in_tier:
        # Still work to do in this tier
        return current_priority, seen_in_tier

    # Current tier is fully done — advance to the next tier
    idx = sorted_priorities.index(current_priority)
    if idx + 1 < len(sorted_priorities):
        next_priority = sorted_priorities[idx + 1]
        logger.debug(
            "_working_priority_and_seen | tier %d complete, advancing to %d",
            current_priority, next_priority,
        )
        return next_priority, set()  # fresh seen set for the new tier

    # All tiers exhausted — return highest priority with full seen set
    # Caller will handle PHASE_COMPLETE
    return sorted_priorities[-1], seen_in_tier


def _unseen_at_priority(
    candidates: List[CompatibleExercise],
    priority: int,
    seen: set,
) -> List[CompatibleExercise]:
    """Return unseen exercises at the given priority, sorted deterministically by exercise_id."""
    unseen = [ex for ex in candidates if ex.priority == priority and ex.exercise_id not in seen]
    unseen.sort(key=lambda ex: ex.exercise_id)
    return unseen


def _downgrade_priority(
    current_phase_id: str,
    working_priority: int,
    sorted_priorities: List[int],
) -> Dict:
    """
    Given mild pain, determine the downgraded priority/phase target.

    Returns a dict with:
        {"cross_phase": bool, "phase_id": str (only if cross_phase), "priority": int}

    Downgrade rules:
    - Priority 2 or 3 → drop one tier within the same phase
    - Priority 1 in a non-P1 phase → cross-phase drop to P1 (both sub-phases),
      targeting the highest priority number that exists there
    - Priority 1 in a P1 phase → already at absolute easiest; no further downgrade
    """
    if working_priority > sorted_priorities[0]:
        # Drop one tier within the same phase
        idx = sorted_priorities.index(working_priority)
        lower_priority = sorted_priorities[idx - 1]
        return {"cross_phase": False, "priority": lower_priority}

    # Already at the lowest priority tier in the current phase
    if current_phase_id.startswith("P1"):
        # Already at the absolute easiest — stay here
        return {"cross_phase": False, "priority": sorted_priorities[0]}
    else:
        # Cross-phase drop: signal the caller to query P1 and find the highest priority there
        return {"cross_phase": True}


def _select_exercise(
    candidates: List[CompatibleExercise],
    nlu: NLUOutput,
    phase_id: str,
    equipment_available: List[str],
    history: List[dict],
    surgery_type: str,
) -> Dict:
    """
    Core selection logic. Returns a result dict with either a selected exercise
    or an outcome indicating completion/unavailability.

    Stage 1 — Equipment filter
        Keep exercises whose required equipment is all available to the user.
        If nothing survives, return no-equipment subset or full pool as fallback.

    Stage 2 — Priority-Based Progression
        Determine current working priority tier from history.
        Select the first unseen exercise at that tier.

    Stage 3 — Pain downgrade (pain_score == 1 only)
        If user has mild pain, downgrade working priority by one tier.
        Add recommend_painkiller flag to output.
        Pain == 0: proceed normally.
        Pain >= 2: handled by rule engine (not planner).
    """

    # --- Stage 1: Equipment filter ---
    compatible = [ex for ex in candidates if _equipment_compatible(ex, equipment_available)]
    if compatible:
        pool = compatible
        logger.debug("Equipment filter: %d/%d compatible", len(pool), len(candidates))
    else:
        pool = candidates
        logger.debug("Equipment filter: no compatible exercises, using full pool as fallback")

    sorted_priorities = sorted({ex.priority for ex in pool})
    working_priority, seen = _working_priority_and_seen(history, phase_id, pool)
    pain_score = nlu.pain_score or 0

    logger.debug(
        "Stage 2 | phase=%s sorted_priorities=%s working_priority=%d seen_in_current_tier=%s",
        phase_id, sorted_priorities, working_priority, sorted(seen),
    )
    logger.debug("Stage 3 | pain_score=%d swelling_score=%s", pain_score, nlu.swelling_score)

    recommend_painkiller = False

    # --- Stage 3: Pain downgrade (mild pain only) ---
    if pain_score == 1:
        recommend_painkiller = True
        downgrade = _downgrade_priority(phase_id, working_priority, sorted_priorities)
        logger.debug("Pain downgrade decision | %s", downgrade)

        if downgrade.get("cross_phase"):
            # Cross-phase drop: query all P1 sub-phases, combine, find highest priority tier
            p1_all: List[CompatibleExercise] = []
            for p1_phase in ("P1_1", "P1_2"):
                p1_all.extend(list_exercises_for_phase(surgery_type, p1_phase))

            p1_compatible = [ex for ex in p1_all if _equipment_compatible(ex, equipment_available)]
            p1_pool = p1_compatible if p1_compatible else p1_all

            # Highest priority number = the hardest tier available in P1
            p1_priorities = sorted({ex.priority for ex in p1_pool})
            target_priority = p1_priorities[-1] if p1_priorities else 1

            logger.debug(
                "Pain downgrade | cross-phase: %s priority %d → P1 priority %d | "
                "P1 pool=%s",
                phase_id, working_priority, target_priority,
                [ex.exercise_id for ex in p1_pool],
            )
            # Ignore history — user must redo P1 tier from scratch due to pain
            candidates_at_target = [ex for ex in p1_pool if ex.priority == target_priority]
            candidates_at_target.sort(key=lambda ex: ex.exercise_id)
        else:
            target_priority = downgrade["priority"]
            logger.debug(
                "Pain downgrade | same phase: %s priority %d → priority %d | "
                "pool ids=%s",
                phase_id, working_priority, target_priority,
                [ex.exercise_id for ex in pool],
            )
            # Ignore history for the downgraded tier — user must redo from scratch due to pain
            candidates_at_target = [
                ex for ex in pool if ex.priority == target_priority
            ]
            candidates_at_target.sort(key=lambda ex: ex.exercise_id)

        logger.debug(
            "Pain downgrade | candidates_at_target=%s",
            [ex.exercise_id for ex in candidates_at_target],
        )

        if not candidates_at_target:
            logger.warning(
                "Pain downgrade | no candidates at target priority %d — "
                "falling back to lowest priority %d in current pool",
                target_priority, sorted_priorities[0],
            )
            # Fallback: use lowest priority in current pool, also ignoring history
            candidates_at_target = [ex for ex in pool if ex.priority == sorted_priorities[0]]
            candidates_at_target.sort(key=lambda ex: ex.exercise_id)
            logger.debug(
                "Pain downgrade | fallback candidates=%s",
                [ex.exercise_id for ex in candidates_at_target],
            )

    else:
        # --- Stage 2: Normal priority-based progression ---
        target_priority = working_priority
        candidates_at_target = _unseen_at_priority(pool, target_priority, seen)
        logger.debug(
            "Normal progression | target_priority=%d unseen_candidates=%s",
            target_priority, [ex.exercise_id for ex in candidates_at_target],
        )

    if not candidates_at_target:
        logger.warning(
            "No candidates remaining | phase=%s pain_score=%d seen=%s pool=%s → PHASE_COMPLETE",
            phase_id, pain_score, sorted(seen), [ex.exercise_id for ex in pool],
        )
        # This phase is complete — all priorities exhausted
        return {"outcome": "PHASE_COMPLETE"}

    best = candidates_at_target[0]
    logger.debug(
        "Selected | exercise_id=%s name=%s priority=%d pain_score=%d recommend_painkiller=%s",
        best.exercise_id, best.name, best.priority, pain_score, recommend_painkiller,
    )

    return {"exercise": best, "recommend_painkiller": recommend_painkiller}


def plan(
    nlu: NLUOutput,
    phase_id: str,
    equipment_available: List[str] | None = None,
    exercise_history: List[dict] | None = None,
) -> Dict:
    equipment_available = equipment_available or []
    exercise_history = exercise_history or []

    logger.debug(
        "plan() | surgery_type=%s phase_id=%s equipment=%s history_len=%d",
        nlu.surgery_type,
        phase_id,
        equipment_available,
        len(exercise_history),
    )

    candidates = list_exercises_for_phase(nlu.surgery_type, phase_id)
    logger.debug(
        "KG candidates | count=%d ids=%s",
        len(candidates),
        [ex.exercise_id for ex in candidates],
    )

    if not candidates:
        logger.warning("No candidates | surgery_type=%s phase_id=%s", nlu.surgery_type, phase_id)
        return {"plan": None, "notes": ["No exercises found for this phase."]}

    result = _select_exercise(
        candidates, nlu, phase_id, equipment_available, exercise_history, nlu.surgery_type
    )

    if result.get("outcome") == "PHASE_COMPLETE":
        return {
            "plan": None,
            "outcome": "ESCALATE",
            "notes": ["All exercises in the current phase have been completed. Ready to progress."],
        }

    best: CompatibleExercise = result["exercise"]
    recommend_painkiller: bool = result["recommend_painkiller"]

    return {
        "exercise_id": best.exercise_id,
        "exercise_name": best.name,
        "phase_id": phase_id,
        "priority": best.priority,
        "position": best.position,
        "caution": best.caution,
        "equipment_required": best.equipment_required,
        "recommend_painkiller": recommend_painkiller,
        # For RAG: structured evidence chunks (chunk_id + source_id)
        "evidence_chunks": [
            {"chunk_id": ec.chunk_id, "source_id": ec.source_id}
            for ec in best.evidence_chunks
        ],
        # For display: flat citation strings
        "citations": best.source_refs,
        "stop_conditions": ["pain increases"],
    }


def plan_alternatives(surgery_type: str, phase_id: str, top_k: int = 5) -> Dict:
    logger.debug(
        "plan_alternatives() | surgery_type=%s phase_id=%s top_k=%d",
        surgery_type,
        phase_id,
        top_k,
    )

    candidates = list_exercises_for_phase(surgery_type, phase_id)
    logger.debug(
        "Alternatives | count=%d ids=%s",
        len(candidates),
        [e.exercise_id for e in candidates],
    )

    return {
        "type": "alternatives",
        "phase_id": phase_id,
        "items": [
            {
                "exercise_id": e.exercise_id,
                "name": e.name,
                "priority": e.priority,
                "caution": e.caution,
                "citations": e.source_refs,
                "evidence_chunks": [
                    {"chunk_id": ec.chunk_id, "source_id": ec.source_id}
                    for ec in e.evidence_chunks
                ],
            }
            for e in candidates[:top_k]
        ],
    }
