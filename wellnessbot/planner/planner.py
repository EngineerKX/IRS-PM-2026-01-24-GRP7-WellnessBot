from __future__ import annotations

import logging
from typing import Dict, List, Optional

from wellnessbot.kg.kg import CompatibleExercise, list_exercises_for_phase
from wellnessbot.nlu.schema import NLUOutput

logger = logging.getLogger(__name__)

# Swelling ordinal for comparisons
_SWELLING_RANK = {"none": 0, "mild": 1, "moderate": 2, "severe": 3, "unknown": -1}

# Caution keywords that map to detectable NLU conditions.
# If the caution text contains any of these keywords AND the NLU condition is true,
# the exercise is skipped.
_CAUTION_RULES = [
    # (caution_keyword, lambda nlu: True if condition applies)
    ("knee extension lag", lambda nlu: nlu.weight_bearing in ("none", "partial")),
]

_MILD_PERSISTENT_THRESHOLD = 3  # rule 5: stop after this many consecutive mild sessions
_STOP_EXERCISE_SENTINEL = "STOP_EXERCISE"


def _swelling_rank(level: str) -> int:
    return _SWELLING_RANK.get(str(level).lower(), -1)


def _caution_applies(ex: CompatibleExercise, nlu: NLUOutput) -> bool:
    """
    Returns True if the exercise caution text matches a detectable NLU condition,
    meaning the exercise should be skipped for this user right now.
    """
    if not ex.caution:
        return False
    caution_lower = ex.caution.lower()
    for keyword, condition_fn in _CAUTION_RULES:
        if keyword in caution_lower and condition_fn(nlu):
            logger.debug(
                "Caution filter | exercise=%s skipped: caution='%s' matches condition '%s'",
                ex.exercise_id,
                ex.caution,
                keyword,
            )
            return True
    return False


def _equipment_compatible(ex: CompatibleExercise, equipment_available: List[str]) -> bool:
    """
    An exercise is compatible if:
    - It requires no equipment, OR
    - ALL of its required equipment is in equipment_available.
    """
    if not ex.equipment_required:
        return True
    return all(req in equipment_available for req in ex.equipment_required)


def _phase_history_entries(history: List[dict], phase_id: str) -> List[dict]:
    """
    Return all RECOMMEND entries for the given phase, oldest-first.
    """
    result = []
    for entry in history:
        if entry.get("action") != "RECOMMEND":
            continue
        ex_phase = entry.get("phase_id")
        if ex_phase and ex_phase != phase_id:
            continue
        result.append(entry)
    return result


def _last_recommended(history: List[dict], phase_id: str) -> Optional[dict]:
    """
    Return the most recent RECOMMEND entry for this phase, or None.
    """
    phase_entries = _phase_history_entries(history, phase_id)
    return phase_entries[-1] if phase_entries else None


def _previous_swelling(history: List[dict], phase_id: str) -> Optional[str]:
    """
    Return the swelling_level from the second-most-recent RECOMMEND entry for
    this phase (the session before the current one).
    Returns None if fewer than 2 history entries exist for this phase.
    """
    phase_entries = _phase_history_entries(history, phase_id)
    if len(phase_entries) < 2:
        return None
    return phase_entries[-2].get("swelling_level")


def _consecutive_mild_swelling_count(history: List[dict], phase_id: str) -> int:
    """
    Count how many consecutive most-recent RECOMMEND entries for this phase
    all have swelling_level == 'mild'. Stops as soon as a non-mild entry is found.
    """
    count = 0
    phase_entries = _phase_history_entries(history, phase_id)
    for entry in reversed(phase_entries):
        if entry.get("swelling_level") == "mild":
            count += 1
        else:
            break
    return count


def _seen_exercise_ids_in_current_cycle(
    history: List[dict], phase_id: str, all_exercise_ids: set,
    current_swelling: str = "none",
) -> set:
    """
    Return the set of exercise IDs seen in the *current* cycle only.

    A cycle resets (seen_ids cleared) when either:
    - All exercises in all_exercise_ids have been seen at least once (full cycle complete), OR
    - Swelling was active in an entry and clears to 'none' in the next entry
      (swelling-driven downgrade ended — user must restart from priority 1).

    The current session's swelling level is passed in separately (it is not yet
    recorded in history) so the function can detect when swelling just cleared
    at the boundary between the last history entry and the current session.

    Scans phase history oldest-to-newest, accumulating seen IDs and resetting on
    either condition, so only IDs from the most recent uninterrupted run are returned.
    """
    seen: set = set()
    phase_entries = _phase_history_entries(history, phase_id)

    for i, entry in enumerate(phase_entries):
        ex_id = entry.get("exercise_id", "")
        cur_swell = str(entry.get("swelling_level", "none")).lower()

        # Reset if swelling just cleared vs the previous history entry
        if i > 0:
            prev_swell = str(phase_entries[i - 1].get("swelling_level", "none")).lower()
            if _swelling_rank(prev_swell) > 0 and cur_swell == "none":
                seen = set()

        if ex_id:
            seen.add(ex_id)

        # Full cycle complete — reset so the next entry starts a fresh cycle
        if all_exercise_ids and seen >= all_exercise_ids:
            seen = set()

    # Check the boundary: last history entry had active swelling, current session has none
    if phase_entries:
        last_swell = str(phase_entries[-1].get("swelling_level", "none")).lower()
        if _swelling_rank(last_swell) > 0 and current_swelling == "none":
            seen = set()

    return seen


def _last_recommended_priority(history: List[dict], phase_id: str) -> Optional[int]:
    """Return the priority of the most recently recommended exercise in this phase."""
    last = _last_recommended(history, phase_id)
    if last is None:
        return None
    return last.get("priority")


def _pick_least_recently_done(
    pool: List[CompatibleExercise], history: List[dict]
) -> CompatibleExercise:
    """Pick the exercise least recently seen in history (or first alphabetically if never seen)."""
    def last_seen_index(ex: CompatibleExercise) -> int:
        for i, entry in enumerate(reversed(history)):
            if entry.get("exercise_id") == ex.exercise_id:
                return i
        return len(history)  # never seen = oldest

    return sorted(pool, key=lambda ex: (-last_seen_index(ex), ex.exercise_id))[0]


def _select_exercise(
    candidates: List[CompatibleExercise],
    nlu: NLUOutput,
    phase_id: str,
    equipment_available: List[str],
    history: List[dict],
) -> Optional[CompatibleExercise] | str:
    """
    Core selection logic.

    Stage 1 — Equipment filter
        Keep exercises whose required equipment is all available to the user.
        No-equipment exercises always pass. Falls back to full pool if nothing survives.

    Stage 2 — Caution filter
        Skip exercises whose caution text matches a known NLU condition.
        Falls back to pre-filter pool if nothing survives.

    Stage 3 — Swelling trajectory decision
        Drives whether to stay, downgrade, progress, or stop:

        severe               : cap to lowest-priority available exercise     (implicit via downgrade)
        severe → moderate    : downgrade 1 priority level from last done     (rule 1)
        moderate → moderate  : downgrade 1 more priority level from last done (rule 2)
        moderate → mild      : stay at same exercise                         (rule 3)
        mild → mild          : stay at same exercise                         (rule 4)
        mild ×3+ (consec.)   : return STOP_EXERCISE sentinel                 (rule 5)
        none / cleared       : progress — next unseen in current priority,
                               or advance to next priority tier              (rules 6 & 7)

    Stage 4 — Priority-based exercise selection
        Within the eligible pool, prefer unseen exercises in priority order.
        If all exercises in a tier have been seen, rotate (least-recently-done).
        User must clear all exercises in a priority tier before advancing to next tier.
    """
    if not candidates:
        return None

    # --- Stage 1: Equipment filter ---
    compatible = [ex for ex in candidates if _equipment_compatible(ex, equipment_available)]
    pool = compatible if compatible else candidates
    if not compatible:
        logger.debug("Equipment filter: no compatible exercises, using full pool as fallback")
    else:
        logger.debug("Equipment filter: %d/%d compatible", len(pool), len(candidates))

    # --- Stage 2: Caution filter ---
    safe = [ex for ex in pool if not _caution_applies(ex, nlu)]
    if safe:
        pool = safe
    else:
        logger.debug("Caution filter: nothing survived, skipping caution filter")

    # --- Stage 3: Swelling trajectory ---
    cur_swelling = str(nlu.swelling_level).lower()
    prev_swelling = _previous_swelling(history, phase_id)
    last_priority = _last_recommended_priority(history, phase_id)

    all_exercise_ids = {ex.exercise_id for ex in pool}
    seen_ids = _seen_exercise_ids_in_current_cycle(history, phase_id, all_exercise_ids, cur_swelling)

    logger.debug(
        "Swelling trajectory | cur=%s prev=%s last_priority=%s seen=%d",
        cur_swelling, prev_swelling, last_priority, len(seen_ids),
    )

    # Rule 5: mild persisting for too many sessions → stop
    if cur_swelling == "mild":
        consecutive_mild = _consecutive_mild_swelling_count(history, phase_id)
        # +1 because current session is not yet in history
        if consecutive_mild + 1 >= _MILD_PERSISTENT_THRESHOLD:
            logger.debug(
                "Swelling trajectory | mild persisting %d+ consecutive sessions → STOP_EXERCISE",
                consecutive_mild + 1,
            )
            return _STOP_EXERCISE_SENTINEL

    sorted_priorities = sorted({ex.priority for ex in pool})

    # --- Swelling-present modes (rules 1–4) ---

    if cur_swelling == "severe":
        # Severe: use the safest (lowest priority number = highest clinical importance) exercise.
        lowest_priority = sorted_priorities[0]
        tier = [ex for ex in pool if ex.priority == lowest_priority]
        result = _pick_least_recently_done(tier, history)
        logger.debug("Swelling severe | pinning to priority=%d → %s", lowest_priority, result.exercise_id)
        return result

    if cur_swelling == "moderate":
        # Rules 1 & 2: drop 1 priority tier from last done (or stay at lowest if already there).
        # If last_priority is None (no history) treat as starting from lowest available tier.
        if last_priority is not None:
            target_priority = last_priority - 1 if last_priority > sorted_priorities[0] else sorted_priorities[0]
        else:
            target_priority = sorted_priorities[0]

        tier = [ex for ex in pool if ex.priority == target_priority]
        if not tier:
            # Fallback: use lowest available priority
            tier = [ex for ex in pool if ex.priority == sorted_priorities[0]]
        result = _pick_least_recently_done(tier, history)
        logger.debug(
            "Swelling moderate | last_priority=%s target_priority=%d → %s",
            last_priority, target_priority, result.exercise_id,
        )
        return result

    if cur_swelling == "mild":
        # Rules 3 & 4: stay at the same exercise as last session.
        last = _last_recommended(history, phase_id)
        if last and last.get("exercise_id"):
            last_ex = next((ex for ex in pool if ex.exercise_id == last["exercise_id"]), None)
            if last_ex:
                logger.debug("Swelling mild | staying on %s", last_ex.exercise_id)
                return last_ex
        # If last exercise isn't in pool (equipment changed / caution), stay in same priority tier.
        if last_priority is not None:
            tier = [ex for ex in pool if ex.priority == last_priority]
            if tier:
                result = _pick_least_recently_done(tier, history)
                logger.debug(
                    "Swelling mild | last exercise unavailable, using priority=%d → %s",
                    last_priority, result.exercise_id,
                )
                return result
        # Fallback: lowest priority tier
        tier = [ex for ex in pool if ex.priority == sorted_priorities[0]]
        return _pick_least_recently_done(tier, history)

    # --- Stage 4: Normal progression (no swelling or unknown) ---
    # Rules 6 & 7: progress through exercises in priority order.
    # Within each priority tier: complete all unseen exercises before advancing.
    # Once all in a tier are done, move to the next priority tier.

    for priority_tier in sorted_priorities:
        tier = [ex for ex in pool if ex.priority == priority_tier]
        unseen = [ex for ex in tier if ex.exercise_id not in seen_ids]
        if unseen:
            # Prefer unseen in this tier; pick first alphabetically for determinism
            unseen.sort(key=lambda ex: ex.exercise_id)
            logger.debug(
                "Normal progression | priority=%d → %s (unseen: %d remaining)",
                priority_tier, unseen[0].exercise_id, len(unseen),
            )
            return unseen[0]
        # All in this tier seen — continue to next tier

    # Unreachable: _seen_exercise_ids_in_current_cycle resets on full cycle,
    # so at least one priority tier always has unseen exercises above.
    return _pick_least_recently_done(pool, history)


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

    best = _select_exercise(candidates, nlu, phase_id, equipment_available, exercise_history)

    if best is None:
        return {"plan": None, "notes": ["No suitable exercise could be selected."]}

    if best == _STOP_EXERCISE_SENTINEL:
        return {
            "plan": None,
            "outcome": "STOP_EXERCISE",
            "notes": [
                "Swelling has remained mild for 3 or more consecutive sessions. "
                "Please rest and avoid exercise until swelling subsides."
            ],
        }

    logger.debug(
        "Selected | exercise_id=%s name=%s priority=%d",
        best.exercise_id, best.name, best.priority,
    )

    return {
        "exercise_id": best.exercise_id,
        "exercise_name": best.name,
        "phase_id": phase_id,
        "priority": best.priority,
        "position": best.position,
        "caution": best.caution,
        "equipment_required": best.equipment_required,
        # For RAG: structured evidence chunks (chunk_id + source_id)
        "evidence_chunks": [
            {"chunk_id": ec.chunk_id, "source_id": ec.source_id}
            for ec in best.evidence_chunks
        ],
        # For display: flat citation strings
        "citations": best.source_refs,
        "stop_conditions": ["pain increases", "swelling increases"],
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
