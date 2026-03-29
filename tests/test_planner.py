"""Tests for the Intervention Planner (A* informed search exercise selection)."""
import pytest

from wellnessbot.planner.planner import (
    run_planner,
    prune_candidates,
    compute_g,
    compute_h,
    score_candidates,
    select_best,
    SWELLING_RANK,
)


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

P1_EXERCISES = [
    {
        "exercise_id": "P1_E1",
        "name": "Ankle Pumps",
        "allowed_phases": ["P1"],
        "effort_level": 0.2,
        "priority": 1,
        "limits": {
            "pain_max": 6.0,
            "swelling_max": "moderate",
            "weight_bearing_required": "none",
            "notes": "Safe for all early post-op patients",
        },
        "dose": {"sets": 3, "reps": 20, "hold_sec": 0, "frequency_per_day": 3},
        "evidence": [{"page": 10, "section": "Early mobilisation"}],
    },
    {
        "exercise_id": "P1_E2",
        "name": "Knee Locking",
        "allowed_phases": ["P1"],
        "effort_level": 0.4,
        "priority": 2,
        "limits": {
            "pain_max": 5.0,
            "swelling_max": "mild",
            "weight_bearing_required": "none",
            "notes": "Quad activation focus",
        },
        "dose": {"sets": 3, "reps": 10, "hold_sec": 5, "frequency_per_day": 2},
        "evidence": [{"page": 12, "section": "Quadriceps activation"}],
    },
    {
        "exercise_id": "P1_E3",
        "name": "Straight Leg Raises",
        "allowed_phases": ["P1"],
        "effort_level": 0.6,
        "priority": 3,
        "limits": {
            "pain_max": 4.0,
            "swelling_max": "mild",
            "weight_bearing_required": "none",
            "notes": "Requires adequate quad control",
        },
        "dose": {"sets": 3, "reps": 10, "hold_sec": 3, "frequency_per_day": 2},
        "evidence": [{"page": 14, "section": "SLR progression"}],
    },
    {
        "exercise_id": "P1_E4",
        "name": "Heel Slides (lying)",
        "allowed_phases": ["P1"],
        "effort_level": 0.8,
        "priority": 4,
        "limits": {
            "pain_max": 4.0,
            "swelling_max": "mild",
            "weight_bearing_required": "none",
            "notes": "ROM exercise -- monitor flexion range",
        },
        "dose": {"sets": 2, "reps": 15, "hold_sec": 0, "frequency_per_day": 2},
        "evidence": [{"page": 15, "section": "Range of motion"}],
    },
    {
        "exercise_id": "P1_E5",
        "name": "Heel Slides (sitting)",
        "allowed_phases": ["P1"],
        "effort_level": 1.0,
        "priority": 5,
        "limits": {
            "pain_max": 3.0,
            "swelling_max": "none",
            "weight_bearing_required": "partial",
            "notes": "Gravity-assisted -- harder than lying variant",
        },
        "dose": {"sets": 2, "reps": 10, "hold_sec": 0, "frequency_per_day": 1},
        "evidence": [{"page": 16, "section": "Seated ROM progression"}],
    },
]


# ---------------------------------------------------------------------------
# Test 1 -- Normal RECOMMEND
# ---------------------------------------------------------------------------

class TestNormalRecommend:
    """User: Day 7, Phase P1, pain 3/10, swelling mild, WB partial.
    Some candidates should survive and the highest f(n) is selected."""

    def test_outcome_is_recommend(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 3.0,
            "swelling_level": "mild",
            "weight_bearing_status": "partial",
        }
        result = run_planner(user, P1_EXERCISES)
        assert result["outcome"] == "RECOMMEND"

    def test_selected_exercise_present(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 3.0,
            "swelling_level": "mild",
            "weight_bearing_status": "partial",
        }
        result = run_planner(user, P1_EXERCISES)
        assert "selected_exercise" in result
        assert result["selected_exercise"]["exercise_id"] is not None

    def test_audit_log_has_scored_and_eliminated(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 3.0,
            "swelling_level": "mild",
            "weight_bearing_status": "partial",
        }
        result = run_planner(user, P1_EXERCISES)
        audit = result["audit_log"]
        assert audit["candidates_received"] == 5
        assert len(audit["scored"]) > 0
        assert audit["selected_exercise_id"] == result["selected_exercise"]["exercise_id"]


# ---------------------------------------------------------------------------
# Test 2 -- ESCALATE (all eliminated)
# ---------------------------------------------------------------------------

class TestEscalateAllEliminated:
    """User: Day 7, Phase P1, pain 9/10, swelling severe, WB non_partial.
    All candidates should fail hard constraints."""

    def test_outcome_is_escalate(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 9.0,
            "swelling_level": "severe",
            "weight_bearing_status": "non_partial",
        }
        result = run_planner(user, P1_EXERCISES)
        assert result["outcome"] == "ESCALATE"

    def test_escalate_reason(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 9.0,
            "swelling_level": "severe",
            "weight_bearing_status": "non_partial",
        }
        result = run_planner(user, P1_EXERCISES)
        assert "No safe exercise" in result["reason"]

    def test_all_candidates_eliminated(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 9.0,
            "swelling_level": "severe",
            "weight_bearing_status": "non_partial",
        }
        result = run_planner(user, P1_EXERCISES)
        audit = result["audit_log"]
        assert len(audit["eliminated"]) == 5
        assert len(audit["scored"]) == 0
        assert audit["selected_exercise_id"] is None


# ---------------------------------------------------------------------------
# Test 3 -- Tie-break scenarios
# Tests select_best() directly with forced identical f(n) scores.
# effort_level feeds into h(n), so different effort_levels always produce
# different f(n) in practice. To test tie-breaking in isolation, we
# construct pre-scored records with identical f_n.
# ---------------------------------------------------------------------------

class TestTieBreaking:

    def test_lower_effort_level_wins(self):
        """When f(n) is tied, the candidate with lower effort_level wins."""
        scored = [
            {
                "exercise_id": "TIE_A",
                "name": "Exercise A",
                "g_n": 0.3,
                "h_n": 0.7,
                "f_n": 1.0,
                "_exercise": {"exercise_id": "TIE_A", "effort_level": 0.6, "priority": 3},
            },
            {
                "exercise_id": "TIE_B",
                "name": "Exercise B",
                "g_n": 0.3,
                "h_n": 0.7,
                "f_n": 1.0,
                "_exercise": {"exercise_id": "TIE_B", "effort_level": 0.4, "priority": 3},
            },
        ]
        original = [{"exercise_id": "TIE_A"}, {"exercise_id": "TIE_B"}]
        best, rationale = select_best(scored, original)
        assert best["exercise_id"] == "TIE_B"
        assert "effort_level" in rationale

    def test_lower_priority_number_wins(self):
        """When f(n) and effort_level are tied, lower priority number wins."""
        scored = [
            {
                "exercise_id": "TIE_C",
                "name": "Exercise C",
                "g_n": 0.3,
                "h_n": 0.7,
                "f_n": 1.0,
                "_exercise": {"exercise_id": "TIE_C", "effort_level": 0.4, "priority": 5},
            },
            {
                "exercise_id": "TIE_D",
                "name": "Exercise D",
                "g_n": 0.3,
                "h_n": 0.7,
                "f_n": 1.0,
                "_exercise": {"exercise_id": "TIE_D", "effort_level": 0.4, "priority": 2},
            },
        ]
        original = [{"exercise_id": "TIE_C"}, {"exercise_id": "TIE_D"}]
        best, rationale = select_best(scored, original)
        assert best["exercise_id"] == "TIE_D"
        assert "priority" in rationale

    def test_position_order_wins(self):
        """When f(n), effort, and priority are all tied, earlier position wins."""
        scored = [
            {
                "exercise_id": "TIE_X",
                "name": "Exercise X",
                "g_n": 0.3,
                "h_n": 0.7,
                "f_n": 1.0,
                "_exercise": {"exercise_id": "TIE_X", "effort_level": 0.4, "priority": 3},
            },
            {
                "exercise_id": "TIE_Y",
                "name": "Exercise Y",
                "g_n": 0.3,
                "h_n": 0.7,
                "f_n": 1.0,
                "_exercise": {"exercise_id": "TIE_Y", "effort_level": 0.4, "priority": 3},
            },
        ]
        # TIE_X appears first in the original candidate list
        original = [{"exercise_id": "TIE_X"}, {"exercise_id": "TIE_Y"}]
        best, rationale = select_best(scored, original)
        assert best["exercise_id"] == "TIE_X"
        assert "position" in rationale


# ---------------------------------------------------------------------------
# Test 4 -- Phase mismatch
# ---------------------------------------------------------------------------

class TestPhaseMismatch:
    """User in P2, but all candidates are P1 exercises. All should be
    eliminated on phase eligibility, resulting in ESCALATE."""

    def test_escalate_on_phase_mismatch(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P2",
            "pain_level": 2.0,
            "swelling_level": "none",
            "weight_bearing_status": "full",
        }
        result = run_planner(user, P1_EXERCISES)
        assert result["outcome"] == "ESCALATE"

    def test_all_eliminated_for_phase(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P2",
            "pain_level": 2.0,
            "swelling_level": "none",
            "weight_bearing_status": "full",
        }
        result = run_planner(user, P1_EXERCISES)
        for entry in result["audit_log"]["eliminated"]:
            assert "Phase mismatch" in entry["reason"]


# ---------------------------------------------------------------------------
# Test 5 -- Partial elimination
# ---------------------------------------------------------------------------

class TestPartialElimination:
    """User: Day 7, Phase P1, pain 5/10, swelling moderate, WB partial.
    Some harder exercises should be eliminated, easier ones survive.
    Conservative selection expected."""

    def test_recommend_with_partial_elimination(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 5.0,
            "swelling_level": "moderate",
            "weight_bearing_status": "partial",
        }
        result = run_planner(user, P1_EXERCISES)
        assert result["outcome"] == "RECOMMEND"

    def test_easiest_survives_hardest_eliminated(self):
        user = {
            "days_post_op": 7,
            "rehab_phase": "P1",
            "pain_level": 5.0,
            "swelling_level": "moderate",
            "weight_bearing_status": "partial",
        }
        result = run_planner(user, P1_EXERCISES)
        audit = result["audit_log"]

        surviving_ids = {s["exercise_id"] for s in audit["scored"]}
        eliminated_ids = {e["exercise_id"] for e in audit["eliminated"]}

        assert "P1_E1" in surviving_ids, "Ankle Pumps (easiest) should survive"
        assert "P1_E5" in eliminated_ids, "Heel Slides sitting (hardest) should be eliminated"
        assert len(audit["eliminated"]) > 0
        assert len(audit["scored"]) > 0


# ---------------------------------------------------------------------------
# Unit tests for scoring functions
# ---------------------------------------------------------------------------

class TestComputeG:
    """Verify g(n) risk cost computation."""

    def test_zero_pain_zero_swelling(self):
        user = {"pain_level": 0.0, "swelling_level": "none", "weight_bearing_status": "full"}
        ex = {
            "limits": {"pain_max": 6.0, "swelling_max": "moderate", "weight_bearing_required": "none"},
        }
        g = compute_g(user, ex)
        assert g == pytest.approx(0.0, abs=1e-6)

    def test_at_pain_ceiling(self):
        user = {"pain_level": 6.0, "swelling_level": "none", "weight_bearing_status": "full"}
        ex = {
            "limits": {"pain_max": 6.0, "swelling_max": "moderate", "weight_bearing_required": "none"},
        }
        g = compute_g(user, ex)
        # pain_risk = 1.0, swelling_risk = 0.0, wb_burden = 0.0 => mean = 0.333...
        assert g == pytest.approx(1.0 / 3, abs=1e-6)


class TestComputeH:
    """Verify h(n) rehab benefit computation."""

    def test_perfect_candidate(self):
        user = {"rehab_phase": "P1", "pain_level": 0.0, "swelling_level": "none"}
        ex = {
            "allowed_phases": ["P1"],
            "effort_level": 0.2,
            "priority": 1,
            "limits": {"pain_max": 6.0, "swelling_max": "moderate"},
        }
        h = compute_h(user, ex)
        # phase_match=1.0, pain_margin=1.0, swelling_margin=1.0,
        # effort_inv=0.8, priority=1.0 => mean = 0.96
        assert h == pytest.approx(0.96, abs=1e-6)


# ---------------------------------------------------------------------------
# Unit tests for pruning
# ---------------------------------------------------------------------------

class TestPruneCandidates:

    def test_empty_candidates(self):
        user = {"rehab_phase": "P1", "pain_level": 0, "swelling_level": "none", "weight_bearing_status": "full"}
        survivors, eliminated = prune_candidates(user, [])
        assert survivors == []
        assert eliminated == []

    def test_single_candidate_passes(self):
        user = {"rehab_phase": "P1", "pain_level": 2.0, "swelling_level": "none", "weight_bearing_status": "full"}
        survivors, eliminated = prune_candidates(user, [P1_EXERCISES[0]])
        assert len(survivors) == 1
        assert len(eliminated) == 0

    def test_multiple_constraint_failures_combined(self):
        """A candidate failing multiple constraints has all reasons joined."""
        user = {"rehab_phase": "P1", "pain_level": 9.0, "swelling_level": "severe", "weight_bearing_status": "non_partial"}
        _, eliminated = prune_candidates(user, [P1_EXERCISES[4]])  # P1_E5
        assert len(eliminated) == 1
        reason = eliminated[0]["reason"]
        assert "Pain too high" in reason
        assert "Swelling too high" in reason
        assert "Weight-bearing insufficient" in reason
