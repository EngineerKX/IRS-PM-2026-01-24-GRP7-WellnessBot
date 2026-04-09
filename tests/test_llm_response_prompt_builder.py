from __future__ import annotations

from wellnessbot.llm_response import (
    FINAL_RESPONSE_SYSTEM_PROMPT,
    build_final_response_messages,
    build_final_response_payload,
    generate_final_response_text,
    validate_final_response_text,
)


def _sample_result() -> dict:
    return {
        "decision": {"action": "RECOMMEND"},
        "nlu": {
            "surgery_type": "arthroscopic_knee_surgery",
            "pain_score": 1,
            "swelling_score": 1,
        },
        "audit_trace": {
            "audit_context": {"phase_name": "Early"},
            "state": {"phase_id": "P1_2"},
            "rules_fired": [
                {
                    "action": "RECOMMEND",
                    "rationale": "Mild symptoms noted. Proceed conservatively with supportive follow-up.",
                }
            ],
            "planner": {
                "exercise_id": "P1_2_E3",
                "exercise_name": "Straight leg raises, supine",
                "phase_id": "P1_2",
                "position": "Lying",
                "caution": "Do not perform straight leg raise if you have a knee extension lag",
                "equipment_required": [],
                "stop_conditions": ["pain increases"],
                "selfcare_routine": ["Ice and elevate for 10 minutes after exercise if needed."],
            },
        },
    }


def _sample_evidence_rows() -> list[dict]:
    return [
        {
            "chunk_id": "P1_2_E3_C1",
            "text": "Lying flat, lock your knee straight and then lift the whole leg about 30cms off the bed.",
            "source_id": "S2",
            "source_url": "https://www.melbournehipandknee.com.au/pdf/knee-arthroscopy-rehabilitation-protocol.pdf",
        },
        {
            "chunk_id": "P1_2_E3_C2",
            "text": "Lying on your back, bend the uninvolved knee and put it on the floor.",
            "source_id": "S5",
            "source_link": "https://hamishlove.nz/wp-content/uploads/2017/10/SSO-Knee-Initial-post-op-rehab.pdf",
        },
    ]


def test_build_final_response_payload_normalizes_expected_fields():
    payload = build_final_response_payload(_sample_result(), _sample_evidence_rows())

    assert payload["patient_context"]["phase_id"] == "P1_2"
    assert payload["recommendation"]["exercise_id"] == "P1_2_E3"
    assert payload["recommendation"]["exercise_name"] == "Straight leg raises, supine"
    assert payload["supportive_care"] == ["Ice and elevate for 10 minutes after exercise if needed."]
    assert "rationale" not in payload
    assert payload["evidence_rows"][0]["source_link"] == "https://www.melbournehipandknee.com.au/pdf/knee-arthroscopy-rehabilitation-protocol.pdf"
    assert payload["evidence_rows"][1]["source_link"] == "https://hamishlove.nz/wp-content/uploads/2017/10/SSO-Knee-Initial-post-op-rehab.pdf"


def test_build_final_response_messages_contains_system_prompt_and_json_payload():
    messages = build_final_response_messages(_sample_result(), _sample_evidence_rows())

    assert messages[0]["role"] == "system"
    assert FINAL_RESPONSE_SYSTEM_PROMPT in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert '"exercise_id": "P1_2_E3"' in messages[1]["content"]
    assert '"chunk_id": "P1_2_E3_C1"' in messages[1]["content"]


def test_validate_final_response_text_flags_internal_details_and_missing_exercise_name():
    payload = build_final_response_payload(_sample_result(), _sample_evidence_rows())

    errors = validate_final_response_text(
        "This planner JSON output has confidence 0.7.",
        payload,
    )

    assert any("recommended exercise name" in error.lower() for error in errors)
    assert any("confidence" in error.lower() for error in errors)


def test_llm_response_package_exports_verbalizer_symbol():
    assert callable(generate_final_response_text)