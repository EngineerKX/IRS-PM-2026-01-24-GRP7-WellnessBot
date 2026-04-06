from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Sequence


FINAL_RESPONSE_SYSTEM_PROMPT = """
You are generating the final recommendation card for a rehabilitation support application.

You must only verbalize the structured recommendation and supporting evidence provided.
Do not change the recommended exercise.
Do not add new exercises, precautions, repetitions, diagnoses, or advice.
Do not invent evidence.
Do not mention internal logic, planner, rules, confidence, or JSON.
If evidence_rows is empty, omit the Evidence Support section.
If supportive_care is empty, omit the Supportive Care section.
Use professional, calm, patient-friendly clinical language.

Required sections when data exists:
1. Recommendation
2. Why This Is Appropriate
3. How To Perform
4. Safety Note
5. Evidence Support
6. Supportive Care

For each evidence bullet, include the citation at the end in this format:
(SOURCE_ID: SOURCE_LINK)
""".strip()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_list(items: Iterable[Any] | None) -> List[str]:
    return [_safe_text(item) for item in (items or []) if _safe_text(item)]


def _primary_rationale(result: Dict[str, Any]) -> str:
    action = _safe_text(((result.get("decision") or {}).get("action")))
    rules = ((result.get("audit_trace") or {}).get("rules_fired") or [])

    for rule in rules:
        if _safe_text(rule.get("action")) == action and _safe_text(rule.get("rationale")):
            return _safe_text(rule.get("rationale"))

    for rule in rules:
        if _safe_text(rule.get("rationale")):
            return _safe_text(rule.get("rationale"))

    return ""


def _normalize_evidence_rows(evidence_rows: Sequence[Dict[str, Any]] | None) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for row in evidence_rows or []:
        text = _safe_text(row.get("text"))
        if not text:
            continue

        normalized.append(
            {
                "chunk_id": _safe_text(row.get("chunk_id")),
                "text": text,
                "source_id": _safe_text(row.get("source_id")),
                "source_link": _safe_text(row.get("source_url") or row.get("source_link")),
            }
        )
    return normalized


def build_final_response_payload(
    result: Dict[str, Any],
    evidence_rows: Sequence[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    planner = ((result.get("audit_trace") or {}).get("planner") or {})
    nlu = result.get("nlu") or {}
    state = ((result.get("audit_trace") or {}).get("state") or {})

    return {
        "patient_context": {
            "surgery_type": _safe_text(nlu.get("surgery_type")),
            "phase_id": _safe_text(planner.get("phase_id") or state.get("phase_id")),
            "phase_name": _safe_text(((result.get("audit_trace") or {}).get("audit_context") or {}).get("phase_name")),
            "pain_score": nlu.get("pain_score"),
            "swelling_score": nlu.get("swelling_score"),
        },
        "recommendation": {
            "action": _safe_text(((result.get("decision") or {}).get("action"))),
            "exercise_id": _safe_text(planner.get("exercise_id")),
            "exercise_name": _safe_text(planner.get("exercise_name")),
            "position": _safe_text(planner.get("position")),
            "caution": _safe_text(planner.get("caution")),
            "equipment_required": _safe_list(planner.get("equipment_required")),
            "stop_conditions": _safe_list(planner.get("stop_conditions")),
        },
        "supportive_care": _safe_list(planner.get("selfcare_routine")),
        "rationale": _primary_rationale(result),
        "evidence_rows": _normalize_evidence_rows(evidence_rows),
    }


def build_final_response_messages(
    result: Dict[str, Any],
    evidence_rows: Sequence[Dict[str, Any]] | None,
) -> List[Dict[str, str]]:
    payload = build_final_response_payload(result, evidence_rows)
    user_prompt = (
        "Generate the final recommendation card from the structured data below.\n\n"
        f"Structured input:\n{json.dumps(payload, indent=2, ensure_ascii=True)}\n\n"
        "Writing instructions:\n"
        "- Use the exercise_name exactly as provided.\n"
        "- If caution is non-empty, incorporate it naturally into Safety Note.\n"
        "- Do not mention chunk_id in the patient-facing prose unless explicitly asked.\n"
        "- Keep citations attached to each evidence bullet using the provided source_id and source_link.\n"
    )

    return [
        {"role": "system", "content": FINAL_RESPONSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def validate_final_response_text(text: str, payload: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    normalized_text = _safe_text(text)
    recommendation = payload.get("recommendation") or {}
    exercise_name = _safe_text(recommendation.get("exercise_name"))

    if exercise_name and exercise_name.lower() not in normalized_text.lower():
        errors.append("Final response does not mention the recommended exercise name.")

    for forbidden_token in ("rule ids", "confidence", "json", "planner"):
        if forbidden_token in normalized_text.lower():
            errors.append(f"Final response mentions forbidden internal detail: {forbidden_token}.")

    return errors