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
If evidence_rows is empty, omit the References section.
If supportive_care is empty, omit the Self Care section.
Use professional, calm, patient-friendly clinical language.

Required sections when data exists, in this order:
1. Self Care
2. Recommended exercise
3. How To Perform
4. References

Do not add any introductory summary sentence before the first section.
Do not restate clinical rationale such as "Mild symptoms noted".

How To Perform grounding rule:
- Rephrase the content from how_to_perform into clear, natural patient-friendly language.
- Do NOT add new steps, repetitions, or procedural details that are not in the source.
- Do NOT omit or summarise any dosage, repetition, or set information present in the source (e.g. "1 set of 10 reps", "increase to 3 sets of 15 reps"). These must appear in full.
- If how_to_perform is empty, omit the How To Perform section.

For each References bullet:
- Use the source_id and source_link from evidence_rows ONLY.
- Do NOT include the evidence text in the References section.
- Format each bullet as: (SOURCE_ID: SOURCE_LINK)
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


def _build_how_to_perform_lines(evidence_rows: Sequence[Dict[str, Any]] | None, max_items: int = 1) -> List[str]:
    lines: List[str] = []
    for row in evidence_rows or []:
        text = _safe_text(row.get("text"))
        if not text:
            continue
        if text not in lines:
            lines.append(text)
        if len(lines) >= max(max_items, 1):
            break
    return lines


def build_final_response_payload(
    result: Dict[str, Any],
    evidence_rows: Sequence[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    planner = ((result.get("audit_trace") or {}).get("planner") or {})
    nlu = result.get("nlu") or {}
    state = ((result.get("audit_trace") or {}).get("state") or {})
    normalized_evidence = _normalize_evidence_rows(evidence_rows)
    how_to_perform_lines = _build_how_to_perform_lines(normalized_evidence)
    filtered_evidence = [row for row in normalized_evidence if row["text"] in how_to_perform_lines]

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
        "how_to_perform": how_to_perform_lines,
        "evidence_rows": filtered_evidence,
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
        "- Start directly with the first available section. Do not add any lead-in sentence before Self Care.\n"
        "- Use the exercise_name exactly as provided.\n"
        "- Do not generate a Safety Note or Why This Is Appropriate section.\n"
        "- In How To Perform, rephrase the content from how_to_perform into clear, patient-friendly language. Preserve all repetition and set counts exactly as stated (e.g. '1 set of 10 reps', '3 sets of 15 reps'). Do not omit any dosage information.\n"
        "- Do not mention chunk_id in the patient-facing prose unless explicitly asked.\n"
        "- In References, each bullet must contain ONLY the source_id and source_link. Do not include the evidence text in References.\n"
        "- Label the exercise section 'Recommended exercise', the evidence section 'References', and the self-care section 'Self Care'.\n"
        "- Only include the Self Care section if supportive_care is non-empty. If supportive_care is empty or missing, omit the Self Care section entirely — do not output the heading.\n"
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
    expected_how_to_perform = _safe_list(payload.get("how_to_perform"))

    if exercise_name and exercise_name.lower() not in normalized_text.lower():
        errors.append("Final response does not mention the recommended exercise name.")

    for forbidden_token in ("rule ids", "confidence", "json", "planner"):
        if forbidden_token in normalized_text.lower():
            errors.append(f"Final response mentions forbidden internal detail: {forbidden_token}.")

    for hallucinated_section in ("why this is appropriate", "safety note"):
        if hallucinated_section in normalized_text.lower():
            errors.append(f"Final response contains removed section: {hallucinated_section}.")

    if expected_how_to_perform and "how to perform" not in normalized_text.lower():
        errors.append("Final response is missing the How To Perform section.")

    return errors