import os
import re
import csv
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL = "gpt-4o-mini"

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input" / "Scenario.txt"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "baseline_results.csv"
METRICS_FILE = OUTPUT_DIR / "baseline_metrics.csv"
CONFUSION_FILE = OUTPUT_DIR / "confusion_matrix.csv"

VALID_LABELS = ["RECOMMEND", "FORBID", "ESCALATE"]
VALID_LABEL_SET = set(VALID_LABELS)

# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# PROMPT BUILDER
# --------------------------------------------------
def build_instructions(reference_date: str, weeks_since_event, phase: str) -> str:
    return f"""
You are a baseline decision model for a rehabilitation support system.

Output exactly one label only:
RECOMMEND
FORBID
ESCALATE

Do not explain.
Do not ask questions.
Do not output anything else.

Always follow the rules strictly.

Reference date: {reference_date}
Weeks since surgery: {weeks_since_event}
Computed rehabilitation phase: {phase}

Phase reference:
- P1_1: Initial phase, week 0 to 2
- P1_2: Early phase, week >2 to 4
- P2: Intermediate phase, week >4 to 6
- P3: Late phase, week >6 to 10
- P4: Transitional phase, week >10 to 13
- P5: Return to sport phase, week >13 to 21

Use the computed rehabilitation phase above.
Do not recalculate the phase yourself.

Rules:
- Fever or excessive bleeding → ESCALATE
- Exception: Fever in P1_1 → FORBID
- If symptom response = none:
    - pain score = 3 OR swelling score = 3 → ESCALATE
    - pain score = 2 → FORBID
    - pain score < 2 AND swelling score <= 2 → RECOMMEND

Priority:
ESCALATE > FORBID > RECOMMEND
""".strip()

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def extract_label(text: str) -> str:
    text = text.strip().upper()

    if text in VALID_LABEL_SET:
        return text

    match = re.search(r"(RECOMMEND|FORBID|ESCALATE)", text)
    if match:
        return match.group(1)

    return "INVALID"


def parse_field(block: str, field_name: str) -> str:
    pattern = rf"{re.escape(field_name)}:\s*(.*)"
    match = re.search(pattern, block, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def parse_any_field(block: str, field_names: list[str]) -> str:
    for field_name in field_names:
        value = parse_field(block, field_name)
        if value:
            return value
    return ""


def parse_date(date_text: str):
    if not date_text:
        return None

    date_text = date_text.strip()

    for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"]:
        try:
            return datetime.strptime(date_text, fmt)
        except ValueError:
            continue

    return None


def compute_weeks(reference_date: str, surgery_date: str):
    ref = parse_date(reference_date)
    surg = parse_date(surgery_date)

    if ref is None or surg is None:
        return None

    delta_days = (ref - surg).days
    return round(delta_days / 7, 2)


def map_to_phase(weeks_since_event):
    if weeks_since_event is None:
        return "UNKNOWN"

    if 0 <= weeks_since_event < 2:
        return "P1_1"
    elif 2 <= weeks_since_event < 4:
        return "P1_2"
    elif 4 <= weeks_since_event < 6:
        return "P2"
    elif 6 <= weeks_since_event < 10:
        return "P3"
    elif 10 <= weeks_since_event < 13:
        return "P4"
    elif 13 <= weeks_since_event < 21:
        return "P5"
    else:
        return "UNKNOWN"


def parse_scenarios(text: str):
    pattern = r"(Scenario\s+\d+:\s*.*?)(?=\n\s*Scenario\s+\d+:\s*|$)"
    blocks = re.findall(pattern, text, re.DOTALL)

    scenarios = []

    for block in blocks:
        scenario_no_match = re.search(r"Scenario\s+(\d+):", block, re.IGNORECASE)
        scenario_no = int(scenario_no_match.group(1)) if scenario_no_match else None

        test_date = parse_field(block, "Test date")

        surgery_date = parse_any_field(block, [
            "Date of surgery",
            "Surgery date",
            "Operation date",
            "Procedure date"
        ])

        actual_action = parse_field(block, "Actual action").upper()

        # Remove actual action before sending to LLM to avoid answer leakage
        llm_input = re.sub(
            r"\n?Actual action:\s*.*",
            "",
            block,
            flags=re.IGNORECASE
        ).strip()

        scenarios.append({
            "scenario_no": scenario_no,
            "test_date": test_date,
            "surgery_date": surgery_date,
            "actual_action": actual_action,
            "llm_input": llm_input
        })

    return scenarios


def classify(
    scenario_input: str,
    reference_date: str,
    weeks_since_event,
    phase: str
) -> str:
    response = client.responses.create(
        model=MODEL,
        instructions=build_instructions(
            reference_date=reference_date,
            weeks_since_event=weeks_since_event,
            phase=phase
        ),
        input=scenario_input,
    )

    return extract_label(response.output_text)


def build_confusion_matrix(rows):
    matrix = {
        actual: {pred: 0 for pred in VALID_LABELS}
        for actual in VALID_LABELS
    }

    for row in rows:
        actual = row["actual_action"]
        pred = row["predicted_action"]

        if actual in VALID_LABEL_SET and pred in VALID_LABEL_SET:
            matrix[actual][pred] += 1

    return matrix


def compute_metrics(rows):
    valid_rows = [
        r for r in rows
        if r["actual_action"] in VALID_LABEL_SET
        and r["predicted_action"] in VALID_LABEL_SET
    ]

    total = len(valid_rows)

    correct = sum(
        1 for r in valid_rows
        if r["actual_action"] == r["predicted_action"]
    )

    accuracy = correct / total if total else 0

    metrics = []

    for label in VALID_LABELS:
        tp = sum(
            1 for r in valid_rows
            if r["actual_action"] == label
            and r["predicted_action"] == label
        )

        fp = sum(
            1 for r in valid_rows
            if r["actual_action"] != label
            and r["predicted_action"] == label
        )

        fn = sum(
            1 for r in valid_rows
            if r["actual_action"] == label
            and r["predicted_action"] != label
        )

        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )

        metrics.append({
            "label": label,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(
                1 for r in valid_rows
                if r["actual_action"] == label
            )
        })

    return accuracy, metrics


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    content = INPUT_FILE.read_text(encoding="utf-8")
    scenarios = parse_scenarios(content)

    results = []

    for scenario in scenarios:
        scenario_no = scenario["scenario_no"]
        test_date = scenario["test_date"]
        surgery_date = scenario["surgery_date"]
        actual_action = scenario["actual_action"]

        weeks_since_event = compute_weeks(test_date, surgery_date)
        phase = map_to_phase(weeks_since_event)

        if actual_action not in VALID_LABEL_SET:
            predicted_action = "SKIPPED"
        elif phase == "UNKNOWN":
            predicted_action = "SKIPPED_PHASE_UNKNOWN"
        else:
            try:
                predicted_action = classify(
                    scenario_input=scenario["llm_input"],
                    reference_date=test_date,
                    weeks_since_event=weeks_since_event,
                    phase=phase
                )
            except Exception as e:
                predicted_action = f"ERROR: {e}"

        match = predicted_action == actual_action

        print(
            f"Scenario {scenario_no}: "
            f"Surgery={surgery_date}, Test={test_date}, "
            f"Weeks={weeks_since_event}, Phase={phase}, "
            f"Actual={actual_action}, Predicted={predicted_action}, Match={match}"
        )

        results.append({
            "scenario_no": scenario_no,
            "test_date": test_date,
            "surgery_date": surgery_date,
            "weeks_since_event": weeks_since_event,
            "phase": phase,
            "actual_action": actual_action,
            "predicted_action": predicted_action,
            "match": match
        })

    # Save prediction results
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario_no",
                "test_date",
                "surgery_date",
                "weeks_since_event",
                "phase",
                "actual_action",
                "predicted_action",
                "match"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    # Confusion matrix
    matrix = build_confusion_matrix(results)

    with open(CONFUSION_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual \\ Predicted"] + VALID_LABELS)

        for actual in VALID_LABELS:
            writer.writerow(
                [actual] + [matrix[actual][pred] for pred in VALID_LABELS]
            )

    # Metrics
    accuracy, metrics = compute_metrics(results)

    with open(METRICS_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "precision", "recall", "f1", "support"]
        )

        writer.writeheader()
        writer.writerows(metrics)
        writer.writerow({})
        writer.writerow({
            "label": "accuracy",
            "precision": round(accuracy, 4),
            "recall": "",
            "f1": "",
            "support": ""
        })

    print("\nDone.")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Confusion matrix saved to: {CONFUSION_FILE}")
    print(f"Metrics saved to: {METRICS_FILE}")
    print(f"Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()