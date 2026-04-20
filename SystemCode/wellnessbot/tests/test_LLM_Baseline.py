import os
import re
import csv
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL = "gpt-4o-mini"
REFERENCE_DATE = "18/4/2026"

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "input" / "Scenario.txt"

OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "baseline_results.csv"

VALID_LABELS = {"RECOMMEND", "ESCALATE", "FORBID"}

# --------------------------------------------------
# OPENAI CLIENT
# --------------------------------------------------
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=api_key)

# --------------------------------------------------
# PROMPT
# --------------------------------------------------
INSTRUCTIONS = f"""
You are a baseline decision model for a rehabilitation support system.

Output exactly one label only:
RECOMMEND
ESCALATE
FORBID

Do not explain.
Do not ask questions.
Do not output anything else.

Always follow the rules strictly.

Reference date: {REFERENCE_DATE}

Rules:
- Fever or excessive bleeding → ESCALATE
- Exception: Fever in P1_1 → FORBID
- If symptom = none:
    - pain = 3 OR swelling = 3 → ESCALATE
    - pain = 2 → FORBID
    - pain < 2 AND swelling <= 2 → RECOMMEND

Priority:
ESCALATE > FORBID > RECOMMEND
"""

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def extract_label(text):
    text = text.strip().upper()

    if text in VALID_LABELS:
        return text

    match = re.search(r"(RECOMMEND|ESCALATE|FORBID)", text)
    if match:
        return match.group(1)

    return "INVALID"


def parse_scenarios(text):
    pattern = r"(Scenario\s+\d+:\s*.*?)(?=\n\s*Scenario\s+\d+:\s*|$)"
    return re.findall(pattern, text, re.DOTALL)


def classify(scenario):
    response = client.responses.create(
        model=MODEL,
        instructions=INSTRUCTIONS,
        input=scenario,
    )
    return extract_label(response.output_text)

# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # Create output folder
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    content = INPUT_FILE.read_text(encoding="utf-8")
    scenarios = parse_scenarios(content)

    results = []

    for i, scenario in enumerate(scenarios, 1):
        try:
            label = classify(scenario)
        except Exception as e:
            label = f"ERROR: {e}"

        # ✅ Console output
        print(f"Scenario {i}: {label}")

        # ✅ Save same format
        results.append({
            "output": f"Scenario {i}: {label}"
        })

    # Save CSV
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["output"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\n✅ Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()