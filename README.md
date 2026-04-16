# IRS-PM-2025-[StartDate]-[BatchCode]-GRP-[TeamName]-WellnessBot

---

## SECTION 1 : PROJECT TITLE

### WellnessBot — Neuro-Symbolic AI Chatbot for Arthroscopic Meniscal Repair Rehabilitation

---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT

Arthroscopic meniscal repair is one of the most commonly performed orthopaedic procedures, yet patient adherence to post-operative rehabilitation remains a persistent clinical challenge. Without guided exercise progression, patients risk re-injury, delayed recovery, or unsafe exertion — outcomes that are costly both medically and personally. Access to personalised, on-demand rehabilitation guidance is typically constrained by clinic appointment schedules, leaving patients with little structured support between visits.

WellnessBot is a neuro-symbolic hybrid AI chatbot designed to bridge this gap. It provides patients with contextually appropriate, evidence-grounded exercise recommendations during arthroscopic knee rehabilitation, adapting dynamically to each user's surgery type, recovery phase, reported symptoms, available equipment, and exercise history.

The system integrates eight tightly coupled pipeline components. Natural Language Understanding (NLU) is performed via OpenAI's GPT-4o-mini to extract structured clinical slots from free-form patient input. A slot-filling dialogue manager handles missing or ambiguous information through targeted follow-up questions. User state inference maps the patient's surgery date and type to a clinical phase (Phase 1 through Phase 5). A forward-chaining rule engine — operating under a strict ESCALATE > FORBID > CLARIFY > RECOMMEND dominance order — enforces safety gates before any exercise is recommended. When the rule engine clears a recommendation, a priority-based heuristic planner selects the most appropriate exercise, walking a three-tier priority structure and applying a pain-downgrade cascade when mild pain is reported. BM25-based retrieval-augmented generation (RAG) fetches grounded evidence chunks from a curated clinical database, and a final LLM generation step produces a validated, patient-facing response grounded exclusively in retrieved evidence.

Safety is a first-class design concern throughout. Red-flag symptoms (fever, active bleeding) trigger immediate escalation before any rule evaluation. Pain and swelling thresholds gate the planner's input. LLM output is validated to reject hallucinated sections and internal token leakage. All interactions are logged to a structured audit trail for post-hoc clinical review and EU AI Act explainability compliance.

The system is deployed as a Streamlit web application and was developed as a Master of AI Systems group project, with each team member owning a discrete pipeline component and one member responsible for final end-to-end integration.

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name | Student ID (MTech Applicable) | Work Items (Who Did What) | Email (Optional) |
|--------------------|-------------------------------|---------------------------|------------------|
| Member 1 | | | |
| Member 2 | | | |
| Member 3 | | | |
| Member 4 | | | |
| Member 5 | | | |

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

[WellnessBot System Demo — link to be added]

> Note: It is not mandatory for every project member to appear in the video presentation; presentation by one project member is acceptable.

---

## SECTION 5 : USER GUIDE

Refer to appendix **Installation & User Guide** in the project report at Github Folder: `ProjectReport`

### Prerequisites

- Python 3.10+
- An OpenAI API key (set in `SystemCode/.env`)

### [ 1 ] Extract the zip archive

Extract the submitted zip to a location of your choice. Then open a terminal and navigate into the `SystemCode` folder — **all commands below must be run from inside `SystemCode/`**:

```powershell
# Windows
cd path\to\WellnessBot_v2\SystemCode
```
```bash
# macOS / Linux
cd path/to/WellnessBot_v2/SystemCode
```

### [ 2 ] Set up a virtual environment and install dependencies

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
```bash
# macOS / Linux
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### [ 3 ] Configure environment variables

Copy the example file and fill in your OpenAI API key:

```powershell
# Windows
Copy-Item .env.example .env
```
```bash
# macOS / Linux
cp .env.example .env
```

Open `SystemCode/.env` and set your key:
```dotenv
OPENAI_API_KEY=sk-proj-...
```

To run without an API key, use Mock Mode instead:
```dotenv
MOCK_NLU=1
```

### [ 4 ] Run the Streamlit application

**Windows (PowerShell) — from inside `SystemCode/`:**
```powershell
.\run_streamlit.ps1
```

If PowerShell blocks script execution, run this once first:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**macOS / Linux — from inside `SystemCode/`:**
```bash
PYTHONPATH=.. streamlit run wellnessbot/Wellnessbot.py
```

Open your browser at `http://localhost:8501`

### [ 5 ] Run the test suite

**Windows — from inside `SystemCode/`:**
```powershell
$env:PYTHONPATH = (Get-Item ..).FullName
$env:MOCK_NLU = "1"
pytest wellnessbot/tests/ -v
```

**macOS / Linux — from inside `SystemCode/`:**
```bash
MOCK_NLU=1 PYTHONPATH=.. pytest wellnessbot/tests/ -v
```

### [ 6 ] Run the log analysis script

**Windows — from inside `SystemCode/`:**
```powershell
$env:PYTHONPATH = (Get-Item ..).FullName
python wellnessbot/scripts/analyze_logs.py
```

**macOS / Linux — from inside `SystemCode/`:**
```bash
PYTHONPATH=.. python wellnessbot/scripts/analyze_logs.py
```

---

## SECTION 6 : PROJECT REPORT / PAPER

Refer to project report at Github Folder: `ProjectReport`

### Recommended Sections for Project Report / Paper

1. Executive Summary / Paper Abstract
2. Business Problem Background
3. Market Research
4. Project Objectives & Success Measurements
5. Project Solution — Domain Modelling & System Design
   - Pipeline architecture (8 components)
   - Neuro-symbolic hybrid reasoning design
   - Knowledge graph & clinical protocol modelling
   - Safety mechanism design
6. Project Implementation — System Development & Testing
   - NLU & dialogue slot-filling
   - Rule engine & dominance ordering
   - Priority-based heuristic planner & pain-downgrade cascade
   - BM25 RAG retrieval & evidence grounding
   - LLM response generation & validation
7. Project Performance & Validation
8. Project Conclusions: Findings & Recommendations
9. Appendix: Project Proposal
10. Appendix: Mapped System Functionalities against Knowledge, Techniques and Skills of Modular Courses
11. Appendix: Installation and User Guide
12. Appendix: Individual Project Report (per member)
    - Personal contribution to group project
    - Most useful learnings
    - Application of knowledge in other situations or workplaces
13. Appendix: List of Abbreviations
14. Appendix: References

---

## SECTION 7 : MISCELLANEOUS

Refer to Github Folder: `SystemCode/data`

### System Architecture — Pipeline Overview

```
User Input
  → [1] NLU Extraction          (OpenAI GPT-4o-mini / Mock)
  → [2] Dialog Merge            (Slot filling into ConversationState)
  → [3] CLARIFY Check           (Missing slots trigger question loop)
  → [4] User State Inference    (Derives phase_id, risk_flags)
  → [5] Rule Engine             (RECOMMEND / CLARIFY / FORBID / ESCALATE)
  → [6] Intervention Planner    (Exercise selection — RECOMMEND path only)
  → [7] RAG Retrieval           (BM25 evidence chunk lookup)
  → [8] LLM Response Generation (Patient-facing validated output)
  → User Output (Streamlit chat UI)
```

### Four System Behaviours

| Behaviour | Trigger | Description |
|-----------|---------|-------------|
| **RECOMMEND** | Safe input, sufficient information | Exercise selected and evidence-grounded response generated |
| **CLARIFY** | Missing or ambiguous slots | Follow-up question posed; partial exercise alternatives shown |
| **FORBID** | Medical clearance gate triggered | Exercise blocked with explanation |
| **ESCALATE** | Red-flag symptoms detected | Immediate referral to clinical professional advised |

### Key Technologies

| Component | Technology / Technique |
|-----------|------------------------|
| NLU | LLM semantic extraction (OpenAI GPT-4o-mini) |
| Dialogue Management | Slot-filling with pending follow-up queue |
| User State Inference | Rule-based phase mapping from surgery date |
| Rule Engine | Forward-chaining with ESCALATE > FORBID > CLARIFY > RECOMMEND dominance |
| Intervention Planner | Priority-based tier walking with pain-downgrade cascade (Heuristic Search) |
| RAG Retrieval | BM25 keyword search with exact chunk-ID boosting |
| LLM Response | GPT-4o-mini with evidence grounding and output validation |
| UI | Streamlit |
| Audit Trail | Structured JSONL interaction logging |

### Known Limitations

| ID | Limitation |
|----|------------|
| L1 | Time-based phase mapping (not criterion-based) |
| L2 | Self-reported, unverified user inputs |
| L3 | Stateless server — inter-session history via file store only |
| L4 | Single exercise recommended per session |
| L5 | Hand-tuned priority tiers — not empirically validated |
| L6 | No real-time protocol updates (static knowledge graph) |
| L7 | No EHR integration |
