# IRS-PM-2026-01-24-GRP7-TheStacks-WellnessBot

---

## SECTION 1 : PROJECT TITLE

### WellnessBot - Rehabilitation Decision Support System

---

## SECTION 2 : EXECUTIVE SUMMARY / PAPER ABSTRACT

Discharged patients undergoing post-arthroscopic knee rehabilitation often lack continuous clinical support, increasing the risk of improper recovery and unsafe exercise practices. This project presents Wellness Bot, a home-based virtual decision-support assistant designed to provide safe, personalized rehabilitation guidance and self-care recommendations.

The system adopts a neuro-symbolic approach, combining a structured knowledge graph and rule-based reasoning to ensure deterministic and explainable decision-making. Unlike purely LLM-based systems, Wellness Bot follows a decision-first workflow, where recommendations are generated based on explicit clinical rules and constraints. A retrieval-augmented generation (RAG) module is incorporated to provide evidence-based explanations, while the large language model is restricted to natural language understanding and explanation. This ensures that all critical decisions remain controlled, transparent, and auditable.

The system is evaluated using a multi-layer framework, including decision-level performance through confusion matrix analysis and recommendation-level validation based on rehabilitation phase and user condition. In addition, an offline knowledge loop is implemented to analyse interaction logs, identify recurring issues, and refine rules and knowledge in a controlled manner.

Overall, Wellness Bot demonstrates a safe, interpretable, and modular approach to rehabilitation support, bridging the gap between research and real-world application while enabling continuous improvement through structured and traceable system refinement.


---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name | Student ID (MTech Applicable) | Work Items (Who Did What) | Email (Optional) |
|--------------------|-------------------------------|---------------------------|------------------|
| Loo Kin Xing | A0073762B | • Led system architecture and technical direction<br>• Coordinated integration and API contracts<br>• Designed KG and rule engine<br>• Built input pipeline, validation and logging<br>• Engineered NLU extraction and output validation<br>• Developed knowledge loop (pattern mining)<br>• Ensured system auditability| GC.Kinxing.Loo@u.nus.edu |
| Sherwin Tang Wei Hao | A0125052W | •	Designed and implemented informed search planner<br>•	Evaluated search strategies<br>•	Orchestrated end-to-end pipeline integration<br>•	Developed UI and translated system decisions into user responses<br>•	Handled errors, edge cases and fallback logic<br>•	Led integration testing<br>•	Manage end-to-end debugging| sherwin.wei.hao.tang@u.nus.edu |
| Foo Chit Yan | A0341007W | •	Designed and implemented RAG pipeline<br>•	evaluated retrieval architecture<br>•	engineered prompts for grounded explanations<br>•	implemented output validation<br>•	conducted retrieval quality evaluation<br>•	ensured hallucination control and explanation reliability| foochityan@u.nus.edu |
| Foo Swee Yoong | A0341027R |  •	Domain lead<br>•	Ensured KG/rule correctness & system quality<br>•	Designed data schema, edge cases and clinical test scenarios<br>•	curated and structured clinical evidence<br>•	Defined ground truth and evaluation metrics<br>•	Conducted scenario-based and safety validation<br>•	Prepared demo scenarios and evaluation results | foosweeyoong@u.nus.edu |

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

[WellnessBot Promotional Video](https://1drv.ms/v/c/2d6153e9eaa15534/IQDJbAQ6vD-oQKfvw-SWmv57AUog6XFTj3iSxaQp9OGtnzw?e=QycnXj)

[WellnessBot System Design Video](https://1drv.ms/v/c/2d6153e9eaa15534/IQD5bBuOL4SjQpUV1__XpuTzAfGRUpQ2a35kHwITJ9nvVfo?e=4uwZFb)

Refer to **video** at Github Folder: `Video`

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
cd path\to\IRS-PM-2026-01-24-GRP7-WellnessBot\SystemCode
```
```bash
# macOS / Linux
cd path/to/IRS-PM-2026-01-24-GRP7-WellnessBot/SystemCode
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

---

## SECTION 6 : PROJECT REPORT / PAPER

Refer to project report at Github Folder: `ProjectReport`

### Recommended Sections for Project Report / Paper

•	Abstract<br>
•	Project Introduction<br>
•	Project Background & Market Research<br>
•	Project Scope<br>
•	Data Collection and Preparation<br>
•	System Design<br>
•	Implementation<br>
•	System Demonstration<br>
•	Implementation Challenges and Resolutions<br>
•	Logging and Auditability<br>
•	Knowledge Loop and Mining<br>
•	Result and Progress<br>
•	Project Conclusion<br>
•	Appendix of report: Project Proposal<br>
•	Appendix of report: Mapped System Functionalities<br>
•	Appendix of report: Installation and User Guide<br>
•	Appendix of report: List of Tested Scenario<br>


---

## SECTION 7 : MISCELLANEOUS

Refer to Github Folder: `Miscellaneous`

LLM Baseline Evaluation<br>
•	Contains scripts, test scenarios, and outputs for evaluating LLM decision performance against ground truth, serving as a benchmark for comparison with WellnessBot.
