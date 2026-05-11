# SOW Risk & Readiness Command Center

Streamlit app for SOW governance, delivery readiness, and compliance intelligence using CrewAI and Groq.

## What It Does

- Runs a SOW readiness audit for PII, process-circle alignment, and pre-kickoff compliance signals.
- Checks SOW quality for missing sections and ambiguous clauses.
- Compares uploaded MSA and PO documents against the SOW for consistency risks.
- Generates MAVCA task intelligence for delivery planning.
- Supports PII redaction and an audit-style leaderboard/history view.

## Project Layout

- `app.py` - Streamlit UI and workflow orchestration.
- `audit_crew.py` - CrewAI audit and PII remediation agents.
- `phase_agents.py` - data quality, MAVCA, MSA, and PO consistency logic.
- `documents.py` - PDF/DOCX text extraction helpers.
- `knowledge_base/` - process-circle and policy reference material.
- `fixtures/` - sample SOW/PO inputs for demos and testing.

## Run Locally

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Environment Variables

Create a local `.env` file for development. Do not commit it to GitHub.

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
STAKEHOLDER_WEBHOOK_URL=
```

`GROQ_API_KEY` is required to run the AI workflows. `STAKEHOLDER_WEBHOOK_URL` is optional.

## Streamlit Community Deployment

- Main file path: `app.py`
- Python version: defined in `runtime.txt`
- Dependencies: `requirements.txt`
- Configure secrets in Streamlit Community secrets, not in this repository.

Example Streamlit secret:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

## GitHub Check-In

Before pushing, verify no secrets or local runtime files are staged:

```powershell
git status --short
git add .gitignore README.md requirements.txt app.py audit_crew.py phase_agents.py documents.py env_config.py groq_llm.py json_utils.py redaction.py knowledge_base fixtures Dockerfile runtime.txt TROUBLESHOOTING.md oasis_alpha_revised_sow.md
git status --short
git commit -m "Clean up Streamlit SOW governance app for GitHub"
git push origin main
```

If `data/sow_run_history.json` contains real run history, keep it out of GitHub.
