"""
SOW Risk & Readiness Command Center — Streamlit UI for SOW audit (CrewAI) and PII redaction.
"""

from __future__ import annotations

import json
import io
import re
import uuid
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import streamlit as st

from documents import read_docx_text, read_pdf_text
from env_config import (
    get_groq_api_key,
    get_stakeholder_webhook_url,
    load_dotenv_and_resolve,
    notify_stakeholder_webhook,
)

load_dotenv_and_resolve()

from audit_crew import run_audit_crew, run_remediation_crew
from approved_sow_archive import archive_approved_sow_text
from phase_agents import (
    run_data_quality_precheck,
    run_mavca_decomposition,
    run_msa_consistency_check,
    run_po_sow_consistency_check,
    validate_phase1b_mavca,
)
from rag_store import (
    STEP_QUERIES,
    build_corpus,
    corpus_fingerprint,
    format_retrieved_context,
    merge_chunks_unique,
    rag_evidence_rows,
    retrieve,
)
from redaction import redact_docx_bytes

ROOT = Path(__file__).resolve().parent
DEFAULT_KB = ROOT / "knowledge_base" / "process_circles.md"
DEFAULT_ETHICS_PDF = ROOT / "knowledge_base" / "ASCENDION_AI_HYGIENE_AND_ETHICS_MANUAL_2026.pdf"
RUN_HISTORY_PATH = ROOT / "data" / "sow_run_history.json"


def _load_kb_text(upload) -> str:
    if upload is not None:
        return upload.getvalue().decode("utf-8", errors="replace")
    if DEFAULT_KB.is_file():
        return DEFAULT_KB.read_text(encoding="utf-8")
    return ""


def _load_ethics_manual_text(upload) -> str:
    """Prefer uploaded PDF; else use on-disk file in knowledge_base/ if present."""
    if upload is not None:
        return read_pdf_text(upload.getvalue())
    if DEFAULT_ETHICS_PDF.is_file():
        return read_pdf_text(DEFAULT_ETHICS_PDF.read_bytes())
    return ""


def _looks_like_po_filename(lower_name: str) -> bool:
    return bool(
        re.search(r"(^|[^a-z0-9])po([^a-z0-9]|$)", lower_name)
        or re.search(r"purchase[\s_-]*order", lower_name)
    )


def _classify_uploaded_inputs(uploaded_files: list[Any] | None) -> dict[str, Any]:
    """Single-uploader classifier for SOW/MSA/PDF/KB inputs."""
    sow_bytes: bytes | None = None
    sow_name: str | None = None
    kb_text = ""
    ethics_text = ""
    msa_text = ""
    msa_name: str | None = None
    po_text = ""
    po_name: str | None = None

    files = uploaded_files or []
    for f in files:
        name = str(getattr(f, "name", "") or "")
        lower = name.lower()
        suffix = Path(lower).suffix
        data = f.getvalue()

        if suffix == ".md":
            if not kb_text:
                kb_text = data.decode("utf-8", errors="replace")
            continue

        if suffix == ".pdf":
            text = read_pdf_text(data)
            if _looks_like_po_filename(lower) and not po_text:
                po_text = text
                po_name = name
            elif ("msa" in lower or "master services" in lower) and not msa_text:
                msa_text = text
                msa_name = name
            elif not ethics_text:
                ethics_text = text
            continue

        if suffix == ".docx":
            if _looks_like_po_filename(lower) and not po_text:
                po_text = read_docx_text(data)
                po_name = name
            elif ("msa" in lower or "master services" in lower) and not msa_text:
                msa_text = read_docx_text(data)
                msa_name = name
            elif sow_bytes is None:
                sow_bytes = data
                sow_name = name
            elif "sow" in lower and sow_name and "sow" not in sow_name.lower():
                sow_bytes = data
                sow_name = name
            continue

    if not kb_text and DEFAULT_KB.is_file():
        kb_text = DEFAULT_KB.read_text(encoding="utf-8")
    if not ethics_text and DEFAULT_ETHICS_PDF.is_file():
        ethics_text = read_pdf_text(DEFAULT_ETHICS_PDF.read_bytes())

    return {
        "sow_bytes": sow_bytes,
        "sow_name": sow_name,
        "kb_text": kb_text,
        "ethics_text": ethics_text,
        "msa_text": msa_text,
        "msa_name": msa_name,
        "po_text": po_text,
        "po_name": po_name,
    }


def _init_session() -> None:
    if st.session_state.get("_del_ex_init"):
        return
    st.session_state._del_ex_init = True
    st.session_state.audit_result = None
    st.session_state.sow_bytes = None
    st.session_state.sow_name = None
    st.session_state.redacted_buf = None
    st.session_state.redaction_mapping = None
    st.session_state.last_upload_sig = None
    st.session_state.mavca_result = None
    st.session_state.phase1b_gate_result = None
    st.session_state.data_quality_result = None
    st.session_state.msa_consistency_result = None
    st.session_state.po_text = ""
    st.session_state.po_name = None
    st.session_state.po_consistency_result = None
    st.session_state.manual_go_ahead_override = False
    st.session_state.manual_go_ahead_reason = ""
    st.session_state.manual_go_ahead_actor = ""
    st.session_state.last_audit_run_id = ""
    st.session_state.insights_sow_plaintext = ""
    st.session_state.rag_retrieval_by_step = {}
    st.session_state.rag_corpus_fingerprint = ""
    st.session_state.rag_archive_done_run_id = ""


def _tl_emoji(status: str) -> str:
    s = (status or "").upper()
    return "🟢 PASS" if s == "PASS" else "🔴 RED"


def now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_local_datetime(value: str) -> str:
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return value


def load_run_history() -> list[dict[str, Any]]:
    if not RUN_HISTORY_PATH.is_file():
        return []
    try:
        data = json.loads(RUN_HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def save_run_history(rows: list[dict[str, Any]]) -> None:
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUN_HISTORY_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


def append_run_history(row: dict[str, Any]) -> None:
    rows = load_run_history()
    rows.append(row)
    save_run_history(rows)


def update_run_history_entry(run_id: str, updates: dict[str, Any]) -> None:
    if not run_id or not isinstance(updates, dict):
        return
    rows = load_run_history()
    changed = False
    for r in rows:
        if str(r.get("run_id", "")) == run_id:
            r.update(updates)
            changed = True
            break
    if changed:
        save_run_history(rows)


def reset_run_history() -> None:
    save_run_history([])


def _parse_common_date(date_text: str) -> datetime | None:
    raw = (date_text or "").strip()
    if not raw:
        return None
    normalized = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", raw, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip(" ,.")
    for fmt in (
        "%b %d, %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%m/%d/%Y",
        "%m/%d/%y",
        "%m-%d-%Y",
        "%m-%d-%y",
    ):
        try:
            return datetime.strptime(normalized, fmt)
        except ValueError:
            continue
    return None


def extract_project_name_from_sow(text: str, fallback: str = "") -> str:
    m = re.search(r"(?im)^\s*\d+\.\s*Project\s*[-–—:]\s*(.+?)\s*$", text)
    if m:
        return m.group(1).strip(" -:\t")
    m = re.search(
        r"(?is)Project\s*Summary\s*:?\s*(.+?)(?:\n\s*(?:Scope of Services|Out of Scope|Deliverables)\b|\n_{3,}|\Z)",
        text,
    )
    if m:
        line = next((ln.strip() for ln in m.group(1).splitlines() if ln.strip()), "")
        if line:
            return re.sub(r"^\s*Project\s*[-–—:]\s*", "", line, flags=re.IGNORECASE).strip()
    m = re.search(r"(?im)^\s*Project\s*[-–—:]\s*(.+?)\s*$", text)
    if m:
        return m.group(1).strip(" -:\t")
    return (fallback or "").strip()


def extract_timeline_term(text: str) -> str:
    m = re.search(
        r"(?is)(?:^\s*\d+\.\s*)?Timeline\s*/\s*Term\s*\n(.*?)(?:\n\s*_{3,}|\n\s*\d+\.\s+[A-Za-z]|$)",
        text,
    )
    if not m:
        return ""
    block = m.group(1)
    cleaned = " ".join(part.strip() for part in block.splitlines() if part.strip())
    return re.sub(r"\s+", " ", cleaned).strip()


def parse_timeline_end_date(value: str) -> str | None:
    if not value:
        return None
    pattern = (
        r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+\d{2,4})\b"
    )
    candidates = re.findall(pattern, value, flags=re.IGNORECASE)
    parsed = [dt for dt in (_parse_common_date(c) for c in candidates) if dt is not None]
    if not parsed:
        return None
    return max(parsed).date().isoformat()


def extract_estimated_contract_value(text: str) -> tuple[str, float | None]:
    m = re.search(r"(?i)Estimated\s+Contract\s+Value\s*:\s*\$?\s*([0-9][0-9,]*(?:\.\d+)?)", text)
    if not m:
        return "", None
    raw_num = m.group(1).replace(",", "")
    try:
        amount = float(raw_num)
    except ValueError:
        amount = None
    display = f"${m.group(1)}"
    return display, amount


def _build_run_record(sow_name_file: str, sow_text: str, audit_result: dict[str, Any]) -> dict[str, Any]:
    timeline_raw = extract_timeline_term(sow_text)
    cost_display, cost_value = extract_estimated_contract_value(sow_text)
    run_id = str(uuid.uuid4())
    base_go_ahead = bool(audit_result.get("executive_go_ahead", False))
    row: dict[str, Any] = {
        "run_id": run_id,
        "record_type": "audit",
        "sow_name_file": sow_name_file or "",
        "project_name": extract_project_name_from_sow(sow_text, fallback=sow_name_file or ""),
        "timeline_raw": timeline_raw,
        "timeline_end_iso": parse_timeline_end_date(timeline_raw),
        "estimated_contract_value": cost_display,
        "estimated_contract_value_numeric": cost_value,
        "circle": str((audit_result.get("agent2") or {}).get("process_circle", "") or ""),
        "sow_ready_score": float(audit_result.get("sow_ready_score", 0) or 0),
        "msa_consistency_status": str(
            ((audit_result.get("msa_consistency") or {}).get("status") if isinstance(audit_result.get("msa_consistency"), dict) else "")
            or ""
        ),
        "po_consistency_status": str(
            ((audit_result.get("po_consistency") or {}).get("status") if isinstance(audit_result.get("po_consistency"), dict) else "")
            or ""
        ),
        "po_conflict_count": int(
            ((audit_result.get("po_consistency") or {}).get("conflict_count") if isinstance(audit_result.get("po_consistency"), dict) else 0)
            or 0
        ),
        "go_ahead_system": "PASS" if base_go_ahead else "RED",
        "go_ahead_effective": "PASS" if base_go_ahead else "RED",
        "go_ahead_override_applied": False,
        "go_ahead_override_reason": "",
        "go_ahead_override_actor": "",
        "last_run_crewai": now_iso_utc(),
    }
    for opt_key in (
        "rag_corpus_fingerprint",
        "rag_chunk_ids_by_step",
        "approved_sow_archive_path",
    ):
        if opt_key in audit_result:
            row[opt_key] = audit_result[opt_key]
    return row


def _build_phase1b_run_record(
    sow_name_file: str,
    sow_text: str,
    mavca_result: dict[str, Any],
    gate_result: dict[str, Any],
) -> dict[str, Any]:
    timeline_raw = extract_timeline_term(sow_text)
    cost_display, cost_value = extract_estimated_contract_value(sow_text)
    tasks = mavca_result.get("tasks") if isinstance(mavca_result.get("tasks"), list) else []
    shifts = (
        mavca_result.get("classification_shifts")
        if isinstance(mavca_result.get("classification_shifts"), list)
        else []
    )
    return {
        "run_id": str(uuid.uuid4()),
        "record_type": "phase1b_mavca",
        "sow_name_file": sow_name_file or "",
        "project_name": extract_project_name_from_sow(sow_text, fallback=sow_name_file or ""),
        "timeline_raw": timeline_raw,
        "timeline_end_iso": parse_timeline_end_date(timeline_raw),
        "estimated_contract_value": cost_display,
        "estimated_contract_value_numeric": cost_value,
        "circle": "",
        "sow_ready_score": float(gate_result.get("score", 0) or 0),
        "last_run_crewai": now_iso_utc(),
        "mavca_output": mavca_result,
        "phase1b_gate_result": gate_result,
        "phase1b_tasks_count": len(tasks),
        "phase1b_shift_count": len(shifts),
        "phase1b_status": str(mavca_result.get("status", "") or ""),
        "phase1b_gate_status": str(gate_result.get("status", "") or ""),
    }


def _status_pass(v: Any) -> bool:
    return str(v or "").upper() == "PASS"


def _task_level_recommendation(level: str) -> str:
    lvl = str(level or "").strip().lower()
    if lvl == "autonomous":
        return "Define policy guardrails and production rollback controls."
    if lvl == "validated":
        return "Add a human validation checkpoint before sign-off."
    if lvl == "augmented":
        return "Provide tool-assisted execution with explicit owner approval."
    if lvl == "curated":
        return "Standardize reusable playbooks and peer review checkpoints."
    return "Keep human-owned execution with clear acceptance criteria."


def _component_display_name(component: str) -> str:
    raw = str(component or "")
    if "—" in raw:
        return raw.split("—", 1)[1].strip()
    return raw.strip()


def _score_breakdown_consolidated(
    result: dict[str, Any],
    dq: dict[str, Any],
    mavca: dict[str, Any],
    msa_consistency: dict[str, Any],
    policy_compliance: dict[str, Any],
) -> list[dict[str, Any]]:
    a1 = result.get("agent1") if isinstance(result.get("agent1"), dict) else {}
    a2 = result.get("agent2") if isinstance(result.get("agent2"), dict) else {}
    a3 = result.get("agent3") if isinstance(result.get("agent3"), dict) else {}
    # Policy 101/102/103/104 row is a derived roll-up; do not score it (see "non-scoring" filter below).
    # Executive summary and PII remediation are not separate score rows.
    rows = [
        ("SOW controls — PII detection", _status_pass(a1.get("status", "RED"))),
        ("SOW controls — Process circle & deliverables", _status_pass(a2.get("status", "RED"))),
        ("SOW controls — Pre-kickoff compliance gate", _status_pass(a3.get("status", "RED"))),
        ("Planning — Task intelligence (MAVCA)", _status_pass(mavca.get("status", "RED"))),
        ("Input readiness — Data quality precheck", _status_pass(dq.get("status", "RED"))),
        ("Contracts — MSA vs SOW consistency", _status_pass(msa_consistency.get("status", "RED"))),
        (
            "Compliance — Policy 101/102/103/104 (derived from checks, non-scoring)",
            _status_pass(policy_compliance.get("overall_status", "RED")),
        ),
    ]
    scoring_rows = [r for r in rows if "non-scoring" not in r[0]]
    contribution = round(100.0 / len(scoring_rows), 2)
    result_rows = [
        {
            "Component": label,
            "Weight": 0.0 if "non-scoring" in label else contribution,
            "Status": "PASS" if passed else "RED",
            "Score contribution": (0.0 if "non-scoring" in label else (contribution if passed else 0.0)),
        }
        for label, passed in rows
    ]
    return result_rows


def _compliance_audit_trail(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for r in history[-10:]:
        rows.append(
            {
                "Run time": format_local_datetime(str(r.get("last_run_crewai", "") or "")),
                "Run type": str(r.get("record_type", "") or ""),
                "SOW": str(r.get("project_name") or r.get("sow_name_file") or ""),
                "PII gate": str(r.get("pii_gate", "") or ""),
                "Traffic light": str(r.get("traffic_light", "") or ""),
                "PO consistency": str(r.get("po_consistency_status", "") or ""),
                "Score": str(r.get("sow_ready_score", "")),
            }
        )
    return list(reversed(rows))


def build_leaderboard_rows(run_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in run_history:
        project_name = str(row.get("project_name", "") or "").strip()
        sow_name_file = str(row.get("sow_name_file", "") or "").strip()
        key = (project_name or sow_name_file or "unknown").lower()
        grouped.setdefault(key, []).append(row)

    rows: list[dict[str, Any]] = []
    for _, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: str(r.get("last_run_crewai", "")))
        latest = items_sorted[-1]
        timeline_end = str(latest.get("timeline_end_iso", "") or "")
        timeline_ord = 0
        if timeline_end:
            try:
                timeline_ord = datetime.fromisoformat(timeline_end).toordinal()
            except Exception:
                timeline_ord = 0
        cost_num = latest.get("estimated_contract_value_numeric")
        if not isinstance(cost_num, (int, float)):
            cost_num = 0.0
        rows.append(
            {
                "SOW name": str(latest.get("project_name") or latest.get("sow_name_file") or "Unknown"),
                "SOW timeline": str(latest.get("timeline_raw", "") or ""),
                "Estimated contract value": str(latest.get("estimated_contract_value", "") or ""),
                "Process circle": str(latest.get("circle", "") or ""),
                "SOW score": float(latest.get("sow_ready_score", 0) or 0),
                "PO consistency": str(latest.get("po_consistency_status", "") or ""),
                "PO conflicts": int(latest.get("po_conflict_count", 0) or 0),
                "Last run by CrewAI (DateTime)": format_local_datetime(
                    str(latest.get("last_run_crewai", "") or "")
                ),
                "# of runs by CrewAI": len(items),
                "_timeline_missing": 0 if timeline_ord else 1,
                "_timeline_ord": timeline_ord,
                "_cost_num": float(cost_num),
            }
        )

    rows.sort(
        key=lambda r: (
            int(r["_timeline_missing"]),
            -int(r["_timeline_ord"]),
            -float(r["_cost_num"]),
            str(r["SOW name"]).lower(),
        )
    )
    for row in rows:
        row.pop("_timeline_missing", None)
        row.pop("_timeline_ord", None)
        row.pop("_cost_num", None)
    return rows


def leaderboard_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = StringIO()
    headers = list(rows[0].keys())
    output.write(",".join(f'"{h}"' for h in headers) + "\n")
    for row in rows:
        vals = [str(row.get(h, "")).replace('"', '""') for h in headers]
        output.write(",".join(f'"{v}"' for v in vals) + "\n")
    return output.getvalue()


def main() -> None:
    st.set_page_config(page_title="SOW Risk & Readiness Command Center", layout="wide")
    _init_session()
    st.markdown(
        """
<style>
html, body, [data-testid="block-container"] {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 0.95rem;
    line-height: 1.5;
}

[data-testid="block-container"] {
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}

section[data-testid="stSidebar"] {
    background: #f9fafb;
    color: #111827;
    border-right: 1px solid #e5e7eb;
}

section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    color: #111827;
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #4b5563;
}

section[data-testid="stSidebar"] button[kind="primary"] {
    color: #ffffff !important;
}

div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
    border: 1px solid #dbeafe;
    border-radius: 14px;
    padding: 14px;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
}
div[data-testid="stMetricLabel"] > div {
    color: #1e3a8a !important;
    font-weight: 600 !important;
}
div[data-testid="stMetricValue"] > div {
    color: #0f172a !important;
    font-size: 2rem !important;
}

[data-testid="stHeader"] {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 55%, #38bdf8 100%);
    padding: 1.25rem 1.5rem 1rem 1.5rem;
}

[data-testid="stHeader"] h1 {
    color: #f9fafb !important;
    letter-spacing: 0.02em;
}

[data-testid="stHeader"] [data-testid="stCaptionContainer"] {
    color: #e0f2fe !important;
}

div[data-baseweb="tab-list"] {
    margin-top: 0.75rem;
    border-bottom: 1px solid #e5e7eb;
}

div[data-baseweb="tab"] {
    padding-top: 0.75rem;
    padding-bottom: 0.75rem;
}

div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stDataFrame"]) {
    background: #ffffff;
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    padding: 0.75rem;
}

div[data-testid="stExpander"] {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    background: #ffffff;
}

div[data-testid="stExpander"] > summary {
    font-weight: 600;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}
</style>
        """,
        unsafe_allow_html=True,
    )

    st.title("SOW Risk & Readiness Command Center")
    st.caption("CIO/CXO View: AI governance, delivery readiness, and compliance intelligence")

    with st.sidebar:
        st.subheader("Workflow Inputs")
        st.caption(
            "Workflow: Data quality → SOW insights → MSA/PO consistency → "
            "task intelligence → policy consolidation."
        )
        uploaded_files = st.file_uploader(
            "Upload project files (single placeholder)",
            type=["docx", "pdf", "md"],
            accept_multiple_files=True,
            help=(
                "Upload SOW (.docx), PO (.docx/.pdf), MSA (.docx/.pdf), ethics manual (.pdf), and process circles (.md) here."
            ),
        )
        run_audit = st.button("1) Run Insights Audit", type="primary", use_container_width=True)
        st.divider()
        classified = _classify_uploaded_inputs(uploaded_files)
        if get_stakeholder_webhook_url():
            st.caption("Stakeholder webhook: configured (`STAKEHOLDER_WEBHOOK_URL`).")

    sig = None
    if uploaded_files:
        file_sig = tuple(sorted((f.name, f.size) for f in uploaded_files))
        sig = (
            file_sig,
        )
    if sig != st.session_state.last_upload_sig:
        st.session_state.last_upload_sig = sig
        st.session_state.audit_result = None
        st.session_state.redacted_buf = None
        st.session_state.redaction_mapping = None
        st.session_state.mavca_result = None
        st.session_state.phase1b_gate_result = None
        st.session_state.data_quality_result = None
        st.session_state.msa_consistency_result = None
        st.session_state.po_consistency_result = None
        st.session_state.po_text = ""
        st.session_state.po_name = None
        st.session_state.manual_go_ahead_override = False
        st.session_state.manual_go_ahead_reason = ""
        st.session_state.manual_go_ahead_actor = ""
        st.session_state.last_audit_run_id = ""
        st.session_state.insights_sow_plaintext = ""
        st.session_state.rag_retrieval_by_step = {}
        st.session_state.rag_corpus_fingerprint = ""
        st.session_state.rag_archive_done_run_id = ""

    st.session_state.sow_bytes = classified.get("sow_bytes")
    st.session_state.sow_name = classified.get("sow_name")
    st.session_state.po_text = str(classified.get("po_text") or "")
    st.session_state.po_name = classified.get("po_name")
    kb_text = classified.get("kb_text") or ""
    ethics_text = classified.get("ethics_text") or ""
    msa_text = classified.get("msa_text") or ""
    po_text = str(classified.get("po_text") or "")

    def run_data_quality_guard(sow_text: str) -> None:
        with st.status("Running data quality pre-check...", expanded=False) as dq_status:
            dq = run_data_quality_precheck(sow_text)
            st.session_state.data_quality_result = dq
            dq_status.update(label="Data quality pre-check finished", state="complete")
        if str((st.session_state.data_quality_result or {}).get("status", "RED")).upper() != "PASS":
            st.warning("Data quality pre-check is RED. Review missing sections/ambiguities before running.")

    if run_audit:
        if not st.session_state.sow_bytes:
            st.warning("Upload project files including a SOW `.docx` first.")
        elif not get_groq_api_key():
            st.error("Missing Groq credentials. Set **`GROQ_API_KEY`** or **`test3`** in `.env`.")
        else:
            st.session_state.redacted_buf = None
            st.session_state.redaction_mapping = None
            st.session_state.po_consistency_result = None
            st.session_state.manual_go_ahead_override = False
            st.session_state.manual_go_ahead_reason = ""
            st.session_state.manual_go_ahead_actor = ""
            st.session_state.rag_archive_done_run_id = ""
            text = read_docx_text(st.session_state.sow_bytes)
            run_data_quality_guard(text)
            with st.status("Running unified governance workflow...", expanded=True) as status_box:
                def progress(msg: str) -> None:
                    st.write(msg)

                try:
                    st.session_state.insights_sow_plaintext = text
                    rag_chunks = build_corpus(
                        kb_text=kb_text,
                        ethics_text=ethics_text,
                        msa_text=msa_text,
                        po_text=po_text,
                        sow_text=text,
                        sow_source_label=st.session_state.sow_name or "sow.docx",
                        root=ROOT,
                    )
                    fp = corpus_fingerprint(rag_chunks)
                    st.session_state.rag_corpus_fingerprint = fp

                    ch_pc = retrieve(STEP_QUERIES["process_circle"], rag_chunks, 6)
                    ch_gov = retrieve(
                        "PII confidential privacy security controls data handling governance",
                        rag_chunks,
                        4,
                    )
                    ch_a2 = merge_chunks_unique([ch_pc, ch_gov], top_k=8)
                    gov_a2 = format_retrieved_context(ch_a2).strip() or None

                    ch_mv = retrieve(STEP_QUERIES["mavca"], rag_chunks, 8)
                    gov_mavca = format_retrieved_context(ch_mv).strip() or None

                    msa_rag = rag_evidence_rows(
                        retrieve(STEP_QUERIES["msa"], rag_chunks, 5),
                        reason="Lexical match for MSA / legal alignment",
                    )
                    po_rag = rag_evidence_rows(
                        retrieve(STEP_QUERIES["po"], rag_chunks, 5),
                        reason="Lexical match for PO / commercial alignment",
                    )
                    st.session_state.rag_retrieval_by_step = {
                        "process_circle": rag_evidence_rows(
                            ch_a2,
                            reason="Process circle, governance, and hygiene retrieval",
                        ),
                        "mavca": rag_evidence_rows(
                            ch_mv,
                            reason="MAVCA and task-intelligence retrieval",
                        ),
                        "msa": msa_rag,
                        "po": po_rag,
                    }

                    st.write("Step 1/4: SOW insights")
                    st.session_state.audit_result = run_audit_crew(
                        text,
                        kb_text=kb_text,
                        ethics_manual_text=ethics_text or None,
                        progress=progress,
                        retrieved_governance=gov_a2,
                    )
                    ar = st.session_state.audit_result
                    if isinstance(ar, dict) and "error" not in ar:
                        st.write("Step 2/5: MSA consistency check")
                        msa_result: dict[str, Any] | None = None
                        if msa_text.strip():
                            msa_result = run_msa_consistency_check(text, msa_text)
                            if isinstance(msa_result, dict):
                                msa_result = {**msa_result, "retrieved_evidence": msa_rag}
                        else:
                            msa_result = {
                                "status": "PASS",
                                "summary": "MSA input not provided; consistency check skipped as non-blocking.",
                                "conflicts": [],
                                "review_flags": [],
                                "conflict_count": 0,
                                "retrieved_evidence": msa_rag,
                            }
                        st.session_state.msa_consistency_result = msa_result
                        ar["msa_consistency"] = msa_result

                        st.write("Step 3/5: PO consistency check")
                        po_result: dict[str, Any]
                        if po_text.strip():
                            po_result = run_po_sow_consistency_check(text, po_text)
                            if isinstance(po_result, dict):
                                po_result = {**po_result, "retrieved_evidence": po_rag}
                        else:
                            po_result = {
                                "status": "PASS",
                                "summary": "PO input not provided; PO consistency check skipped (advisory only).",
                                "commercial_summary": {
                                    "po_total": None,
                                    "sow_estimated_contract_value": None,
                                    "variance_abs": None,
                                    "variance_pct": None,
                                    "within_tolerance": None,
                                    "tolerance_basis": "N/A — no PO uploaded",
                                    "tolerance_threshold_abs": None,
                                    "po_payment_net_days": None,
                                    "sow_payment_net_days": None,
                                    "payment_terms_aligned": None,
                                },
                                "conflicts": [],
                                "review_flags": [],
                                "recommendations": [],
                                "conflict_count": 0,
                                "retrieved_evidence": po_rag,
                            }
                        st.session_state.po_consistency_result = po_result
                        ar["po_consistency"] = po_result

                        st.write("Step 4/5: Task Intelligence decomposition")
                        mavca_result = run_mavca_decomposition(
                            text,
                            kb_text=kb_text,
                            ethics_text=ethics_text or None,
                            progress=progress,
                            retrieved_governance=gov_mavca,
                        )
                        gate_result = validate_phase1b_mavca(mavca_result if isinstance(mavca_result, dict) else {})
                        st.session_state.mavca_result = mavca_result
                        st.session_state.phase1b_gate_result = gate_result
                        append_run_history(
                            _build_phase1b_run_record(
                                st.session_state.sow_name or "",
                                text,
                                mavca_result if isinstance(mavca_result, dict) else {},
                                gate_result,
                            )
                        )

                        st.write("Step 5/5: Persisting audit run + notifying stakeholders")
                        ar["rag_corpus_fingerprint"] = fp
                        ar["rag_chunk_ids_by_step"] = {
                            k: [x.get("chunk_id") for x in v if isinstance(x, dict) and x.get("chunk_id")]
                            for k, v in st.session_state.rag_retrieval_by_step.items()
                            if isinstance(v, list)
                        }
                        run_record = _build_run_record(st.session_state.sow_name or "", text, ar)
                        append_run_history(run_record)
                        st.session_state.last_audit_run_id = str(run_record.get("run_id", ""))
                        notify_stakeholder_webhook(
                            "DelEx Sentinel — Audit complete",
                            payload={
                                "sow_name": st.session_state.sow_name,
                                "sow_ready_score": ar.get("sow_ready_score"),
                                "traffic_light": ar.get("traffic_light"),
                                "executive_go_ahead": ar.get("executive_go_ahead"),
                            },
                        )
                        st.session_state._just_completed_insights = True
                except Exception as e:
                    st.session_state.audit_result = {"error": str(e)}
                status_box.update(label="Unified workflow finished", state="complete")

    insights_done = isinstance(st.session_state.audit_result, dict) and "error" not in (st.session_state.audit_result or {})
    if st.session_state.get("_just_completed_insights"):
        st.session_state._just_completed_insights = False
        st.rerun()
    task_intel_done = isinstance(st.session_state.mavca_result, dict) and bool(st.session_state.mavca_result)
    step1_status = "Complete" if insights_done else "Pending"
    step2_status = "Complete" if task_intel_done else ("Running with Insights" if insights_done else "Pending")
    st.caption(
        f"Workflow: Data quality → SOW insights → MSA/PO consistency → Task intelligence → Policy consolidation. "
        f"(Status: Insights {step1_status}, Task Intelligence {step2_status})."
    )

    tab_insights, tab_leaderboard = st.tabs(["Insights", "Leaderboard"])

    with tab_insights:
        dq = st.session_state.data_quality_result or {}

        result = st.session_state.audit_result
        if result:
            if "error" in result:
                st.error(result["error"])
                return

            st.subheader("Insights — Master Scorecard")
            a1 = result.get("agent1") or {}
            a2 = result.get("agent2") or {}
            a3 = result.get("agent3") or {}
            policy_compliance = (
                result.get("policy_compliance")
                if isinstance(result.get("policy_compliance"), dict)
                else {}
            )
            msa_consistency = (
                result.get("msa_consistency")
                if isinstance(result.get("msa_consistency"), dict)
                else st.session_state.get("msa_consistency_result") or {}
            )
            po_consistency = (
                result.get("po_consistency")
                if isinstance(result.get("po_consistency"), dict)
                else st.session_state.get("po_consistency_result") or {}
            )
            mavca_for_score = st.session_state.mavca_result if isinstance(st.session_state.mavca_result, dict) else {}
            score_rows = _score_breakdown_consolidated(
                result=result,
                dq=dq if isinstance(dq, dict) else {},
                mavca=mavca_for_score,
                msa_consistency=msa_consistency if isinstance(msa_consistency, dict) else {},
                policy_compliance=policy_compliance if isinstance(policy_compliance, dict) else {},
            )
            score_raw = sum(float(r.get("Score contribution", 0) or 0) for r in score_rows)
            score = round(score_raw, 0)
            system_go_ahead = all(str(r.get("Status", "RED")).upper() == "PASS" for r in score_rows)
            blocking_components = [
                str(r.get("Component", ""))
                for r in score_rows
                if str(r.get("Status", "RED")).upper() != "PASS" and float(r.get("Weight", 0) or 0) > 0
            ]
            dq_only_blocking = (
                len(blocking_components) == 1
                and blocking_components[0] == "Input readiness — Data quality precheck"
            )

            dq_status = str((dq or {}).get("status", "RED")).upper() if dq else "N/A"
            m1, m2, m3 = st.columns(3)
            m1.metric("Data Quality Status", _tl_emoji(dq_status))
            m2.metric("SOW Ready Score (consolidated)", f"{score:.0f}%")
            override_eligible = (not system_go_ahead) and dq_only_blocking
            if not override_eligible and st.session_state.get("manual_go_ahead_override"):
                st.session_state.manual_go_ahead_override = False

            if override_eligible:
                with st.container(border=True):
                    st.markdown("**Delivery Manager / Account Executive decision override**")
                    st.caption(
                        "Only Data Quality is blocking go-ahead. Authorized approver may override to proceed "
                        "only if the **consolidated score is greater than 83%** — then Executive GO-Ahead becomes PASS "
                        "and the SOW text is archived under `knowledge_base/approved_sow/` for future RAG retrieval."
                    )
                    st.checkbox(
                        "Proceed despite Data Quality RED",
                        key="manual_go_ahead_override",
                    )
                    if st.session_state.manual_go_ahead_override:
                        st.text_input(
                            "Approver (name/email)",
                            key="manual_go_ahead_actor",
                            placeholder="delivery.manager@company.com",
                        )
                        st.text_area(
                            "Override rationale",
                            key="manual_go_ahead_reason",
                            placeholder="Business-critical timeline; risks acknowledged and accepted.",
                        )

            manual_override_applied = bool(
                st.session_state.get("manual_go_ahead_override") and override_eligible
            )
            if system_go_ahead:
                effective_go_ahead = True
            elif manual_override_applied:
                effective_go_ahead = float(score_raw) > 83.0
            else:
                effective_go_ahead = False
            if manual_override_applied:
                m3.metric(
                    "Executive GO-Ahead",
                    "🟢 PASS (Manual Override)"
                    if effective_go_ahead
                    else f"🔴 RED (need score > 83%; current {score_raw:.1f}%)",
                )
            else:
                m3.metric("Executive GO-Ahead", _tl_emoji("PASS" if effective_go_ahead else "RED"))

            update_run_history_entry(
                str(st.session_state.get("last_audit_run_id", "")),
                {
                    "go_ahead_system": "PASS" if system_go_ahead else "RED",
                    "go_ahead_effective": "PASS" if effective_go_ahead else "RED",
                    "go_ahead_override_applied": bool(manual_override_applied),
                    "go_ahead_override_reason": str(st.session_state.get("manual_go_ahead_reason", "") or ""),
                    "go_ahead_override_actor": str(st.session_state.get("manual_go_ahead_actor", "") or ""),
                    "last_run_crewai": now_iso_utc(),
                },
            )

            run_id_hist = str(st.session_state.get("last_audit_run_id", "") or "")
            if run_id_hist and effective_go_ahead and st.session_state.get("rag_archive_done_run_id") != run_id_hist:
                arch_text = (st.session_state.insights_sow_plaintext or "").strip()
                rb = st.session_state.redacted_buf
                if rb is not None:
                    try:
                        arch_text = read_docx_text(rb.getvalue()).strip()
                    except Exception:
                        pass
                if arch_text:
                    try:
                        apath = archive_approved_sow_text(
                            ROOT,
                            run_id=run_id_hist,
                            sow_name=st.session_state.sow_name or "sow",
                            text=arch_text,
                            consolidated_score=float(score_raw),
                            system_go_ahead=bool(system_go_ahead),
                            effective_go_ahead=bool(effective_go_ahead),
                            override_applied=bool(manual_override_applied),
                        )
                        if apath:
                            st.session_state.rag_archive_done_run_id = run_id_hist
                            update_run_history_entry(
                                run_id_hist,
                                {
                                    "approved_sow_archive_path": apath,
                                    "rag_corpus_fingerprint": str(
                                        st.session_state.get("rag_corpus_fingerprint") or ""
                                    ),
                                },
                            )
                            if st.session_state.get("_rag_archive_notice_path") != apath:
                                st.session_state._rag_archive_notice_path = apath
                                st.success(f"Approved SOW archived for RAG: `{apath}`")
                    except OSError as oe:
                        st.warning(f"Could not archive approved SOW: {oe}")

            with st.expander("Retrieval & citations (lexical RAG)", expanded=False):
                fp_disp = str(st.session_state.get("rag_corpus_fingerprint") or "")
                st.caption(
                    f"Corpus fingerprint: `{fp_disp}` — lexical scores are TF-IDF cosine similarity, not probabilities."
                )
                rmap = st.session_state.get("rag_retrieval_by_step") or {}
                if not rmap:
                    st.info("No retrieval index for this session yet — run Insights.")
                else:
                    for step, rows in rmap.items():
                        st.markdown(f"**{step}**")
                        if isinstance(rows, list) and rows:
                            st.dataframe(rows, use_container_width=True, hide_index=True)
                        else:
                            st.caption("No chunks retrieved for this query.")
                ev2 = a2.get("evidence_sources") if isinstance(a2.get("evidence_sources"), list) else []
                if ev2:
                    st.markdown("**Model-cited sources (Agent 2 — process circle)**")
                    st.dataframe(
                        [x for x in ev2 if isinstance(x, dict)][:20],
                        use_container_width=True,
                        hide_index=True,
                    )
                mavca_ev = (
                    mavca_for_score.get("evidence_sources")
                    if isinstance(mavca_for_score.get("evidence_sources"), list)
                    else []
                )
                if mavca_ev:
                    st.markdown("**Model-cited sources (MAVCA)**")
                    st.dataframe(
                        [x for x in mavca_ev if isinstance(x, dict)][:20],
                        use_container_width=True,
                        hide_index=True,
                    )

            report_payload = {
                "run_id": str(st.session_state.get("last_audit_run_id", "") or ""),
                "consolidated_score": score_raw,
                "system_go_ahead": system_go_ahead,
                "effective_go_ahead": effective_go_ahead,
                "manual_override_applied": manual_override_applied,
                "rag_corpus_fingerprint": str(st.session_state.get("rag_corpus_fingerprint") or ""),
                "rag_retrieval": st.session_state.get("rag_retrieval_by_step") or {},
                "agent2": a2,
                "mavca": mavca_for_score,
                "msa": msa_consistency,
                "po": po_consistency,
            }
            st.download_button(
                label="Download audit evidence bundle (JSON)",
                data=json.dumps(report_payload, indent=2, default=str),
                file_name=f"sow_audit_evidence_{st.session_state.get('last_audit_run_id', 'run')[:8]}.json",
                mime="application/json",
                key="download_audit_evidence_json",
            )

            st.divider()
            st.markdown("**Deep Dive (click to expand)**")
            with st.expander("Data quality precheck — input readiness", expanded=False):
                if not dq:
                    st.caption(
                        "Not yet evaluated — run Insights to assess SOW input quality."
                    )
                    st.info("Run Insights to evaluate SOW input quality.")
                else:
                    dq_detail_status = str(dq.get("status", "RED")).upper()
                    st.caption(
                        "Input quality is acceptable."
                        if dq_detail_status == "PASS"
                        else "Input quality needs improvement before reliable outputs."
                    )
                    st.write("Status:", _tl_emoji(dq_detail_status))
                    red_reasons = dq.get("red_reasons") if isinstance(dq.get("red_reasons"), list) else []
                    if red_reasons:
                        st.error("Why RED:\n- " + "\n- ".join(str(x) for x in red_reasons))
                    missing_sections = dq.get("missing_sections") if isinstance(dq.get("missing_sections"), list) else []
                    ambiguous = dq.get("ambiguous_clauses") if isinstance(dq.get("ambiguous_clauses"), list) else []
                    notes = dq.get("quality_notes") if isinstance(dq.get("quality_notes"), list) else []
                    recs = dq.get("recommendations") if isinstance(dq.get("recommendations"), list) else []
                    if missing_sections:
                        st.markdown("**Missing sections**")
                        st.write("\n".join(f"- {str(x)}" for x in missing_sections[:10]))
                    if ambiguous:
                        st.markdown("**Ambiguous excerpts**")
                        st.write("\n".join(f"- {str(x)}" for x in ambiguous[:10]))
                    if notes:
                        st.markdown("**Quality notes**")
                        st.write("\n".join(f"- {str(x)}" for x in notes[:10]))
                    if recs:
                        st.markdown("**Recommendations**")
                        st.write("\n".join(f"- {str(x)}" for x in recs[:10]))

            with st.expander("Pre-kickoff compliance gate", expanded=False):
                a3_status = str(a3.get("status", "RED")).upper()
                st.caption(
                    "Required governance signal is present."
                    if a3_status == "PASS"
                    else "Required pre-kickoff compliance signal is missing or unclear."
                )
                st.write("Status:", _tl_emoji(a3_status))
                req = str(a3.get("governance_requirement", "") or "").strip()
                if req:
                    st.markdown(f"- **Governance requirement:** {req}")
                p103 = (
                    policy_compliance.get("policy_103")
                    if isinstance(policy_compliance.get("policy_103"), dict)
                    else {}
                )
                ev = p103.get("evidence") if isinstance(p103.get("evidence"), list) else []
                if ev:
                    st.write("Evidence:\n" + "\n".join(f"- {str(x)}" for x in ev[:10]))
                recs = p103.get("recommendations") if isinstance(p103.get("recommendations"), list) else []
                if recs:
                    st.write("Recommendations:\n" + "\n".join(f"- {str(x)}" for x in recs[:10]))

            with st.expander("PII detection in SOW", expanded=False):
                a1_status = str(a1.get("status", "RED")).upper()
                pii_list = a1.get("pii_list") if isinstance(a1.get("pii_list"), list) else []
                st.caption(
                    "No PII risk found."
                    if a1_status == "PASS"
                    else "PII risk detected; remediation required."
                )
                st.write("Status:", _tl_emoji(a1_status))
                if pii_list:
                    st.markdown("**Detected excerpts**")
                    st.write("\n".join(f"- {str(x)}" for x in pii_list[:20]))
                    st.markdown("**Recommendations**")
                    st.write(
                        "- Run document sanitization to redact detected sensitive entities.\n"
                        "- Re-upload sanitized SOW and rerun Insights."
                    )
                else:
                    st.success("No PII excerpts detected in deterministic/LLM checks.")

                if a1_status == "RED":
                    if st.button("Sanitize Document & Redact PII", key="agent1_sanitize_btn"):
                        if st.session_state.sow_bytes is None:
                            st.error("Upload a SOW document first.")
                        else:
                            with st.spinner("PII remediation (data sanitizer + python-docx)..."):
                                try:
                                    mapping = run_remediation_crew([str(x) for x in pii_list])
                                    st.session_state.redaction_mapping = mapping
                                    buf = io.BytesIO(st.session_state.sow_bytes)
                                    st.session_state.redacted_buf = redact_docx_bytes(buf, mapping)
                                    st.success("Redaction complete. Download the sanitized document below.")
                                    notify_stakeholder_webhook(
                                        "DelEx Sentinel — SOW redacted",
                                        payload={
                                            "sow_name": st.session_state.sow_name,
                                            "redaction_keys": len(mapping),
                                        },
                                    )
                                except Exception as e:
                                    st.session_state.redacted_buf = None
                                    st.exception(e)

                if st.session_state.redacted_buf is not None:
                    base = (st.session_state.sow_name or "sow").replace(".docx", "")
                    st.download_button(
                        label="Download sanitized .docx",
                        data=st.session_state.redacted_buf.getvalue(),
                        file_name=f"{base}_sanitized.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    )
                    mapping = st.session_state.redaction_mapping or {}
                    if isinstance(mapping, dict) and mapping:
                        st.markdown("**Applied redaction mapping**")
                        mapping_rows = [{"Original": str(k), "Replacement": str(v)} for k, v in mapping.items()]
                        st.dataframe(mapping_rows, use_container_width=True, hide_index=True)

            with st.expander("Process circle & deliverables coverage", expanded=False):
                a2_status = str(a2.get("status", "RED")).upper()
                st.caption(
                    "Circle and deliverables are substantially aligned."
                    if a2_status == "PASS"
                    else "Coverage gaps found in required deliverables/guardrails."
                )
                st.write("Status:", _tl_emoji(a2_status))
                circle = str(a2.get("process_circle", "Not determined"))
                rationale = str(a2.get("classification_rationale", "No rationale provided."))
                st.markdown(f"- **Process circle:** {circle}")
                st.markdown(f"- **Rationale:** {rationale}")
                found = a2.get("deliverables_found") if isinstance(a2.get("deliverables_found"), list) else []
                missing = a2.get("deliverables_missing") if isinstance(a2.get("deliverables_missing"), list) else []
                if found:
                    st.markdown("**Coverage excerpts**")
                    st.write("\n".join(f"- {str(x)}" for x in found[:10]))
                if missing:
                    st.markdown("**Recommendations**")
                    st.write("\n".join(f"- Add explicit SOW language for: {str(x)}" for x in missing[:10]))
                elif a2_status == "PASS":
                    st.success("Required circle deliverables and guardrails are covered.")
                pc_rag = (st.session_state.get("rag_retrieval_by_step") or {}).get("process_circle") or []
                if isinstance(pc_rag, list) and pc_rag:
                    st.markdown("**Retrieved evidence (this run)**")
                    st.caption("Rule-based / lexical — not model-generated citations.")
                    st.dataframe(pc_rag, use_container_width=True, hide_index=True)
                evs2 = a2.get("evidence_sources") if isinstance(a2.get("evidence_sources"), list) else []
                if evs2:
                    st.markdown("**Model-cited evidence sources**")
                    st.dataframe([x for x in evs2 if isinstance(x, dict)][:15], use_container_width=True, hide_index=True)

            with st.expander("Executive summary", expanded=False):
                st.caption("Leadership-facing synthesis of the checks above.")
                summary = str(result.get("summary", "") or "")
                if summary:
                    st.write(summary)
                else:
                    st.info("Executive summary not generated for this run.")

            with st.expander("PII remediation mapping", expanded=False):
                mapping = st.session_state.redaction_mapping or {}
                if isinstance(mapping, dict) and mapping:
                    st.caption("Redaction mapping from the latest sanitize run (original text → placeholder).")
                    mapping_rows = [{"Original": str(k), "Replacement": str(v)} for k, v in mapping.items()]
                    st.dataframe(mapping_rows, use_container_width=True, hide_index=True)
                else:
                    st.caption("No mapping yet — sanitize from **PII detection in SOW** when PII is detected.")
                    st.info("No remediation mapping yet. Run sanitization from **PII detection in SOW** when PII is detected.")

            with st.expander("Task intelligence — tasks & MAVCA", expanded=False):
                mavca = st.session_state.mavca_result or {}
                if not mavca:
                    st.caption("Run Task Intelligence to generate decomposition and MAVCA classification.")
                    st.info("Run Task Intelligence to generate decomposition.")
                else:
                    t_status = str(mavca.get("status", "RED")).upper()
                    tasks = mavca.get("tasks") if isinstance(mavca.get("tasks"), list) else []
                    shifts = mavca.get("classification_shifts") if isinstance(mavca.get("classification_shifts"), list) else []
                    st.caption(
                        "Task decomposition and MAVCA classification generated."
                        if t_status == "PASS"
                        else "Task decomposition exists but needs quality review."
                    )
                    st.write("Status:", _tl_emoji(t_status))
                    st.markdown(f"- **Tasks generated:** {len(tasks)}")
                    st.markdown(f"- **Shift scenarios:** {len(shifts)}")
                    if tasks:
                        st.markdown("**All tasks with recommendations**")
                        task_rows: list[dict[str, Any]] = []
                        for t in tasks:
                            if not isinstance(t, dict):
                                continue
                            mavca_level = str(t.get("mavca_level", ""))
                            task_rows.append(
                                {
                                    "Task ID": str(t.get("task_id", "")),
                                    "Task name": str(t.get("task_name", "")),
                                    "MAVCA level": mavca_level,
                                    "SOW evidence": str(t.get("sow_evidence", "")),
                                    "Recommendation": _task_level_recommendation(mavca_level),
                                }
                            )
                        st.dataframe(task_rows, use_container_width=True, hide_index=True)
                    if shifts:
                        st.markdown("**Constraint shift recommendations**")
                        shift_rows: list[dict[str, Any]] = []
                        for s in shifts:
                            if not isinstance(s, dict):
                                continue
                            shift_rows.append(
                                {
                                    "Task ID": str(s.get("task_id", "")),
                                    "Assumption": str(s.get("constraint_assumption", "")),
                                    "Shifted MAVCA": str(s.get("new_mavca_level", "")),
                                    "Recommendation": str(s.get("why", "")),
                                }
                            )
                        st.dataframe(shift_rows, use_container_width=True, hide_index=True)
                    mv_rag = (st.session_state.get("rag_retrieval_by_step") or {}).get("mavca") or []
                    if isinstance(mv_rag, list) and mv_rag:
                        st.markdown("**Retrieved evidence (this run)**")
                        st.dataframe(mv_rag, use_container_width=True, hide_index=True)
                    mev = mavca.get("evidence_sources") if isinstance(mavca.get("evidence_sources"), list) else []
                    if mev:
                        st.markdown("**Model-cited evidence sources**")
                        st.dataframe([x for x in mev if isinstance(x, dict)][:15], use_container_width=True, hide_index=True)

            with st.expander("MSA consistency — contract alignment", expanded=False):
                if not msa_consistency:
                    st.caption("Not evaluated — no MSA input detected for this run.")
                    st.info("No MSA input detected for this run.")
                else:
                    msa_status = str(msa_consistency.get("status", "RED")).upper()
                    st.caption(
                        "No deterministic contradictions detected."
                        if msa_status == "PASS"
                        else "Potential SOW vs MSA contradiction(s) detected."
                    )
                    st.write("Status:", _tl_emoji(msa_status))
                    st.write(str(msa_consistency.get("summary", "")))
                    rev_msa = msa_consistency.get("retrieved_evidence")
                    if isinstance(rev_msa, list) and rev_msa:
                        st.markdown("**Retrieved MSA / legal context (lexical)**")
                        st.dataframe(rev_msa, use_container_width=True, hide_index=True)
                    conflicts = (
                        msa_consistency.get("conflicts")
                        if isinstance(msa_consistency.get("conflicts"), list)
                        else []
                    )
                    review_flags = (
                        msa_consistency.get("review_flags")
                        if isinstance(msa_consistency.get("review_flags"), list)
                        else []
                    )
                    if conflicts:
                        st.markdown("**Potential contradictions**")
                        for c in conflicts[:10]:
                            if isinstance(c, dict):
                                st.write(
                                    "- "
                                    + str(c.get("topic", "Contract topic"))
                                    + ": "
                                    + str(c.get("why", "Review required"))
                                )
                    if review_flags:
                        st.markdown("**Review flags**")
                        for rf in review_flags[:10]:
                            if isinstance(rf, dict):
                                st.write("- " + str(rf.get("flag", rf)))

            with st.expander("PO consistency — commercial alignment", expanded=False):
                if not po_consistency:
                    st.caption("Not evaluated — no PO input detected for this run.")
                    st.info("No PO input detected for this run.")
                else:
                    po_status = str(po_consistency.get("status", "RED")).upper()
                    st.caption(
                        "No deterministic PO/SOW mismatch detected."
                        if po_status == "PASS"
                        else "Potential PO/SOW mismatch detected; review advised."
                    )
                    st.write("Status:", _tl_emoji(po_status))
                    st.write(str(po_consistency.get("summary", "")))
                    rev_po = po_consistency.get("retrieved_evidence")
                    if isinstance(rev_po, list) and rev_po:
                        st.markdown("**Retrieved PO / commercial context (lexical)**")
                        st.dataframe(rev_po, use_container_width=True, hide_index=True)
                    cs = (
                        po_consistency.get("commercial_summary")
                        if isinstance(po_consistency.get("commercial_summary"), dict)
                        else {}
                    )
                    po_amt = cs.get("po_total")
                    sow_amt = cs.get("sow_estimated_contract_value")
                    var_abs = cs.get("variance_abs")
                    var_pct = cs.get("variance_pct")
                    within = cs.get("within_tolerance")
                    tol_thr = cs.get("tolerance_threshold_abs")
                    po_net = cs.get("po_payment_net_days")
                    sow_net = cs.get("sow_payment_net_days")
                    pay_align = cs.get("payment_terms_aligned")

                    st.markdown("**Commercial comparison (values parsed from uploaded SOW and PO)**")
                    if po_amt is None and sow_amt is None:
                        st.caption("Could not extract PO total and/or SOW estimated contract value from the documents.")
                    else:
                        comp_rows: list[dict[str, Any]] = []
                        if sow_amt is not None:
                            comp_rows.append(
                                {
                                    "Source": "SOW (uploaded)",
                                    "Field": "Estimated Contract Value",
                                    "Amount (USD)": f"{float(sow_amt):,.2f}",
                                }
                            )
                        else:
                            comp_rows.append(
                                {
                                    "Source": "SOW (uploaded)",
                                    "Field": "Estimated Contract Value",
                                    "Amount (USD)": "Not detected — use `Estimated Contract Value: $X` in SOW",
                                }
                            )
                        if po_amt is not None:
                            comp_rows.append(
                                {
                                    "Source": "PO (uploaded)",
                                    "Field": "Total",
                                    "Amount (USD)": f"{float(po_amt):,.2f}",
                                }
                            )
                        else:
                            comp_rows.append(
                                {
                                    "Source": "PO (uploaded)",
                                    "Field": "Total",
                                    "Amount (USD)": "Not detected — ensure a `Total` line with amount",
                                }
                            )
                        if var_abs is not None and sow_amt is not None:
                            comp_rows.append(
                                {
                                    "Source": "Variance",
                                    "Field": "PO minus SOW",
                                    "Amount (USD)": f"{float(var_abs):+,.2f} ({float(var_pct or 0):+.2f}%)",
                                }
                            )
                        st.dataframe(comp_rows, use_container_width=True, hide_index=True)
                        if isinstance(within, bool):
                            status_note = (
                                "Within tolerance for automated check."
                                if within
                                else "Outside tolerance — see mismatch details below."
                            )
                            st.caption(
                                f"{status_note} Tolerance: {cs.get('tolerance_basis', '')}"
                                + (
                                    f" (threshold ${float(tol_thr):,.2f})."
                                    if tol_thr is not None
                                    else "."
                                )
                            )

                    st.markdown("**Payment terms (Net days)**")
                    pt_rows = [
                        {
                            "Document": "SOW (uploaded)",
                            "Net days": str(sow_net) if sow_net is not None else "Not detected",
                        },
                        {
                            "Document": "PO (uploaded)",
                            "Net days": str(po_net) if po_net is not None else "Not detected",
                        },
                    ]
                    st.dataframe(pt_rows, use_container_width=True, hide_index=True)
                    if pay_align is True:
                        st.success("Payment terms (Net) match between SOW and PO.")
                    elif pay_align is False:
                        st.warning("Payment terms (Net) differ or missing on one side — see conflicts below.")
                    else:
                        st.caption("Net payment terms could not be compared (missing on one or both documents).")
                    conflicts = po_consistency.get("conflicts") if isinstance(po_consistency.get("conflicts"), list) else []
                    review_flags = (
                        po_consistency.get("review_flags") if isinstance(po_consistency.get("review_flags"), list) else []
                    )
                    recs = po_consistency.get("recommendations") if isinstance(po_consistency.get("recommendations"), list) else []
                    if conflicts:
                        st.markdown("**Potential mismatches**")
                        for c in conflicts[:10]:
                            if isinstance(c, dict):
                                st.write(
                                    "- "
                                    + str(c.get("topic", "Commercial topic"))
                                    + " ("
                                    + str(c.get("severity", "MEDIUM"))
                                    + "): "
                                    + str(c.get("why", "Review required"))
                                )
                                pe = str(c.get("po_evidence") or "").strip()
                                se = str(c.get("sow_evidence") or "").strip()
                                if pe or se:
                                    st.caption(
                                        ("PO excerpt: " + pe if pe else "")
                                        + ("  |  " if pe and se else "")
                                        + ("SOW excerpt: " + se if se else "")
                                    )
                    if review_flags:
                        st.markdown("**Review flags**")
                        for rf in review_flags[:10]:
                            if isinstance(rf, dict):
                                st.write("- " + str(rf.get("note", rf)))
                    if recs:
                        st.markdown("**Recommendations**")
                        st.write("\n".join(f"- {str(x)}" for x in recs[:10]))

            with st.expander("Policy compliance — 101 / 102 / 103 / 104 (derived summary)", expanded=False):
                if not policy_compliance:
                    st.caption("Derived policy summary unavailable for this run.")
                    st.info("Policy compliance details unavailable for this run.")
                else:
                    overall_policy = str(policy_compliance.get("overall_status", "RED")).upper()
                    st.caption(
                        "All policy controls are aligned."
                        if overall_policy == "PASS"
                        else "One or more policy controls require remediation."
                    )
                    st.write("Overall status:", _tl_emoji(overall_policy))
                    for pkey, title in (
                        ("policy_101", "Policy 101 — PII Hygiene"),
                        ("policy_102", "Policy 102 — Delivery Governance"),
                        ("policy_103", "Policy 103 — Compliance Gate"),
                        ("policy_104", "Policy 104 — Ethics & Secure Workplace Access"),
                    ):
                        pol = policy_compliance.get(pkey) if isinstance(policy_compliance.get(pkey), dict) else {}
                        if not pol:
                            continue
                        p_status = str(pol.get("status", "RED")).upper()
                        violations = pol.get("violations") if isinstance(pol.get("violations"), list) else []
                        evidence = pol.get("evidence") if isinstance(pol.get("evidence"), list) else []
                        recs = pol.get("recommendations") if isinstance(pol.get("recommendations"), list) else []
                        st.markdown(f"**{title}: {_tl_emoji(p_status)}**")
                        if pkey == "policy_103":
                            req = str(a3.get("governance_requirement", "") or "").strip()
                            if req:
                                st.markdown(f"- **Governance requirement:** {req}")
                        if evidence:
                            st.write("Evidence:\n" + "\n".join(f"- {str(x)}" for x in evidence[:12]))
                        if violations:
                            st.write("Violations:\n" + "\n".join(f"- {str(x)}" for x in violations[:10]))
                        if pkey == "policy_104":
                            checks = pol.get("checks") if isinstance(pol.get("checks"), list) else []
                            if checks:
                                check_rows: list[dict[str, str]] = []
                                for c in checks:
                                    if not isinstance(c, dict):
                                        continue
                                    met = bool(c.get("met"))
                                    check_rows.append(
                                        {
                                            "Control": str(c.get("theme", "")),
                                            "Status": _tl_emoji("PASS" if met else "RED"),
                                            "Matched text": str(c.get("matched_text", "")) if met else "-",
                                        }
                                    )
                                if check_rows:
                                    st.dataframe(check_rows, use_container_width=True, hide_index=True)
                        if recs:
                            st.write("Recommendations:\n" + "\n".join(f"- {str(x)}" for x in recs[:10]))

            summary = result.get("summary")
            if summary:
                st.info(str(summary))

            if score > 0 and not effective_go_ahead:
                if manual_override_applied and float(score_raw) <= 83.0:
                    st.warning(
                        "Override is selected, but the consolidated score must be **greater than 83%** "
                        "for Executive GO-Ahead PASS when Data Quality is the only blocking control."
                    )
                else:
                    st.warning(
                        "Score reflects partial passes across weighted controls. "
                        "Executive GO-Ahead requires every scored control to be PASS, "
                        "or an eligible Data Quality override with score > 83%."
                    )
            elif manual_override_applied and effective_go_ahead:
                st.info(
                    "Executive GO-Ahead is PASS via manual override (Data Quality RED; consolidated score > 83%). "
                    "Follow up on data quality; approved SOW text is archived for future RAG when archiving succeeded."
                )
            with st.expander("Explain score"):
                weighted_rows = [
                    r
                    for r in score_rows
                    if float(r.get("Weight", 0) or 0) > 0
                ]
                display_rows: list[dict[str, Any]] = []
                for r in weighted_rows:
                    display_rows.append(
                        {
                            "Component": _component_display_name(str(r.get("Component", ""))),
                            "Weight": float(r.get("Weight", 0) or 0),
                            "Status": str(r.get("Status", "RED")),
                            "Score contribution": float(r.get("Score contribution", 0) or 0),
                        }
                    )
                if display_rows:
                    st.dataframe(display_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No weighted controls available yet. Run Insights Audit first.")
                st.caption(
                    "Scoring includes PII detection, process circle & gate, task intelligence, data quality precheck, and MSA consistency only. "
                    "Executive summary is informational; PII remediation is a separate action; consolidated policy view is derived and non-scoring. "
                    "PO consistency is advisory in this release and does not affect score or go-ahead."
                )
                st.caption(
                    f"System go-ahead: {_tl_emoji('PASS' if system_go_ahead else 'RED')}  |  "
                    f"Effective go-ahead: {_tl_emoji('PASS' if effective_go_ahead else 'RED')}"
                    + (
                        f"  |  Override by: {str(st.session_state.get('manual_go_ahead_actor', '') or 'N/A')}"
                        if manual_override_applied
                        else ""
                    )
                )
        else:
            st.info("Run an audit to view scorecard insights.")

    with tab_leaderboard:
        if not insights_done:
            st.warning("Leaderboard is locked. Complete **Insights Audit** first.")
            st.info("Upload SOW and run **1) Run Insights Audit** from the left sidebar.")
        else:
            st.subheader("SOW Leaderboard")
            with st.expander("Leaderboard maintenance"):
                st.caption("Clears locally stored CrewAI run history used by this leaderboard.")
                if st.button("Reset leaderboard history", type="secondary"):
                    reset_run_history()
                    st.success("Leaderboard history cleared.")
                    st.rerun()
            history = load_run_history()
            rows = build_leaderboard_rows(history)
            if not rows:
                st.info("No CrewAI runs recorded yet.")
            else:
                st.dataframe(rows, use_container_width=True, hide_index=True)
                csv_text = leaderboard_to_csv(rows)
                st.download_button(
                    label="Export leaderboard (CSV)",
                    data=csv_text,
                    file_name="sow_leaderboard.csv",
                    mime="text/csv",
                )

    st.divider()
    st.caption("Session state retains audit results until you upload a new file or clear the app cache.")


if __name__ == "__main__":
    main()
