"""Build a downloadable PDF audit report from Insights session payload."""

from __future__ import annotations

import re
from typing import Any

from fpdf import FPDF


def _safe_text(value: Any, max_len: int = 2000) -> str:
    """Strip emoji and non-latin-1 chars for core PDF fonts."""
    s = str(value or "").strip()
    s = re.sub(r"[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U00002600-\U000026FF]", "", s)
    s = s.replace("\u2014", "-").replace("\u2013", "-").replace("\u2018", "'").replace("\u2019", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.encode("latin-1", errors="replace").decode("latin-1")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _status_label(value: Any) -> str:
    raw = _safe_text(value, 32).upper()
    if raw in ("PASS", "RED", "CLEAR", "BLOCKED"):
        return raw
    if "PASS" in raw:
        return "PASS"
    if "RED" in raw:
        return "RED"
    return raw or "N/A"


class _AuditPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 8, "SOW Risk & Readiness - Audit Report", align="L", ln=True)
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}", align="C")


def _ensure_page(pdf: _AuditPDF, needed: float = 20) -> None:
    if pdf.get_y() + needed > pdf.h - pdf.b_margin:
        pdf.add_page()


def _text_width(pdf: FPDF) -> float:
    return pdf.epw


def _section_title(pdf: _AuditPDF, title: str) -> None:
    _ensure_page(pdf, 14)
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.multi_cell(_text_width(pdf), 7, _safe_text(title, 200))
    pdf.ln(2)


def _body(pdf: _AuditPDF, text: str) -> None:
    pdf.set_font("Helvetica", "", 10)
    w = _text_width(pdf)
    for para in _safe_text(text, 8000).split("\n"):
        if not para.strip():
            pdf.ln(3)
            continue
        _ensure_page(pdf, 8)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 5, para)
    pdf.ln(2)


def _bullets(pdf: _AuditPDF, items: list[Any], cap: int = 15) -> None:
    pdf.set_font("Helvetica", "", 10)
    w = _text_width(pdf)
    for item in items[:cap]:
        line = _safe_text(item, 500)
        if not line:
            continue
        _ensure_page(pdf, 8)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(w, 5, f"- {line}")
    if len(items) > cap:
        _body(pdf, f"... and {len(items) - cap} more item(s).")
    pdf.ln(1)


def _kv_table(pdf: _AuditPDF, rows: list[tuple[str, str]]) -> None:
    pdf.set_font("Helvetica", "", 10)
    w = _text_width(pdf)
    for k, v in rows:
        _ensure_page(pdf, 10)
        pdf.set_x(pdf.l_margin)
        label = _safe_text(k, 80)
        val = _safe_text(v, 600)
        line = f"{label}: {val}" if val else f"{label}:"
        pdf.multi_cell(w, 6, line)
    pdf.ln(2)


def build_audit_report_pdf_bytes(payload: dict[str, Any]) -> bytes:
    """Render full audit payload to PDF bytes."""
    pdf = _AuditPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=14)
    pdf.add_page()

    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    _section_title(pdf, "Executive summary")
    _kv_table(
        pdf,
        [
            ("Project", meta.get("project_name", "")),
            ("SOW file", meta.get("sow_name_file", "")),
            ("Run ID", meta.get("run_id", "")),
            ("Generated (UTC)", meta.get("generated_at", "")),
            ("Consolidated score", f"{meta.get('consolidated_score', '')}%"),
            ("Data quality", _status_label(meta.get("dq_status"))),
            ("System GO-Ahead", _status_label(meta.get("system_go_ahead"))),
            ("Effective GO-Ahead", _status_label(meta.get("effective_go_ahead"))),
        ],
    )
    if meta.get("manual_override_applied"):
        _kv_table(
            pdf,
            [
                ("Override applied", "Yes"),
                ("Approver", meta.get("override_actor", "")),
                ("Rationale", meta.get("override_reason", "")),
            ],
        )
    if meta.get("approved_sow_archive_path"):
        _body(pdf, f"Approved SOW archive: {meta.get('approved_sow_archive_path')}")

    summary = _safe_text(payload.get("executive_summary", ""), 4000)
    if summary:
        _section_title(pdf, "Leadership narrative")
        _body(pdf, summary)

    score_rows = payload.get("score_rows") if isinstance(payload.get("score_rows"), list) else []
    if score_rows:
        _section_title(pdf, "Score breakdown")
        pdf.set_font("Helvetica", "B", 9)
        col_w = (90, 22, 28, 30)
        headers = ("Component", "Status", "Weight", "Contribution")
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 6, h, border=1)
        pdf.ln()
        pdf.set_font("Helvetica", "", 9)
        for row in score_rows:
            if not isinstance(row, dict):
                continue
            _ensure_page(pdf, 8)
            comp = _safe_text(row.get("Component", ""), 120)
            status = _status_label(row.get("Status"))
            weight = row.get("Weight", 0)
            contrib = row.get("Score contribution", 0)
            pdf.cell(col_w[0], 6, comp[:60], border=1)
            pdf.cell(col_w[1], 6, status, border=1)
            pdf.cell(col_w[2], 6, f"{float(weight):.1f}", border=1)
            pdf.cell(col_w[3], 6, f"{float(contrib):.1f}", border=1)
            pdf.ln()
        pdf.ln(2)

    dq = payload.get("data_quality") if isinstance(payload.get("data_quality"), dict) else {}
    if dq:
        _section_title(pdf, "Data quality precheck")
        _kv_table(pdf, [("Status", _status_label(dq.get("status")))])
        for label, key in (
            ("Red reasons", "red_reasons"),
            ("Missing sections", "missing_sections"),
            ("Ambiguous clauses", "ambiguous_clauses"),
            ("Recommendations", "recommendations"),
        ):
            items = dq.get(key) if isinstance(dq.get(key), list) else []
            if items:
                _body(pdf, label + ":")
                _bullets(pdf, items, cap=10)

    a1 = payload.get("agent1") if isinstance(payload.get("agent1"), dict) else {}
    if a1:
        _section_title(pdf, "PII detection (Agent 1)")
        _kv_table(pdf, [("Status", _status_label(a1.get("status")))])
        pii = a1.get("pii_list") if isinstance(a1.get("pii_list"), list) else []
        if pii:
            _body(pdf, "Detected strings:")
            _bullets(pdf, pii, cap=20)
        else:
            _body(pdf, "No PII strings reported.")

    a2 = payload.get("agent2") if isinstance(payload.get("agent2"), dict) else {}
    if a2:
        _section_title(pdf, "Process circle and deliverables (Agent 2)")
        _kv_table(
            pdf,
            [
                ("Status", _status_label(a2.get("status"))),
                ("Process circle", a2.get("process_circle", "")),
                ("Rationale", a2.get("classification_rationale", "")),
            ],
        )
        for label, key in ("Coverage", "deliverables_found"), ("Gaps", "deliverables_missing"):
            items = a2.get(key[1]) if isinstance(a2.get(key[1]), list) else []
            if items:
                _body(pdf, f"{label}:")
                _bullets(pdf, items, cap=10)
        evs = a2.get("evidence_sources") if isinstance(a2.get("evidence_sources"), list) else []
        if evs:
            _body(pdf, "Model-cited evidence:")
            for ev in evs[:8]:
                if isinstance(ev, dict):
                    _bullets(
                        pdf,
                        [
                            f"{ev.get('source', '')} [{ev.get('chunk_id', '')}]: "
                            f"{ev.get('excerpt', '')} - {ev.get('reason', '')}"
                        ],
                        cap=1,
                    )

    a3 = payload.get("agent3") if isinstance(payload.get("agent3"), dict) else {}
    if a3:
        _section_title(pdf, "Pre-kickoff compliance gate (Agent 3)")
        _kv_table(
            pdf,
            [
                ("Status", _status_label(a3.get("status"))),
                ("Requirement", a3.get("governance_requirement", "")),
            ],
        )
        ev = a3.get("governance_signal_evidence") if isinstance(a3.get("governance_signal_evidence"), list) else []
        if ev:
            _body(pdf, "Evidence:")
            _bullets(pdf, ev, cap=10)

    msa = payload.get("msa") if isinstance(payload.get("msa"), dict) else {}
    if msa:
        _section_title(pdf, "MSA vs SOW consistency")
        _kv_table(pdf, [("Status", _status_label(msa.get("status")))])
        _body(pdf, _safe_text(msa.get("summary", ""), 1500))
        conflicts = msa.get("conflicts") if isinstance(msa.get("conflicts"), list) else []
        if conflicts:
            _body(pdf, "Conflicts:")
            for c in conflicts[:8]:
                if isinstance(c, dict):
                    _bullets(pdf, [f"{c.get('topic', '')}: {c.get('why', '')}"], cap=1)

    po = payload.get("po") if isinstance(payload.get("po"), dict) else {}
    if po:
        _section_title(pdf, "PO vs SOW consistency (advisory)")
        _kv_table(pdf, [("Status", _status_label(po.get("status")))])
        _body(pdf, _safe_text(po.get("summary", ""), 1500))
        conflicts = po.get("conflicts") if isinstance(po.get("conflicts"), list) else []
        if conflicts:
            _body(pdf, "Conflicts:")
            for c in conflicts[:8]:
                if isinstance(c, dict):
                    _bullets(pdf, [f"{c.get('topic', '')}: {c.get('why', '')}"], cap=1)

    mavca = payload.get("mavca") if isinstance(payload.get("mavca"), dict) else {}
    if mavca:
        _section_title(pdf, "Task intelligence (MAVCA)")
        tasks = mavca.get("tasks") if isinstance(mavca.get("tasks"), list) else []
        shifts = mavca.get("classification_shifts") if isinstance(mavca.get("classification_shifts"), list) else []
        _kv_table(
            pdf,
            [
                ("Status", _status_label(mavca.get("status"))),
                ("Tasks", str(len(tasks))),
                ("Classification shifts", str(len(shifts))),
            ],
        )
        if _safe_text(mavca.get("summary", ""), 100):
            _body(pdf, _safe_text(mavca.get("summary", ""), 2000))
        for t in tasks[:12]:
            if isinstance(t, dict):
                _bullets(
                    pdf,
                    [
                        f"{t.get('task_id', '')} {t.get('task_name', '')} - "
                        f"{t.get('mavca_level', '')}"
                    ],
                    cap=1,
                )

    policy = payload.get("policy_compliance") if isinstance(payload.get("policy_compliance"), dict) else {}
    if policy:
        _section_title(pdf, "Policy compliance (derived)")
        _kv_table(pdf, [("Overall", _status_label(policy.get("overall_status")))])
        for pkey, title in (
            ("policy_101", "Policy 101 - PII"),
            ("policy_102", "Policy 102 - Delivery"),
            ("policy_103", "Policy 103 - Compliance gate"),
            ("policy_104", "Policy 104 - Ethics"),
        ):
            pol = policy.get(pkey) if isinstance(policy.get(pkey), dict) else {}
            if pol:
                _body(pdf, f"{title}: {_status_label(pol.get('status'))}")
                viol = pol.get("violations") if isinstance(pol.get("violations"), list) else []
                if viol:
                    _bullets(pdf, viol, cap=5)

    rag = payload.get("rag_retrieval") if isinstance(payload.get("rag_retrieval"), dict) else {}
    fp = _safe_text(payload.get("rag_corpus_fingerprint", ""), 64)
    if rag or fp:
        _section_title(pdf, "Retrieval appendix (lexical RAG)")
        if fp:
            _body(pdf, f"Corpus fingerprint: {fp}")
        for step, rows in rag.items():
            if not isinstance(rows, list) or not rows:
                continue
            _body(pdf, f"Step: {step}")
            for row in rows[:5]:
                if isinstance(row, dict):
                    _bullets(
                        pdf,
                        [
                            f"{row.get('source', '')} [{row.get('chunk_id', '')}] "
                            f"score={row.get('lexical_score', '')}: "
                            f"{row.get('excerpt', '')}"
                        ],
                        cap=1,
                    )

    _section_title(pdf, "Disclaimer")
    _body(
        pdf,
        "This report is generated by the SOW Risk & Readiness Command Center for decision support. "
        "It does not replace legal, commercial, or executive sign-off. Verify all findings against source documents.",
    )

    return bytes(pdf.output())
