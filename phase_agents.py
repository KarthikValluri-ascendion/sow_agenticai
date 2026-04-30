"""Phase-specific CrewAI agents and deterministic gates."""

from __future__ import annotations

import json
import re
from typing import Any, Callable

from crewai import Agent, Crew, Process, Task

from env_config import get_groq_api_key, get_groq_model, get_openai_base_url
from groq_llm import GroqOpenAICompatLLM
from json_utils import extract_json_object

VALID_MAVCA_LEVELS = {"MANUAL", "AUGMENTED", "VALIDATED", "CURATED", "AUTONOMOUS"}


def _make_llm() -> GroqOpenAICompatLLM:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY or test3 in `.env` before running MAVCA flow.")
    base = (get_openai_base_url() or "").strip() or None
    return GroqOpenAICompatLLM(
        model=get_groq_model(),
        api_key=api_key,
        temperature=0,
        base_url=base,
    )


def _truncate(text: str, max_chars: int = 120_000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for model context ...]"


def _parse_task_output(task: Any) -> dict[str, Any]:
    out = getattr(task, "output", None)
    if out is None:
        return {}
    raw = None
    for attr in ("raw", "raw_output", "result"):
        if hasattr(out, attr):
            raw = getattr(out, attr)
            if raw is not None:
                break
    if raw is None:
        raw = str(out)
    return extract_json_object(str(raw))


def _kick_single(agent: Agent, task: Task) -> dict[str, Any]:
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    crew.kickoff()
    tasks = getattr(crew, "tasks", None) or [task]
    try:
        return _parse_task_output(tasks[0])
    except Exception as exc:
        return {"status": "RED", "error": f"Could not parse model JSON output: {exc}"}


def enforce_mavca_schema(raw_output: dict[str, Any]) -> dict[str, Any]:
    """Normalize model output to expected schema with schema validation notes."""
    out = raw_output if isinstance(raw_output, dict) else {}
    schema_errors: list[str] = []

    tasks_in = out.get("tasks")
    tasks: list[dict[str, str]] = []
    if not isinstance(tasks_in, list):
        schema_errors.append("tasks must be a list.")
        tasks_in = []
    for i, task in enumerate(tasks_in):
        if not isinstance(task, dict):
            schema_errors.append(f"tasks[{i}] must be an object.")
            continue
        tasks.append(
            {
                "task_id": str(task.get("task_id", "")).strip(),
                "task_name": str(task.get("task_name", "")).strip(),
                "description": str(task.get("description", "")).strip(),
                "mavca_level": str(task.get("mavca_level", "")).strip(),
                "rationale": str(task.get("rationale", "")).strip(),
                "sow_evidence": str(task.get("sow_evidence", "")).strip(),
            }
        )

    shifts_in = out.get("classification_shifts")
    shifts: list[dict[str, str]] = []
    if not isinstance(shifts_in, list):
        schema_errors.append("classification_shifts must be a list.")
        shifts_in = []
    for i, shift in enumerate(shifts_in):
        if not isinstance(shift, dict):
            schema_errors.append(f"classification_shifts[{i}] must be an object.")
            continue
        shifts.append(
            {
                "task_id": str(shift.get("task_id", "")).strip(),
                "constraint_assumption": str(shift.get("constraint_assumption", "")).strip(),
                "new_mavca_level": str(shift.get("new_mavca_level", "")).strip(),
                "why": str(shift.get("why", "")).strip(),
            }
        )

    normalized = {
        "tasks": tasks,
        "classification_shifts": shifts,
        "summary": str(out.get("summary", "")).strip(),
        "status": str(out.get("status", "")).strip().upper() or "PASS",
        "schema_status": "PASS" if not schema_errors else "RED",
        "schema_errors": schema_errors,
    }
    return normalized


def validate_phase1b_mavca(mavca_output: dict[str, Any]) -> dict[str, Any]:
    """Deterministic gate for Deliverable 1B with explainable rule breakdown."""
    tasks = mavca_output.get("tasks") if isinstance(mavca_output.get("tasks"), list) else []
    shifts = (
        mavca_output.get("classification_shifts")
        if isinstance(mavca_output.get("classification_shifts"), list)
        else []
    )

    missing: list[str] = []
    rules: list[dict[str, Any]] = []
    schema_errors = (
        mavca_output.get("schema_errors")
        if isinstance(mavca_output.get("schema_errors"), list)
        else []
    )
    if schema_errors:
        missing.extend([f"Schema error: {str(err)}" for err in schema_errors])
        rules.append({"rule": "schema_validation", "passed": False, "details": "Output violated schema"})
    else:
        rules.append({"rule": "schema_validation", "passed": True, "details": "Output matches expected schema"})

    count_ok = 8 <= len(tasks) <= 10
    if not count_ok:
        missing.append("Task inventory must contain 8-10 tasks.")
    rules.append(
        {
            "rule": "task_count_8_to_10",
            "passed": count_ok,
            "details": f"tasks_count={len(tasks)}",
        }
    )

    task_ids: set[str] = set()
    task_quality_ok = True
    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            missing.append(f"Task {i + 1} must be an object.")
            task_quality_ok = False
            continue
        task_id = str(task.get("task_id", "")).strip()
        task_name = str(task.get("task_name", "")).strip()
        mavca_level = str(task.get("mavca_level", "")).strip().upper()
        rationale = str(task.get("rationale", "")).strip()
        evidence = str(task.get("sow_evidence", "")).strip()
        if not task_id:
            missing.append(f"Task {i + 1} missing task_id.")
            task_quality_ok = False
        if not task_name:
            missing.append(f"Task {i + 1} missing task_name.")
            task_quality_ok = False
        if mavca_level not in VALID_MAVCA_LEVELS:
            missing.append(f"Task {i + 1} has invalid MAVCA level.")
            task_quality_ok = False
        if len(rationale.split()) < 18:
            missing.append(f"Task {i + 1} rationale too short; provide 2-3 sentences.")
            task_quality_ok = False
        if len(evidence) < 20:
            missing.append(f"Task {i + 1} should include explicit SOW evidence.")
            task_quality_ok = False
        if task_id:
            task_ids.add(task_id)
    rules.append(
        {
            "rule": "task_field_quality",
            "passed": task_quality_ok,
            "details": "Each task includes IDs, valid MAVCA level, rationale, and evidence.",
        }
    )

    shifts_ok = len(shifts) >= 2
    if not shifts_ok:
        missing.append("At least 2 classification-shift scenarios are required.")
    rules.append(
        {
            "rule": "classification_shift_count_min_2",
            "passed": shifts_ok,
            "details": f"classification_shifts_count={len(shifts)}",
        }
    )

    shift_quality_ok = True
    for i, shift in enumerate(shifts):
        if not isinstance(shift, dict):
            missing.append(f"Shift scenario {i + 1} must be an object.")
            shift_quality_ok = False
            continue
        shift_task_id = str(shift.get("task_id", "")).strip()
        new_level = str(shift.get("new_mavca_level", "")).strip().upper()
        reason = str(shift.get("why", "")).strip()
        if not shift_task_id or shift_task_id not in task_ids:
            missing.append(f"Shift scenario {i + 1} must reference an existing task_id.")
            shift_quality_ok = False
        if new_level not in VALID_MAVCA_LEVELS:
            missing.append(f"Shift scenario {i + 1} has invalid new_mavca_level.")
            shift_quality_ok = False
        if len(reason.split()) < 12:
            missing.append(f"Shift scenario {i + 1} needs clearer justification.")
            shift_quality_ok = False
    rules.append(
        {
            "rule": "classification_shift_quality",
            "passed": shift_quality_ok,
            "details": "Shift scenarios reference valid tasks/levels with clear rationale.",
        }
    )

    penalty = min(100, 8 * len(missing))
    score = max(0, 100 - penalty)
    return {
        "gate_name": "task_intelligence_gate",
        "status": "PASS" if not missing else "RED",
        "score": float(score),
        "missing_criteria": missing,
        "evidence": [
            f"tasks_count={len(tasks)}",
            f"classification_shifts_count={len(shifts)}",
        ],
        "rules": rules,
    }


def run_mavca_decomposition(
    sow_text: str,
    kb_text: str | None = None,
    ethics_text: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Generate Deliverable 1B (MAVCA decomposition) from SOW + references."""
    sow = _truncate(sow_text)
    kb = _truncate(kb_text or "", max_chars=40_000)
    ethics = _truncate(ethics_text or "", max_chars=80_000)

    llm = _make_llm()
    if progress:
        progress("Task intelligence agent: extracting workflows and assigning MAVCA levels...")

    agent = Agent(
        role="MAVCA Decomposition Specialist",
        goal=(
            "From a project SOW, produce a high-quality task decomposition and classify each task "
            "as Manual, Augmented, Validated, Curated, or Autonomous."
        ),
        backstory=(
            "You are a delivery architect applying Agent-First Thinking and MAVCA rigor. "
            "You ground classifications in explicit SOW evidence and realistic constraint scenarios."
        ),
        llm=llm,
        verbose=True,
    )

    ethics_section = ""
    if ethics.strip():
        ethics_section = (
            "\nETHICS MANUAL (reference only; use to shape realistic constraints):\n"
            f'"""\n{ethics}\n"""\n'
        )

    task = Task(
        description=(
            "Create a business-ready task decomposition and MAVCA classification.\n"
            "Requirements:\n"
            "1) Extract 8-10 concrete tasks/workflows from the SOW.\n"
            "2) For each task, include task_id, task_name, description, mavca_level, rationale, sow_evidence.\n"
            "3) rationale must be 2-3 sentences minimum and practical.\n"
            "4) Add at least 2 classification_shifts where a constraint changes MAVCA level.\n"
            "5) summary should be a short executive paragraph.\n\n"
            f'KNOWLEDGE BASE:\n"""\n{kb}\n"""\n'
            f"{ethics_section}\n"
            f'SOW TEXT:\n"""\n{sow}\n"""\n\n'
            "Return JSON ONLY:\n"
            "{\n"
            '  "tasks": [\n'
            "    {\n"
            '      "task_id": "T1",\n'
            '      "task_name": "...",\n'
            '      "description": "...",\n'
            '      "mavca_level": "Manual|Augmented|Validated|Curated|Autonomous",\n'
            '      "rationale": "...",\n'
            '      "sow_evidence": "..."\n'
            "    }\n"
            "  ],\n"
            '  "classification_shifts": [\n'
            "    {\n"
            '      "task_id": "T5",\n'
            '      "constraint_assumption": "...",\n'
            '      "new_mavca_level": "Manual|Augmented|Validated|Curated|Autonomous",\n'
            '      "why": "..."\n'
            "    }\n"
            "  ],\n"
            '  "summary": "...",\n'
            '  "status": "PASS"|"RED"\n'
            "}\n"
        ),
        expected_output="Strict JSON with tasks, classification_shifts, summary, status.",
        agent=agent,
    )

    out = _kick_single(agent, task)
    normalized = enforce_mavca_schema(out if isinstance(out, dict) else {})
    if "status" not in normalized:
        normalized["status"] = "RED" if normalized.get("error") else "PASS"
    if normalized.get("schema_status") == "RED":
        normalized["status"] = "RED"
    return normalized


def _extract_ambiguous_clauses_deterministic(sow_text: str) -> list[str]:
    patterns = [
        r"\bTBD\b",
        r"\bto be decided\b",
        r"\bas needed\b",
        r"\bwhere applicable\b",
        r"\bbest effort\b",
        r"\betc\.\b",
        r"\bsubject to change\b",
    ]
    hits: list[str] = []
    for p in patterns:
        for m in re.finditer(p, sow_text, flags=re.IGNORECASE):
            start = max(0, m.start() - 60)
            end = min(len(sow_text), m.end() + 60)
            snippet = sow_text[start:end].replace("\n", " ").strip()
            if snippet and snippet not in hits:
                hits.append(snippet)
    return hits[:10]


def _check_required_sections(sow_text: str) -> dict[str, bool]:
    checks = {
        "timeline_term": bool(re.search(r"(?i)\bTimeline\s*/\s*Term\b|\bTimeline\b", sow_text)),
        "scope": bool(re.search(r"(?i)\bScope of Services\b|\bScope\b", sow_text)),
        "deliverables": bool(re.search(r"(?i)\bDeliverables?\b", sow_text)),
        "roles": bool(re.search(r"(?i)\bProject Management\b|\bpersonnel\b|\bRole\b", sow_text)),
    }
    return checks


def run_data_quality_precheck(
    sow_text: str,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Pre-check SOW quality before phase execution."""
    sow = _truncate(sow_text, max_chars=80_000)
    section_checks = _check_required_sections(sow)
    missing_sections = [k for k, v in section_checks.items() if not v]
    deterministic_ambiguous = _extract_ambiguous_clauses_deterministic(sow)

    llm = _make_llm()
    if progress:
        progress("Data quality pre-check: scanning for missing sections and ambiguous clauses...")
    agent = Agent(
        role="SOW Data Quality Reviewer",
        goal="Detect weak SOW quality signals before downstream phase execution.",
        backstory=(
            "You are strict about completeness and clarity. You flag missing sections, ambiguity, "
            "and statements likely to reduce execution quality."
        ),
        llm=llm,
        verbose=True,
    )
    task = Task(
        description=(
            "Review the SOW and return JSON with:\n"
            "- `ambiguous_clauses`: list of potentially ambiguous or weakly defined clauses\n"
            "- `quality_notes`: list of concise quality observations\n"
            "- `recommendations`: list of improvements to resolve ambiguity\n"
            "- `status`: PASS if quality is acceptable else RED\n\n"
            f'SOW TEXT:\n"""\n{sow}\n"""\n\n'
            "Return JSON ONLY:\n"
            "{\n"
            '  "ambiguous_clauses": ["..."],\n'
            '  "quality_notes": ["..."],\n'
            '  "recommendations": ["..."],\n'
            '  "status": "PASS"|"RED"\n'
            "}\n"
        ),
        expected_output="Strict JSON with data quality pre-check output.",
        agent=agent,
    )

    out = _kick_single(agent, task)
    if not isinstance(out, dict):
        out = {"status": "RED", "error": "Pre-check output parse failure"}

    llm_ambiguous = out.get("ambiguous_clauses") if isinstance(out.get("ambiguous_clauses"), list) else []
    combined_ambiguous = []
    for x in [*deterministic_ambiguous, *[str(i) for i in llm_ambiguous]]:
        if x not in combined_ambiguous:
            combined_ambiguous.append(x)

    quality_notes = out.get("quality_notes") if isinstance(out.get("quality_notes"), list) else []
    recommendations = out.get("recommendations") if isinstance(out.get("recommendations"), list) else []

    # Deterministic final status: any missing required section means RED.
    status = "RED" if missing_sections else str(out.get("status", "PASS")).upper()
    red_reasons: list[str] = []
    if missing_sections:
        red_reasons.append(
            "Required sections missing: " + ", ".join(missing_sections)
        )
    if str(out.get("status", "PASS")).upper() == "RED":
        red_reasons.append("LLM quality reviewer flagged the SOW as low quality.")
    if len(combined_ambiguous) >= 8:
        status = "RED"
        red_reasons.append("High ambiguity detected (8+ ambiguous clauses).")
    if status != "RED":
        red_reasons = []

    return {
        "status": status,
        "required_sections": section_checks,
        "missing_sections": missing_sections,
        "ambiguous_clauses": combined_ambiguous[:12],
        "quality_notes": [str(x) for x in quality_notes][:10],
        "recommendations": [str(x) for x in recommendations][:10],
        "red_reasons": red_reasons,
        "score": float(max(0, 100 - (20 * len(missing_sections)) - (5 * min(10, len(combined_ambiguous))))),
    }


def _extract_snippet(text: str, pattern: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return ""
    start = max(0, m.start() - 80)
    end = min(len(text), m.end() + 120)
    return text[start:end].replace("\n", " ").strip()


def _extract_net_days(text: str) -> int | None:
    m = re.search(r"\bnet\s*(\d{1,3})\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_amount_after_label(text: str, label_regex: str) -> float | None:
    m = re.search(label_regex, text, flags=re.IGNORECASE)
    if not m:
        return None
    raw = str(m.group(1) or "").replace(",", "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def run_po_sow_consistency_check(sow_text: str, po_text: str) -> dict[str, Any]:
    """
    Deterministic PO vs SOW consistency checker (advisory in V1).
    Returns business-friendly conflicts/review flags with evidence snippets.
    """
    sow = sow_text or ""
    po = po_text or ""
    conflicts: list[dict[str, str]] = []
    review_flags: list[dict[str, str]] = []

    po_id_present = bool(re.search(r"\bpo[-\s#:]*[a-z0-9-]{4,}\b", po, flags=re.IGNORECASE))
    if not po_id_present:
        conflicts.append(
            {
                "topic": "po_reference",
                "severity": "HIGH",
                "po_evidence": "PO identifier was not detected.",
                "sow_evidence": "",
                "why": "PO reference is required for auditable commercial traceability.",
            }
        )

    po_net_days = _extract_net_days(po)
    sow_net_days = _extract_net_days(sow)
    if po_net_days is not None and sow_net_days is not None and po_net_days != sow_net_days:
        conflicts.append(
            {
                "topic": "payment_terms",
                "severity": "HIGH",
                "po_evidence": _extract_snippet(po, r"\bnet\s*\d{1,3}\b"),
                "sow_evidence": _extract_snippet(sow, r"\bnet\s*\d{1,3}\b"),
                "why": f"Payment terms mismatch: PO is Net {po_net_days} while SOW is Net {sow_net_days}.",
            }
        )
    elif po_net_days is not None and sow_net_days is None:
        review_flags.append(
            {
                "topic": "payment_terms_sow_missing",
                "severity": "MEDIUM",
                "note": "PO has payment terms but SOW does not explicitly state Net terms.",
            }
        )

    po_total = _extract_amount_after_label(po, r"\btotal\b[^\n\r]{0,120}?([0-9][0-9,]*(?:\.\d{1,2})?)")
    sow_value = _extract_amount_after_label(
        sow, r"Estimated\s+Contract\s+Value\s*:\s*\$?\s*([0-9][0-9,]*(?:\.\d{1,2})?)"
    )
    commercial_summary: dict[str, Any] = {
        "po_total": po_total,
        "sow_estimated_contract_value": sow_value,
        "variance_abs": None,
        "variance_pct": None,
        "within_tolerance": None,
        "tolerance_basis": "max($100, 1% of SOW value) when both amounts are present",
        "tolerance_threshold_abs": None,
        "po_payment_net_days": po_net_days,
        "sow_payment_net_days": sow_net_days,
        "payment_terms_aligned": None,
    }
    if po_net_days is not None and sow_net_days is not None:
        commercial_summary["payment_terms_aligned"] = po_net_days == sow_net_days
    elif po_net_days is None and sow_net_days is None:
        commercial_summary["payment_terms_aligned"] = None
    else:
        commercial_summary["payment_terms_aligned"] = False

    if po_total is not None and sow_value is not None:
        delta = abs(po_total - sow_value)
        allowed_delta = max(100.0, sow_value * 0.01)
        commercial_summary["variance_abs"] = round(po_total - sow_value, 2)
        commercial_summary["variance_pct"] = round(
            ((po_total - sow_value) / sow_value * 100.0) if sow_value else 0.0, 2
        )
        commercial_summary["within_tolerance"] = delta <= allowed_delta
        commercial_summary["tolerance_threshold_abs"] = round(allowed_delta, 2)
        if delta > allowed_delta:
            conflicts.append(
                {
                    "topic": "commercial_amount",
                    "severity": "CRITICAL",
                    "po_evidence": _extract_snippet(po, r"\btotal\b[^\n\r]{0,80}[0-9][0-9,]*(?:\.\d{1,2})?"),
                    "sow_evidence": _extract_snippet(
                        sow, r"Estimated\s+Contract\s+Value\s*:\s*\$?\s*[0-9][0-9,]*(?:\.\d{1,2})?"
                    ),
                    "why": (
                        f"PO total (${po_total:,.2f}) vs SOW estimated contract value (${sow_value:,.2f}) "
                        f"varies by ${delta:,.2f} (beyond tolerance ${allowed_delta:,.2f})."
                    ),
                }
            )
    elif po_total is None:
        review_flags.append(
            {
                "topic": "po_total_missing",
                "severity": "MEDIUM",
                "note": "PO total amount was not deterministically extracted.",
            }
        )
    elif sow_value is None:
        review_flags.append(
            {
                "topic": "sow_value_missing",
                "severity": "MEDIUM",
                "note": "SOW estimated contract value is missing or not in expected format.",
            }
        )

    po_years = set(re.findall(r"\b(20\d{2})\b", po))
    sow_years = set(re.findall(r"\b(20\d{2})\b", sow))
    if po_years and sow_years and po_years.isdisjoint(sow_years):
        conflicts.append(
            {
                "topic": "date_alignment",
                "severity": "MEDIUM",
                "po_evidence": f"PO years detected: {', '.join(sorted(po_years))}",
                "sow_evidence": f"SOW years detected: {', '.join(sorted(sow_years))}",
                "why": "PO and SOW appear to reference different contract periods.",
            }
        )

    po_scope_tokens = [
        "data engg",
        "backend engg",
        "frontend",
        "full stack",
        "sre",
        "qa",
        "ux",
        "bsa",
        "tech writer",
        "ai initiative",
    ]
    po_scope_only = [t for t in po_scope_tokens if t in po.lower() and t not in sow.lower()]
    if po_scope_only:
        review_flags.append(
            {
                "topic": "scope_line_item_drift",
                "severity": "MEDIUM",
                "note": "PO contains role/scope terms not clearly found in SOW: " + ", ".join(po_scope_only[:6]),
            }
        )

    recommendations: list[str] = []
    topics = {str(c.get("topic", "")).lower() for c in conflicts}
    if "commercial_amount" in topics:
        recommendations.append("Reconcile PO total against SOW commercial value or attach approved change-order reference.")
    if "payment_terms" in topics:
        recommendations.append("Align payment terms between PO and SOW (e.g., Net days) before invoicing.")
    if "po_reference" in topics:
        recommendations.append("Ensure PO document includes an explicit PO identifier and revision.")
    if "date_alignment" in topics:
        recommendations.append("Align PO service dates with SOW timeline/term section.")
    if not recommendations:
        recommendations.append("No deterministic PO/SOW conflicts found; proceed with standard finance review.")
    if review_flags:
        recommendations.append("Address review flags to reduce downstream billing and governance disputes.")

    status = "RED" if conflicts else "PASS"
    summary = (
        "Potential PO/SOW commercial or governance mismatches detected; review before billing."
        if conflicts
        else "No deterministic PO/SOW mismatches detected."
    )
    return {
        "status": status,
        "summary": summary,
        "commercial_summary": commercial_summary,
        "conflicts": conflicts,
        "review_flags": review_flags,
        "recommendations": recommendations,
        "conflict_count": len(conflicts),
    }


def run_msa_consistency_check(sow_text: str, msa_text: str) -> dict[str, Any]:
    """
    Deterministic MSA vs SOW contradiction checker.
    Flags high-risk conflicts (warranty, liability, third-party/subprocessor usage).
    """
    sow = sow_text or ""
    msa = msa_text or ""
    conflicts: list[dict[str, str]] = []
    review_flags: list[dict[str, str]] = []

    # Warranty contradiction check.
    msa_has_warranty_limiter = bool(
        re.search(r"does\s+not\s+warrant|no\s+warrant(?:y|ies)|disclaim(?:ed|er)", msa, flags=re.IGNORECASE)
    )
    sow_has_strong_warranty = bool(
        re.search(
            r"unlimited\s+warrant|error[-\s]*free|uninterrupted|guarantee(?:s|d)?\s+(?:all|complete|full)",
            sow,
            flags=re.IGNORECASE,
        )
    )
    if msa_has_warranty_limiter and sow_has_strong_warranty:
        conflicts.append(
            {
                "topic": "warranty",
                "severity": "HIGH",
                "msa_evidence": _extract_snippet(msa, r"does\s+not\s+warrant|no\s+warrant(?:y|ies)|disclaim(?:ed|er)"),
                "sow_evidence": _extract_snippet(
                    sow,
                    r"unlimited\s+warrant|error[-\s]*free|uninterrupted|guarantee(?:s|d)?\s+(?:all|complete|full)",
                ),
                "why": "SOW appears to promise stronger warranty than MSA baseline.",
            }
        )

    # Liability contradiction check.
    msa_has_liability_cap = bool(
        re.search(r"liabilit(?:y|ies).{0,40}limit|aggregate\s+liabilit", msa, flags=re.IGNORECASE | re.DOTALL)
    )
    sow_unlimited_liability = bool(
        re.search(r"unlimited\s+liabilit|no\s+liabilit(?:y|ies)\s+cap", sow, flags=re.IGNORECASE)
    )
    if msa_has_liability_cap and sow_unlimited_liability:
        conflicts.append(
            {
                "topic": "liability",
                "severity": "CRITICAL",
                "msa_evidence": _extract_snippet(msa, r"liabilit(?:y|ies).{0,40}limit|aggregate\s+liabilit"),
                "sow_evidence": _extract_snippet(sow, r"unlimited\s+liabilit|no\s+liabilit(?:y|ies)\s+cap"),
                "why": "SOW appears to remove or override MSA liability limitation.",
            }
        )

    # Third-party / subprocessor contradiction check.
    msa_requires_notice_for_subprocessor = bool(
        re.search(r"subprocessor|third[-\s]*party.{0,50}(notice|consent|approval)", msa, flags=re.IGNORECASE | re.DOTALL)
    )
    sow_allows_unrestricted_third_party = bool(
        re.search(
            r"any\s+third[-\s]*party.{0,30}(without|no)\s+(notice|consent|approval)|at\s+vendor'?s\s+discretion",
            sow,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    if msa_requires_notice_for_subprocessor and sow_allows_unrestricted_third_party:
        conflicts.append(
            {
                "topic": "third_party_usage",
                "severity": "HIGH",
                "msa_evidence": _extract_snippet(msa, r"subprocessor|third[-\s]*party.{0,50}(notice|consent|approval)"),
                "sow_evidence": _extract_snippet(
                    sow,
                    r"any\s+third[-\s]*party.{0,30}(without|no)\s+(notice|consent|approval)|at\s+vendor'?s\s+discretion",
                ),
                "why": "SOW appears to allow third-party usage that may bypass MSA notice/consent controls.",
            }
        )

    # Review flags (not hard conflicts) to guide legal/commercial review.
    if re.search(r"\btermination\b", msa, flags=re.IGNORECASE) and re.search(r"\btermination\b", sow, flags=re.IGNORECASE):
        review_flags.append(
            {
                "topic": "termination",
                "severity": "MEDIUM",
                "note": "Both MSA and SOW include termination terms; confirm no precedence conflict.",
            }
        )
    if re.search(r"\bwarranty\b", msa, flags=re.IGNORECASE) and not re.search(r"\bwarranty\b", sow, flags=re.IGNORECASE):
        review_flags.append(
            {
                "topic": "warranty_silence",
                "severity": "LOW",
                "note": "SOW is silent on warranty; verify MSA-only handling is acceptable.",
            }
        )

    status = "RED" if conflicts else "PASS"
    summary = (
        "Potential MSA/SOW contradictions detected; legal review recommended."
        if conflicts
        else "No deterministic MSA/SOW contradictions detected."
    )
    return {
        "status": status,
        "conflicts": conflicts,
        "review_flags": review_flags,
        "summary": summary,
        "conflict_count": len(conflicts),
    }

