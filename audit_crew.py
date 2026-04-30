"""CrewAI Phase 1 (audit) and Phase 2 (PII mapping) using LangChain Groq."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

from crewai import Agent, Crew, Process, Task

from env_config import get_groq_api_key, get_groq_model, get_openai_base_url, load_dotenv_and_resolve
from groq_llm import GroqOpenAICompatLLM
from json_utils import extract_json_object

load_dotenv_and_resolve()

_DEFAULT_KB = Path(__file__).resolve().parent / "knowledge_base" / "process_circles.md"

# Option A: weighted 0–100 from per-agent PASS; executive go-ahead only if all PASS.
WEIGHT_AGENT1_PII = 34
WEIGHT_AGENT2_CIRCLE = 33
WEIGHT_AGENT3_GATE = 33


def _load_kb(path: Path | None = None) -> str:
    p = path or _DEFAULT_KB
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8")


def _make_llm() -> GroqOpenAICompatLLM:
    """Groq OpenAI-compatible API via `openai` SDK — no LiteLLM dependency."""
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "Set GROQ_API_KEY or test3 in the project `.env` file before running the app."
        )
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


def _serialize_task_output(task: Any) -> str:
    out = getattr(task, "output", None)
    if out is None:
        return ""
    for attr in ("raw", "raw_output", "result"):
        if hasattr(out, attr):
            val = getattr(out, attr)
            if val is not None:
                return str(val)
    return str(out)


def _parse_task_output(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    s = raw if isinstance(raw, str) else str(raw)
    return extract_json_object(s)


def _sow_mentions_compliance_gate(sow: str) -> bool:
    """Deterministic signals for Framework v2.1 Mandatory Action (compliance scan before kickoff)."""
    s = sow.lower()
    if re.search(r"automated\s+compliance", s):
        return True
    if re.search(r"compliance\s+scann", s):
        return True
    if re.search(r"delEx\s+portal", sow, re.IGNORECASE) and any(
        w in s for w in ("compliance", "scan", "block", "kickoff")
    ):
        return True
    if "compliance" in s and "kickoff" in s and ("scan" in s or "automated" in s):
        return True
    return False


def _merge_compliance_gate_verification(sow: str, out3: dict[str, Any]) -> dict[str, Any]:
    """If the model missed obvious compliance-gate language, upgrade RED → PASS with a note."""
    if not isinstance(out3, dict):
        return out3
    if str(out3.get("status", "")).upper() != "RED":
        return out3
    if not _sow_mentions_compliance_gate(sow):
        return out3
    merged = dict(out3)
    merged["governance_signal_detected"] = True
    merged["status"] = "PASS"
    note = "Deterministic check: SOW text contains compliance-scan / automated-compliance signals."
    merged["verification_note"] = note
    ev = merged.get("governance_signal_evidence")
    merged["governance_signal_evidence"] = (list(ev) if isinstance(ev, list) else []) + [note]
    return merged


def _normalize_governance_signal_output(out3: dict[str, Any]) -> dict[str, Any]:
    """Normalize Agent 3 output to business-friendly governance signal fields."""
    if not isinstance(out3, dict):
        return {
            "governance_signal_detected": False,
            "governance_signal_evidence": [],
            "governance_requirement": "Automated Compliance Scanning / DelEx compliance gate",
            "status": "RED",
        }

    detected = out3.get("governance_signal_detected")
    if detected is None:
        detected = out3.get("policy_103_keyword_present")
    if detected is None:
        detected = str(out3.get("status", "")).upper() == "PASS"

    evidence = out3.get("governance_signal_evidence")
    if not isinstance(evidence, list):
        legacy = out3.get("matched_evidence")
        evidence = list(legacy) if isinstance(legacy, list) else []

    requirement = str(
        out3.get("governance_requirement")
        or out3.get("keyword")
        or "Automated Compliance Scanning / DelEx compliance gate"
    )

    normalized: dict[str, Any] = {
        "governance_signal_detected": bool(detected),
        "governance_signal_evidence": [str(x) for x in evidence],
        "governance_requirement": requirement,
        "status": "PASS" if bool(detected) else "RED",
    }
    if "verification_note" in out3:
        normalized["verification_note"] = str(out3["verification_note"])
    return normalized


def _fallback_pii_mapping(pii_list: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s in pii_list:
        if not s or not str(s).strip():
            continue
        t = str(s).strip()
        if re.search(r"\S+@\S+\.\S+", t):
            out[t] = "[REDACTED_EMAIL]"
        elif re.search(r"[\d\-\(\)\+\.\s]{7,}", t) and re.search(r"\d{3}", t):
            out[t] = "[REDACTED_PHONE]"
        else:
            out[t] = "[REDACTED_NAME]"
    return out


def _agent_pass_weight(status: Any, weight: int) -> float:
    return float(weight) if str(status or "").upper() == "PASS" else 0.0


def compute_weighted_sow_ready_score(
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    agent3: dict[str, Any],
) -> float:
    """Sum of weights for each agent with status PASS (0–100)."""
    return (
        _agent_pass_weight(agent1.get("status"), WEIGHT_AGENT1_PII)
        + _agent_pass_weight(agent2.get("status"), WEIGHT_AGENT2_CIRCLE)
        + _agent_pass_weight(agent3.get("status"), WEIGHT_AGENT3_GATE)
    )


def executive_gates_all_pass(
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    agent3: dict[str, Any],
) -> bool:
    """True only when Agents 1–3 are all PASS — PD/manager go-ahead."""
    for a in (agent1, agent2, agent3):
        if str(a.get("status", "")).upper() != "PASS":
            return False
    return True


def enforce_sow_ready_score(
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    agent3: dict[str, Any],
    agent4_raw: dict[str, Any],
) -> dict[str, Any]:
    """Apply weighted score; traffic_light PASS only when all agents PASS."""
    merged = dict(agent4_raw)
    merged["agent1"] = agent1
    merged["agent2"] = agent2
    merged["agent3"] = agent3

    merged["sow_ready_score"] = compute_weighted_sow_ready_score(agent1, agent2, agent3)
    merged["executive_go_ahead"] = executive_gates_all_pass(agent1, agent2, agent3)
    merged["traffic_light"] = "PASS" if merged["executive_go_ahead"] else "RED"

    a1_status = str(agent1.get("status", "")).upper()
    merged["pii_gate"] = "BLOCKED" if a1_status == "RED" else "CLEAR"
    return merged


def _derive_policy_104_ethics(sow_text: str) -> dict[str, Any]:
    """Evaluate Policy 104 ethics controls from SOW text using deterministic signals."""
    s = (sow_text or "").lower()

    def _match_token(pattern: str) -> str:
        m = re.search(pattern, s)
        return str(m.group(0)) if m else ""

    if not s.strip():
        return {
            "id": "policy_104_ethics_secure_access",
            "title": "Ethics and Secure Workplace Access",
            "status": "RED",
            "violations": ["SOW text unavailable; cannot verify Policy 104 controls."],
            "evidence": ["SOW text unavailable in this run."],
            "recommendations": [
                "Provide SOW text and include explicit clauses for VDI-only execution, no unauthorized data sharing, and no mobile access."
            ],
            "checks": [
                {"theme": "Work within VDI", "met": False},
                {"theme": "No unauthorized sharing of data", "met": False},
                {"theme": "No access from mobile devices (unless approved)", "met": False},
            ],
        }

    vdi_match = _match_token(
        r"\b(vdi|virtual\s+desktop|citrix|secure\s+workspace|remote\s+desktop|approved\s+workspace)\b"
    )
    data_sharing_match = _match_token(
        r"\b(do\s*not\s*share|must\s*not\s*(share|disclose)|no\s+(external|unauthorized)\s+sharing|confidential(ity)?\s+(must\s+be\s+maintained|controls?)|prohibit(ed)?\s+disclosure)\b"
    )
    mobile_match = _match_token(
        r"\b(no\s+mobile|mobile\s+access\s+(is\s+)?prohibited|no\s+access\s+from\s+mobile|no\s+personal\s+devices|byod\s+prohibited|unmanaged\s+mobile)\b"
    )
    vdi_ok = bool(vdi_match)
    data_sharing_ok = bool(data_sharing_match)
    mobile_ok = bool(mobile_match)

    checks = [
        {"theme": "Work within VDI", "met": vdi_ok, "matched_text": vdi_match},
        {"theme": "No unauthorized sharing of data", "met": data_sharing_ok, "matched_text": data_sharing_match},
        {"theme": "No access from mobile devices (unless approved)", "met": mobile_ok, "matched_text": mobile_match},
    ]

    violations: list[str] = []
    recs: list[str] = []
    evidence: list[str] = []
    if not vdi_ok:
        violations.append("No clear VDI / secure-workspace execution requirement.")
        recs.append("Add a clause requiring work to be performed in client-approved VDI or secure workspace.")
        evidence.append("Work within VDI: no explicit VDI/secure-workspace phrase detected.")
    else:
        evidence.append(f'Work within VDI: matched "{vdi_match}".')
    if not data_sharing_ok:
        violations.append("No clear restriction against unauthorized data sharing/disclosure.")
        recs.append("Add a clause prohibiting unauthorized sharing/disclosure outside approved channels.")
        evidence.append("No unauthorized sharing of data: no explicit non-sharing/disclosure phrase detected.")
    else:
        evidence.append(f'No unauthorized sharing of data: matched "{data_sharing_match}".')
    if not mobile_ok:
        violations.append("No clear restriction on mobile or unmanaged-device access.")
        recs.append("Add a clause prohibiting mobile/unmanaged-device access unless explicitly approved with controls.")
        evidence.append("No access from mobile devices: no explicit mobile-access restriction phrase detected.")
    else:
        evidence.append(f'No access from mobile devices: matched "{mobile_match}".')

    return {
        "id": "policy_104_ethics_secure_access",
        "title": "Ethics and Secure Workplace Access",
        "status": "PASS" if all(x["met"] for x in checks) else "RED",
        "violations": violations,
        "evidence": evidence,
        "recommendations": (recs if recs else ["Policy 104 ethics controls are explicitly covered."]),
        "checks": checks,
    }


def build_policy_compliance(
    agent1: dict[str, Any],
    agent2: dict[str, Any],
    agent3: dict[str, Any],
    sow_text: str = "",
) -> dict[str, Any]:
    """Create explicit Policy 101/102/103/104 status and remediation guidance."""
    a1_status = str(agent1.get("status", "RED")).upper()
    a2_status = str(agent2.get("status", "RED")).upper()
    a3_status = str(agent3.get("status", "RED")).upper()

    pii_list = agent1.get("pii_list") if isinstance(agent1.get("pii_list"), list) else []
    req_deliverables = (
        agent2.get("required_deliverables")
        if isinstance(agent2.get("required_deliverables"), list)
        else []
    )
    found_deliverables = (
        agent2.get("deliverables_found")
        if isinstance(agent2.get("deliverables_found"), list)
        else []
    )
    missing_deliverables = (
        agent2.get("deliverables_missing")
        if isinstance(agent2.get("deliverables_missing"), list)
        else []
    )
    gate_evidence = (
        agent3.get("governance_signal_evidence")
        if isinstance(agent3.get("governance_signal_evidence"), list)
        else []
    )

    policy_101 = {
        "id": "policy_101_pii_hygiene",
        "title": "PII Hygiene Compliance",
        "status": a1_status,
        "violations": [str(x) for x in pii_list],
        "evidence": (
            ["No PII strings detected in SOW by Agent 1 scan."]
            if a1_status == "PASS"
            else [f'PII detected in SOW: "{str(x)}"' for x in pii_list[:10]]
        ),
        "recommendations": (
            ["Run redaction flow and replace all listed PII with approved placeholders."]
            if a1_status != "PASS"
            else ["PII hygiene control is satisfied."]
        ),
    }
    policy_102 = {
        "id": "policy_102_delivery_governance",
        "title": "Delivery Governance Alignment",
        "status": a2_status,
        "violations": [str(x) for x in missing_deliverables],
        "evidence": (
            [str(x) for x in found_deliverables[:10]]
            + ([f"Required controls for selected circle: {str(x)}" for x in req_deliverables[:10]] if not found_deliverables else [])
            + ([f"Missing coverage: {str(x)}" for x in missing_deliverables[:10]] if missing_deliverables else [])
        ),
        "recommendations": (
            [
                "Add explicit language for missing required deliverables and guardrails from the selected process circle.",
                "Re-run audit after updating SOW to include measurable governance commitments.",
            ]
            if a2_status != "PASS"
            else ["Delivery governance obligations are substantially covered."]
        ),
    }
    policy_103 = {
        "id": "policy_103_compliance_gate",
        "title": "Pre-Kickoff Compliance Gate",
        "status": a3_status,
        "violations": ([] if a3_status == "PASS" else ["No clear acknowledgment of pre-kickoff compliance scan obligation."]),
        "evidence": [str(x) for x in gate_evidence],
        "recommendations": (
            ["Add a clause confirming automated compliance scanning (or equivalent DelEx gate) before kickoff."]
            if a3_status != "PASS"
            else ["Compliance gate acknowledgment is present."]
        ),
    }
    policy_104 = _derive_policy_104_ethics(sow_text)

    all_status = [policy_101["status"], policy_102["status"], policy_103["status"], policy_104["status"]]
    overall = "PASS" if all(s == "PASS" for s in all_status) else "RED"
    return {
        "overall_status": overall,
        "policy_101": policy_101,
        "policy_102": policy_102,
        "policy_103": policy_103,
        "policy_104": policy_104,
    }


def _kick_single(
    agent: Agent,
    task: Task,
) -> dict[str, Any]:
    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    crew.kickoff()
    tasks = getattr(crew, "tasks", None) or [task]
    try:
        return _parse_task_output(_serialize_task_output(tasks[0]))
    except Exception:
        return {}


def run_audit_crew(
    sow_text: str,
    kb_text: str | None = None,
    ethics_manual_text: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """
    Four sequential CrewAI runs (Agents 1–4) so the UI can update st.status between agents.
    Overall behavior matches a single sequential process.
    """
    kb = kb_text if kb_text is not None else _load_kb()
    sow = _truncate(sow_text)
    kb_t = _truncate(kb, max_chars=40_000)
    ethics_t = _truncate(ethics_manual_text or "", max_chars=80_000) if ethics_manual_text else ""
    llm = _make_llm()

    if progress:
        progress("Agent 1 — Compliance Guardian: parsing SOW and scanning for PII...")
    agent1 = Agent(
        role="The Compliance Guardian",
        goal="Parse the SOW text and list exact PII strings that appear verbatim.",
        backstory="You are precise and conservative: only flag strings that clearly match names, emails, or phone numbers.",
        llm=llm,
        verbose=True,
    )
    task1 = Task(
        description=(
            "Analyze the Statement of Work text below.\n\n"
            f'SOW TEXT:\n"""\n{sow}\n"""\n\n'
            "Return JSON ONLY with this shape:\n"
            '{"status": "RED" or "PASS", "pii_list": ["exact string from SOW", ...]}\n'
            "Rules:\n"
            '- status is "RED" if any PII is found, else "PASS".\n'
            "- pii_list contains exact substrings as they appear (names, emails, phone numbers).\n"
            "- If none, pii_list is [].\n"
        ),
        expected_output='JSON: {"status":"RED"|"PASS","pii_list":[...]}',
        agent=agent1,
    )
    out1 = _kick_single(agent1, task1)

    if progress:
        progress("Agent 2 — Knowledge Librarian: Process Circle vs knowledge base...")
    agent2 = Agent(
        role="The Knowledge Librarian",
        goal="Classify the SOW into Process Circle Alpha, Beta, or Gamma per Framework v2.1 and verify core requirements + guardrails.",
        backstory=(
            "You apply the DelEx Project Circle Framework (matrix: engagement focus, core requirement, monitoring/guardrails). "
            "You judge semantic coverage, not only exact string matches."
        ),
        llm=llm,
        verbose=True,
    )
    ethics_section = ""
    if ethics_t.strip():
        ethics_section = (
            "\nASCENDION AI HYGIENE AND ETHICS MANUAL (reference; use for governance alignment).\n"
            f'"""\n{ethics_t}\n"""\n\n'
        )

    task2 = Task(
        description=(
            "Framework v2.1: use the KNOWLEDGE BASE (matrix + classification rules). "
            "1) Assign exactly one Process Circle: Alpha, Beta, or Gamma — justify implicitly by matching "
            "engagement focus (strategic long-term vs rapid scaling vs ad-hoc short-term).\n"
            "2) From the matrix row for that circle, build `required_deliverables` as the **Core Requirement** "
            "and **Key Monitoring & Guardrails** items (typically two bullets; Alpha may list weekly reporting as one item). "
            "3) For each required item, decide if the SOW **substantially addresses** it (synonyms/paraphrase OK).\n"
            "4) If an ethics manual excerpt is provided, flag only clear contradictions with hygiene/ethics.\n\n"
            f'KNOWLEDGE BASE:\n"""\n{kb_t}\n"""\n\n'
            f"{ethics_section}"
            f'SOW TEXT:\n"""\n{sow}\n"""\n\n'
            f"AGENT 1 OUTPUT (reference): {json.dumps(out1)}\n\n"
            "Return JSON ONLY:\n"
            "{\n"
            '  "process_circle": "Alpha"|"Beta"|"Gamma",\n'
            '  "classification_rationale": "one short sentence",\n'
            '  "required_deliverables": ["Core requirement / guardrail 1", "..."],\n'
            '  "deliverables_found": ["which requirement is covered and brief evidence"],\n'
            '  "deliverables_missing": ["..."],\n'
            '  "status": "PASS" if every required_deliverables entry is substantially met else "RED"\n'
            "}\n"
        ),
        expected_output="JSON with process_circle, required_deliverables, found/missing, status PASS or RED.",
        agent=agent2,
    )
    out2 = _kick_single(agent2, task2)

    if progress:
        progress("Agent 3 — Audit Specialist: Compliance Gate (Automated Compliance Scanning)...")
    agent3 = Agent(
        role="The Audit Specialist",
        goal="Verify the Mandatory Action: Automated Compliance Scanning / DelEx compliance gate before kickoff.",
        backstory=(
            "You check the SOW for acknowledgment of pre-kickoff automated compliance scanning or equivalent "
            "(DelEx portal, compliance gate). You report structured JSON only."
        ),
        llm=llm,
        verbose=True,
    )
    task3 = Task(
        description=(
            "Per Framework v2.1 **Mandatory Action**: PMs must run **Automated Compliance Scanning** before formal kickoff; "
            "failure blocks the project in the DelEx portal.\n"
            "PASS if the SOW clearly references automated compliance scanning, a pre-kickoff compliance scan, "
            "or DelEx/compliance gate readiness (semantic match OK).\n"
            "RED only if there is no reasonable mention of this obligation.\n\n"
            f'SOW TEXT:\n"""\n{sow}\n"""\n\n'
            f"AGENT 1 OUTPUT: {json.dumps(out1)}\n"
            f"AGENT 2 OUTPUT: {json.dumps(out2)}\n\n"
            "Return JSON ONLY:\n"
            "{\n"
            '  "governance_signal_detected": true|false,\n'
            '  "governance_signal_evidence": ["short quote or paraphrase from SOW, if any"],\n'
            '  "governance_requirement": "Automated Compliance Scanning / DelEx compliance gate",\n'
            '  "status": "PASS" if governance_signal_detected else "RED"\n'
            "}\n"
        ),
        expected_output='JSON with governance_signal_detected, governance_signal_evidence, status "PASS" or "RED".',
        agent=agent3,
    )
    out3 = _kick_single(agent3, task3)
    out3 = _normalize_governance_signal_output(out3)
    out3 = _merge_compliance_gate_verification(sow, out3)
    out3 = _normalize_governance_signal_output(out3)

    if progress:
        progress("Agent 4 — Master Architect: aggregating scorecard...")
    agent4 = Agent(
        role="The Master Architect",
        goal="Summarize Phase 1 audit results for leadership in plain language.",
        backstory=(
            "You synthesize Agents 1–3. Numeric score and go-ahead are computed by the pipeline, not by you."
        ),
        llm=llm,
        verbose=True,
    )
    task4 = Task(
        description=(
            "Write a short executive summary of the Phase 1 audit. Reference Agents 1–3; call out failures clearly.\n"
            "Do not invent percentages or PASS/RED for the overall score — those are set after the run.\n\n"
            f"AGENT 1 OUTPUT:\n{json.dumps(out1)}\n\n"
            f"AGENT 2 OUTPUT:\n{json.dumps(out2)}\n\n"
            f"AGENT 3 OUTPUT:\n{json.dumps(out3)}\n\n"
            "Return JSON ONLY:\n"
            '{\n  "summary": "short executive summary"\n}\n'
        ),
        expected_output='JSON: {"summary": "..."}',
        agent=agent4,
    )
    out4 = _kick_single(agent4, task4)

    master = enforce_sow_ready_score(out1, out2, out3, out4)
    master["policy_compliance"] = build_policy_compliance(out1, out2, out3, sow_text=sow)
    master["crew_note"] = "Phase 1 executed as four sequential single-agent crews for live UI progress."
    return master


def run_remediation_crew(pii_list: list[str]) -> dict[str, str]:
    """Agent 5: map each original PII string to a [REDACTED_*] placeholder."""
    if not pii_list:
        return {}
    llm = _make_llm()
    agent5 = Agent(
        role="The Data Sanitizer",
        goal="Produce a precise JSON dictionary mapping each exact PII string to a redaction token.",
        backstory="You never invent strings; you only map provided values to consistent [REDACTED_*] labels.",
        llm=llm,
        verbose=True,
    )
    payload = repr(list(pii_list))
    task = Task(
        description=(
            "Given this list of exact strings to redact from a Word document:\n"
            f"{payload}\n\n"
            "Return JSON ONLY: a single object whose keys are the EXACT original strings, "
            'and values are one of "[REDACTED_NAME]", "[REDACTED_EMAIL]", "[REDACTED_PHONE]" '
            "chosen appropriately. Do not add keys that were not listed.\n"
        ),
        expected_output='JSON object like {"name":"[REDACTED_NAME]","a@b.com":"[REDACTED_EMAIL]"}',
        agent=agent5,
    )
    crew = Crew(agents=[agent5], tasks=[task], process=Process.sequential, verbose=True)
    crew.kickoff()
    tasks = getattr(crew, "tasks", None) or [task]
    try:
        data = _parse_task_output(_serialize_task_output(tasks[0]))
        if not isinstance(data, dict):
            raise ValueError("mapping not a dict")
        mapping = {str(k): str(v) for k, v in data.items()}
        return mapping
    except Exception:
        return _fallback_pii_mapping(pii_list)
