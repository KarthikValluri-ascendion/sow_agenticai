"""Persist executive-approved SOW plain text for future RAG retrieval."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def approved_sow_dir(root: Path) -> Path:
    return root / "knowledge_base" / "approved_sow"


def sanitize_filename_base(name: str) -> str:
    base = Path(name or "sow").stem
    base = re.sub(r"[^\w\-]+", "_", base, flags=re.UNICODE)
    return (base or "sow")[:80]


def archive_approved_sow_text(
    root: Path,
    *,
    run_id: str,
    sow_name: str,
    text: str,
    consolidated_score: float,
    system_go_ahead: bool,
    effective_go_ahead: bool,
    override_applied: bool,
) -> str | None:
    """
    Write approved SOW text to knowledge_base/approved_sow when executive path is GREEN.
    Caller must ensure effective_go_ahead is True (including override + score > 83 when applicable).
    """
    if not effective_go_ahead or not (text or "").strip():
        return None
    d = approved_sow_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rid = (run_id or "norun")[:8]
    slug = sanitize_filename_base(sow_name)
    fname = f"{ts}_{rid}_{slug}.md"
    path = d / fname
    header = (
        "<!-- archived_sow_for_rag "
        f"run_id={run_id} consolidated_score={consolidated_score:.2f} "
        f"system_go_ahead={system_go_ahead} effective_go_ahead={effective_go_ahead} "
        f"override_applied={override_applied} -->\n\n"
    )
    path.write_text(header + text.strip(), encoding="utf-8")
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)
