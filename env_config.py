"""Load `.env`, resolve `test3` → `GROQ_API_KEY`, and sync OpenAI-compatible env for CrewAI."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_LOADED = False


def load_dotenv_and_resolve() -> None:
    """Load project `.env` once; prefer `GROQ_API_KEY`, else copy from `test3`; set `OPENAI_*` for Groq OpenAI API."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(_ROOT / ".env")

    direct = (os.environ.get("GROQ_API_KEY") or "").strip()
    if not direct:
        via_test3 = (os.environ.get("test3") or "").strip()
        if via_test3:
            os.environ["GROQ_API_KEY"] = via_test3

    # Do not copy Groq key into OPENAI_API_KEY: LiteLLM/CrewAI may treat that as
    # real OpenAI (api.openai.com) and fail with "Connection error". Use
    # `crewai.LLM(model="groq/...")` + GROQ_API_KEY instead.


def get_groq_api_key() -> str:
    load_dotenv_and_resolve()
    return (os.environ.get("GROQ_API_KEY") or "").strip()


def get_groq_model() -> str:
    load_dotenv_and_resolve()
    return (os.environ.get("GROQ_MODEL") or "llama-3.3-70b-versatile").strip()


def get_openai_base_url() -> str:
    load_dotenv_and_resolve()
    return (os.environ.get("OPENAI_BASE_URL") or "").strip()


def get_stakeholder_webhook_url() -> str:
    load_dotenv_and_resolve()
    return (os.environ.get("STAKEHOLDER_WEBHOOK_URL") or "").strip()


def notify_stakeholder_webhook(title: str, *, payload: dict | None = None) -> bool:
    """POST JSON to STAKEHOLDER_WEBHOOK_URL if set. Returns True if the request succeeded."""
    load_dotenv_and_resolve()
    url = get_stakeholder_webhook_url()
    if not url:
        return False
    body_obj = {"title": title, **(payload or {})}
    body = json.dumps(body_obj).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False
