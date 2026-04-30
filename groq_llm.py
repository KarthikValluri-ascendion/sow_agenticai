"""Groq via OpenAI-compatible HTTPS API — no LiteLLM package required."""

from __future__ import annotations

import os
import ssl
from typing import Any

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from pydantic import BaseModel

from crewai.agent.core import Agent
from crewai.events.types.llm_events import LLMCallType
from crewai.llms.base_llm import BaseLLM, llm_call_context
from crewai.task import Task
from crewai.tools.base_tool import BaseTool
from crewai.utilities.types import LLMMessage

DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# httpx: HTTP/2 can confuse some corporate proxies → EOF / SSL issues
_HTTPX_EXTRA = {"http2": False}


def normalize_groq_openai_base_url(url: str | None) -> str:
    """
    Groq chat completions expect base URL ending in /openai/v1.
    Accepts common variants (missing /v1, host-only).
    """
    u = (url or "").strip() or DEFAULT_GROQ_BASE_URL
    u = u.rstrip("/")
    if u.startswith("http://"):
        u = "https://" + u[len("http://") :]
    host_only = ("api.groq.com" in u and "/openai" not in u) or u.endswith("api.groq.com")
    if host_only:
        return DEFAULT_GROQ_BASE_URL
    if u.endswith("/openai"):
        u = u + "/v1"
    if "api.groq.com" in u and "/openai/v1" not in u and "/openai" in u:
        if not u.endswith("/v1"):
            u = u + "/v1"
    return u


def _tls_verify_modes() -> list[tuple[str, bool | str | ssl.SSLContext]]:
    """
    Modes to try in order (corporate Windows often needs OS trust store via truststore).

    - GROQ_INSECURE_SSL=true → only verify=False (testing; insecure).
    - GROQ_SSL_CERT=path → use that PEM only.
    - Else: truststore (system CAs) if GROQ_USE_SYSTEM_CERTS (default true), then certifi.
    """
    if os.environ.get("GROQ_INSECURE_SSL", "").lower() in ("1", "true", "yes"):
        return [("verify_false", False)]

    custom = (os.environ.get("GROQ_SSL_CERT") or "").strip()
    if custom:
        return [("GROQ_SSL_CERT", custom)]

    modes: list[tuple[str, bool | str | ssl.SSLContext]] = []
    use_system = os.environ.get("GROQ_USE_SYSTEM_CERTS", "true").lower() in (
        "1",
        "true",
        "yes",
    )
    if use_system:
        try:
            import truststore

            modes.append(
                ("truststore_os", truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT))
            )
        except Exception:
            pass
    try:
        import certifi

        modes.append(("certifi", certifi.where()))
    except ImportError:
        pass
    if not modes:
        modes.append(("default", True))
    return modes


def _http_client(trust_env: bool, verify: bool | str | ssl.SSLContext) -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(180.0, connect=45.0, read=180.0, write=60.0),
        trust_env=trust_env,
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        verify=verify,
        **_HTTPX_EXTRA,
    )


def _connection_help(base_url: str) -> str:
    return (
        f"Could not reach Groq at `{base_url}`. "
        "(1) Firewall/VPN: allow `api.groq.com:443`. "
        "(2) **`pip install truststore certifi`** — app tries **OS cert store** then **certifi**. "
        "(3) Corporate TLS: set **`GROQ_SSL_CERT`** to a PEM with your org CA, or **`GROQ_INSECURE_SSL=true`** "
        "for local testing only. "
        "(4) **`GROQ_USE_SYSTEM_CERTS=false`** forces certifi only. "
        "(5) Proxy: `HTTPS_PROXY` or **`GROQ_TRUST_SYSTEM_PROXY=false`**. "
        "(6) Valid **`GROQ_API_KEY`** / **`test3`**."
    )


def _strip_groq_prefix(model: str) -> str:
    m = model.strip()
    if m.lower().startswith("groq/"):
        return m[5:]
    return m


def _message_content(msg: dict[str, Any]) -> str:
    c = msg.get("content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for block in c:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(c)


def _should_trust_system_proxy_first() -> bool:
    return os.environ.get("GROQ_TRUST_SYSTEM_PROXY", "true").lower() in (
        "1",
        "true",
        "yes",
    )


class GroqOpenAICompatLLM(BaseLLM):
    """Groq chat completions using the OpenAI Python SDK (OpenAI-compatible endpoint)."""

    is_litellm: bool = False

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        base_url: str | None = None,
    ) -> None:
        api_model = _strip_groq_prefix(model)
        url = normalize_groq_openai_base_url(base_url)
        super().__init__(
            model=api_model,
            temperature=temperature,
            api_key=api_key,
            base_url=url,
            provider="groq",
        )
        self._groq_base_url = url

    def _openai_sdk_complete(
        self,
        openai_msgs: list[dict[str, str]],
        trust_env: bool,
        verify: bool | str | ssl.SSLContext,
    ) -> str:
        client = OpenAI(
            api_key=self.api_key,
            base_url=self._groq_base_url,
            max_retries=2,
            http_client=_http_client(trust_env, verify),
        )
        resp = client.chat.completions.create(
            model=self.model,
            messages=openai_msgs,
            temperature=float(self.temperature if self.temperature is not None else 0.0),
        )
        return (resp.choices[0].message.content or "").strip()

    def _direct_httpx_complete(
        self,
        openai_msgs: list[dict[str, str]],
        trust_env: bool,
        verify: bool | str | ssl.SSLContext,
    ) -> str:
        """POST /chat/completions without the OpenAI SDK client."""
        url = f"{self._groq_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": openai_msgs,
            "temperature": float(self.temperature if self.temperature is not None else 0.0),
        }
        with _http_client(trust_env, verify) as h:
            r = h.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            r.raise_for_status()
            data = r.json()
        return (data["choices"][0]["message"].get("content") or "").strip()

    def _complete_with_retries(self, openai_msgs: list[dict[str, str]]) -> str:
        """Try TLS modes × proxy settings × OpenAI SDK, then direct httpx."""
        trust_pref = _should_trust_system_proxy_first()
        trust_sequence = [trust_pref, not trust_pref]
        tls_modes = _tls_verify_modes()
        conn_errors: list[str] = []

        for tls_name, verify in tls_modes:
            for te in trust_sequence:
                try:
                    return self._openai_sdk_complete(openai_msgs, te, verify)
                except APIStatusError as e:
                    hint = " Check GROQ_API_KEY / test3 in .env." if e.status_code in (401, 403) else ""
                    raise RuntimeError(f"Groq API HTTP {e.status_code}: {e!s}{hint}") from e
                except (APIConnectionError, APITimeoutError) as e:
                    conn_errors.append(f"OpenAI SDK [{tls_name}, trust_env={te}]: {e!s}")

        for tls_name, verify in tls_modes:
            for te in trust_sequence:
                try:
                    return self._direct_httpx_complete(openai_msgs, te, verify)
                except httpx.HTTPStatusError as e:
                    code = e.response.status_code
                    hint = " Check GROQ_API_KEY / test3 in .env." if code in (401, 403) else ""
                    raise RuntimeError(f"Groq API HTTP {code}: {e!s}{hint}") from e
                except (
                    httpx.ConnectError,
                    httpx.TimeoutException,
                    httpx.ReadTimeout,
                    httpx.ReadError,
                ) as e:
                    conn_errors.append(f"httpx [{tls_name}, trust_env={te}]: {e!s}")

        detail = " | ".join(conn_errors)
        msg = f"{_connection_help(self._groq_base_url)} Attempts: {detail}"
        if "CERTIFICATE_VERIFY_FAILED" in detail or "SSL" in detail.upper():
            msg += (
                " **SSL:** Install `truststore`, set **`GROQ_SSL_CERT`**, or **`GROQ_INSECURE_SSL=true`** (testing only)."
            )
        if "UNEXPECTED_EOF" in detail or "EOF" in detail:
            msg += " **EOF:** Often proxy/firewall; try another network or VPN off, or `GROQ_INSECURE_SSL=true` for testing."
        raise RuntimeError(msg)

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, BaseTool]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Task | None = None,
        from_agent: Agent | None = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        if response_model is not None:
            raise NotImplementedError(
                "Structured outputs are not implemented for GroqOpenAICompatLLM."
            )
        if tools:
            raise NotImplementedError(
                "Tool calling is not implemented for GroqOpenAICompatLLM."
            )
        with llm_call_context():
            self._emit_call_started_event(
                messages=messages,
                tools=tools,
                callbacks=callbacks,
                available_functions=available_functions,
                from_task=from_task,
                from_agent=from_agent,
            )
            try:
                formatted = self._format_messages(messages)
                openai_msgs = [
                    {"role": m["role"], "content": _message_content(m)} for m in formatted
                ]
                text = self._complete_with_retries(openai_msgs)
                if self.stop:
                    text = self._apply_stop_words(text)
                self._emit_call_completed_event(
                    response=text,
                    call_type=LLMCallType.LLM_CALL,
                    from_task=from_task,
                    from_agent=from_agent,
                    messages=messages,
                )
                return text
            except Exception as e:
                self._emit_call_failed_event(
                    error=str(e),
                    from_task=from_task,
                    from_agent=from_agent,
                )
                raise
