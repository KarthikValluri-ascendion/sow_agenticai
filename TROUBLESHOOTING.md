# Windows / Python troubleshooting

## `Unable to initialize LLM` / LiteLLM (legacy)

This project **does not use LiteLLM** for Groq. It uses **`groq_llm.GroqOpenAICompatLLM`** (OpenAI Python SDK + Groq’s OpenAI-compatible URL). If you still see old LiteLLM errors, restart Streamlit after `git pull` / saving files so `audit_crew.py` is reloaded.

## `Connection error.` (Streamlit / CrewAI) vs “Tracing is disabled” (terminal)

- **`Connection error.`** comes from the OpenAI-compatible client when **HTTPS to `api.groq.com` fails** (no route, proxy, firewall, VPN, or SSL interception). It is **not** related to tracing.
- The terminal box **“Info: Tracing is disabled…”** is **normal CrewAI output**. It only explains how to *enable* CrewAI tracing; it does **not** mean something is broken.

**Fixes to try (in order):**

1. Confirm **`GROQ_API_KEY`** / **`test3`** is valid in [Groq Console](https://console.groq.com).
2. In **`.env`**, set **`GROQ_TRUST_SYSTEM_PROXY=false`** if a corporate **HTTP(S)_PROXY** breaks TLS to Groq.
3. Allow **`api.groq.com:443`** through firewall/VPN.
4. On Windows with **OneDrive long paths**, `pip` can fail; move the project to a short path (e.g. `C:\dev\...`) if installs break.

The app **retries** Groq calls with and without the system proxy automatically (`groq_llm.py`).

## `CERTIFICATE_VERIFY_FAILED` / `unable to get local issuer certificate`

Python cannot validate the HTTPS certificate chain to `api.groq.com`. Common on **corporate networks** (SSL inspection / custom CA).

1. **`pip install -U truststore certifi`** — the app tries **`truststore`** first (uses the **Windows/macOS/Linux certificate store**, which often includes your org’s intercepting CA), then **`certifi.where()`**.
2. Set **`GROQ_SSL_CERT`** in `.env` to a **`.pem` file** that includes your **organization’s root/intermediate CA** (IT often provides this). You can append it to the certifi bundle:  
   `type certifi.pem org-ca.pem > combined.pem` then set `GROQ_SSL_CERT=combined.pem`.
3. If truststore misbehaves, set **`GROQ_USE_SYSTEM_CERTS=false`** to use only **certifi** (after `pip install -U certifi`).
4. **Testing only:** **`GROQ_INSECURE_SSL=true`** turns off TLS verification (**insecure**; do not use for production).

## `SSL: UNEXPECTED_EOF_WHILE_READING` / connection drops

Often a **proxy or firewall** closing the TLS session (not only certificate trust). Try another network, VPN on/off, or **`GROQ_TRUST_SYSTEM_PROXY=false`**. The HTTP client uses **`http2=False`** to reduce issues with some proxies. For a quick local test only: **`GROQ_INSECURE_SSL=true`**.

## `Failed to connect to OpenAI API` / Groq

**`GROQ_API_KEY`** (or **`test3`** in `.env`) must be set. The app calls **`https://api.groq.com/openai/v1`** (override with **`OPENAI_BASE_URL`**). If connections fail, check VPN/firewall and that Groq’s API is reachable.

## `Fatal Python error: init_import_site` / `NotImplementedError` (truststore)

If **any** `python` or `pip` command fails with a traceback involving **`certifi_win32`**, **`wrapt_certifi`**, or **`truststore`**, Python 3.13 on your machine is likely conflicting with the **`python-certifi-win32`** hook.

**Fix (one-time):** remove these under your Python install’s `Lib\site-packages` (paths may vary slightly):

- Folder `certifi_win32`
- Folder `python_certifi_win32-*.dist-info`
- File `python-certifi-win32-init.pth`

Then open a **new** terminal and run `python --version` — it should start normally.

**Do not reinstall** `python-certifi-win32` unless a package you need explicitly requires it and a fixed version exists for Python 3.13.

## Step 1: `cd` to the project

Use quotes if the path has spaces:

```powershell
cd "c:\Users\karthik.valluri\OneDrive - ascendion\Desktop\AgenticAI_Ascendion\SOW_Governance_POC"
```

## Step 2: virtual environment

After Python works:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If execution policy blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Run Streamlit

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```
