"""Extract plain text from uploaded .docx and .pdf for the audit pipeline."""

from __future__ import annotations

import io
import os
import tempfile

import docx2txt


def read_docx_text(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return docx2txt.process(path) or ""
    finally:
        os.unlink(path)


def read_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()

