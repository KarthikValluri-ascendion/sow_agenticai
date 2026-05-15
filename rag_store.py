"""Lightweight lexical RAG: chunking, TF-IDF retrieval, context formatting."""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from documents import read_docx_text

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore[misc, assignment]
    cosine_similarity = None  # type: ignore[misc, assignment]


@dataclass
class DocumentChunk:
    source: str
    section: str
    chunk_id: str
    text: str
    score: float | None = None


def _slug(s: str, max_len: int = 48) -> str:
    t = re.sub(r"[^\w]+", "_", (s or "").strip(), flags=re.UNICODE).strip("_")
    return (t or "src")[:max_len]


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """Split plain text into overlapping windows with stable chunk ids."""
    raw = (text or "").strip()
    if not raw:
        return []
    base = _slug(source)
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    step = max(1, chunk_size - overlap)
    chunks: list[DocumentChunk] = []
    i = 0
    n = 0
    while i < len(raw):
        piece = raw[i : i + chunk_size]
        cid = f"{base}-{n:04d}"
        chunks.append(DocumentChunk(source=source, section="", chunk_id=cid, text=piece))
        n += 1
        i += step
    return chunks


def _read_approved_sow_files(approved_dir: Path) -> list[tuple[str, str]]:
    """Load (label, text) from archived approved SOWs for indexing."""
    if not approved_dir.is_dir():
        return []
    out: list[tuple[str, str]] = []
    for p in sorted(approved_dir.iterdir()):
        if p.name.startswith(".") or p.name == ".gitkeep":
            continue
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        try:
            if suf in (".md", ".txt"):
                txt = p.read_text(encoding="utf-8", errors="replace")
            elif suf == ".docx":
                txt = read_docx_text(p.read_bytes())
            else:
                continue
        except OSError:
            continue
        label = f"knowledge_base/approved_sow/{p.name}"
        if txt.strip():
            out.append((label, txt))
    return out


def build_corpus(
    *,
    kb_text: str,
    ethics_text: str,
    msa_text: str,
    po_text: str,
    sow_text: str,
    sow_source_label: str,
    root: Path,
    kb_path_label: str = "knowledge_base/process_circles.md",
) -> list[DocumentChunk]:
    """Build all chunks for the current run (uploads + defaults + approved library)."""
    chunks: list[DocumentChunk] = []

    def add_labeled(label: str, body: str) -> None:
        b = (body or "").strip()
        if not b:
            return
        chunks.extend(chunk_text(b, label))

    add_labeled(kb_path_label, kb_text)
    add_labeled("uploaded_ethics_manual", ethics_text)
    add_labeled("uploaded_msa", msa_text)
    add_labeled("uploaded_po", po_text)
    sow_label = sow_source_label or "uploaded_sow.docx"
    if not sow_label.lower().endswith((".docx", ".md", ".pdf")):
        sow_label = sow_label + ".docx"
    add_labeled(sow_label, sow_text)

    approved_dir = root / "knowledge_base" / "approved_sow"
    for label, body in _read_approved_sow_files(approved_dir):
        add_labeled(label, body)

    return chunks


def corpus_fingerprint(chunks: Iterable[DocumentChunk]) -> str:
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        h.update(c.chunk_id.encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()[:16]


def retrieve(
    query: str,
    chunks: list[DocumentChunk],
    top_k: int = 5,
) -> list[DocumentChunk]:
    """Rank chunks by TF-IDF cosine similarity to the query."""
    if not chunks or not (query or "").strip():
        return []
    if TfidfVectorizer is None or cosine_similarity is None:
        return []

    texts = [c.text for c in chunks]
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            min_df=1,
            ngram_range=(1, 2),
        )
        doc_matrix = vectorizer.fit_transform(texts)
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, doc_matrix).flatten()
    except ValueError:
        return []

    ranked_idx = sorted(range(len(sims)), key=lambda i: float(sims[i]), reverse=True)[:top_k]
    out: list[DocumentChunk] = []
    for i in ranked_idx:
        c = chunks[i]
        out.append(
            DocumentChunk(
                source=c.source,
                section=c.section,
                chunk_id=c.chunk_id,
                text=c.text,
                score=float(sims[i]),
            )
        )
    return out


def format_retrieved_context(scored_chunks: list[DocumentChunk]) -> str:
    """Single block for prompt injection."""
    if not scored_chunks:
        return ""
    lines: list[str] = ['RETRIEVED GOVERNANCE EVIDENCE (lexical match; cite when used):', ""]
    for c in scored_chunks:
        score_note = f" score={c.score:.4f}" if c.score is not None else ""
        lines.append(f"[source={c.source} chunk={c.chunk_id}{score_note}]")
        lines.append(c.text.strip())
        lines.append("")
    return "\n".join(lines).strip()


def rag_evidence_rows(chunks: list[DocumentChunk], *, reason: str) -> list[dict[str, Any]]:
    """Serializable rows for Streamlit / run history."""
    rows: list[dict[str, Any]] = []
    for c in chunks:
        excerpt = (c.text[:400] + "…") if len(c.text) > 400 else c.text
        rows.append(
            {
                "source": c.source,
                "chunk_id": c.chunk_id,
                "excerpt": excerpt.replace("\n", " "),
                "lexical_score": c.score,
                "reason": reason,
            }
        )
    return rows


def merge_chunks_unique(chunks_lists: list[list[DocumentChunk]], top_k: int = 8) -> list[DocumentChunk]:
    """Merge ranked chunk lists, dedupe by chunk_id, keep highest score."""
    best: dict[str, DocumentChunk] = {}
    for lst in chunks_lists:
        for c in lst:
            cur = best.get(c.chunk_id)
            s = c.score if c.score is not None else -1.0
            if cur is None or s > (cur.score if cur.score is not None else -1.0):
                best[c.chunk_id] = c
    merged = sorted(best.values(), key=lambda x: x.score if x.score is not None else -1.0, reverse=True)
    return merged[:top_k]


# Lexical retrieval queries per workflow step (app may override).
STEP_QUERIES: dict[str, str] = {
    "process_circle": (
        "process circle deliverables governance guardrails kickoff compliance "
        "framework alpha beta gamma monitoring DelEx"
    ),
    "mavca": (
        "MAVCA manual augmented validated curated autonomous AI hygiene "
        "task decomposition agent-first classification"
    ),
    "msa": (
        "warranty liability termination indemnity third party precedence "
        "subprocessor consent limitation of liability"
    ),
    "po": (
        "payment terms purchase order contract value dates scope billing "
        "net days total amount PO identifier"
    ),
}
