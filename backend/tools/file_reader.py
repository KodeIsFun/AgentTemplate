"""`read_all_docs` — concatenates every readable file in `DOCS_DIR` into one corpus.

Supports plain-text extensions always, and PDF / DOCX when their libraries are
installed. Per-file text is hard-truncated to `MAX_CHARS_PER_FILE` and the
overall corpus to `MAX_CORPUS_CHARS`, both env-configurable. See tutorial
pitfall §"Context overflow".
"""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterable

from backend import config


# Plain-text extensions — read as UTF-8 (latin-1 fallback for legacy files).
TEXT_EXTS: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".rst",
        ".json",
        ".csv",
        ".tsv",
        ".html",
        ".htm",
        ".xml",
        ".py",
        ".js",
        ".ts",
        ".yml",
        ".yaml",
        ".ini",
        ".cfg",
        ".toml",
        ".log",
    }
)

PDF_EXTS: frozenset[str] = frozenset({".pdf"})
DOCX_EXTS: frozenset[str] = frozenset({".docx"})

HEADER_TMPL = "=== FILE: {rel} ===\n"
FOOTER = "=== END FILE ===\n"


def _iter_files(root: Path) -> Iterable[Path]:
    """Yield supported files under `root`, sorted for deterministic output."""
    for p in sorted(root.rglob("*")):
        if p.is_file():
            ext = p.suffix.lower()
            if ext in TEXT_EXTS or ext in PDF_EXTS or ext in DOCX_EXTS:
                yield p


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return ""  # give up silently; the file will be skipped


def _read_csv(path: Path) -> str:
    """Render a CSV/TSV as a simple text grid so models can scan rows."""
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f, delimiter=sep))
    except Exception:
        return _read_text(path)
    return "\n".join(" | ".join(row) for row in rows)


def _read_pdf(path: Path) -> str:
    try:
        import pdfplumber  # type: ignore
    except ImportError:
        return f"[pdfplumber not installed — cannot read {path.name}]"
    out: list[str] = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                if txt:
                    out.append(txt)
    except Exception as exc:  # malformed or encrypted PDF
        return f"[could not read PDF {path.name}: {exc}]"
    return "\n\n".join(out)


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx  # type: ignore
    except ImportError:
        return f"[python-docx not installed — cannot read {path.name}]"
    try:
        document = docx.Document(str(path))
        return "\n".join(p.text for p in document.paragraphs if p.text)
    except Exception as exc:
        return f"[could not read DOCX {path.name}: {exc}]"


def _read_one(path: Path, root: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".csv" or ext == ".tsv":
        body = _read_csv(path)
    elif ext in PDF_EXTS:
        body = _read_pdf(path)
    elif ext in DOCX_EXTS:
        body = _read_docx(path)
    else:
        body = _read_text(path)

    body = body.strip()
    limit = config.MAX_CHARS_PER_FILE
    if len(body) > limit:
        body = body[:limit] + "\n[…file truncated]"
    return HEADER_TMPL.format(rel=path.relative_to(root)) + body + "\n" + FOOTER


def read_all_docs() -> dict[str, str]:
    """Concatenate every readable file in `DOCS_DIR` into one string.

    Returns a dict (`{"status": "success"|"error", "corpus": str}`) so the LLM
    can see structured feedback (e.g. "no documents found") rather than having
    to parse free-form text — mirrors the `search_web` pattern in the tutorial.
    """
    root = config.docs_dir()

    try:
        files = list(_iter_files(root))
    except Exception as exc:
        return {"status": "error", "error": f"could not list {root}: {exc}", "corpus": ""}

    if not files:
        return {
            "status": "success",
            "files": 0,
            "corpus": f"[no documents found in {root}]",
        }

    parts: list[str] = []
    budget = config.MAX_CORPUS_CHARS
    truncated = False
    for path in files:
        chunk = _read_one(path, root)
        if len(chunk) > budget:
            chunk = chunk[:budget] + "\n[…corpus truncated]"
            parts.append(chunk)
            truncated = True
            break
        parts.append(chunk)
        budget -= len(chunk)

    return {
        "status": "success",
        "files": len(files),
        "truncated": truncated,
        "corpus": "\n".join(parts),
    }
