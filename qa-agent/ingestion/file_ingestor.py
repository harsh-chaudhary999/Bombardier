"""
File-based PRD ingestor for Phase 2.

Accepts uploaded files (Excel, Word, PDF, Markdown, plain text) and converts
them to chunks ready for embedding.

Supported formats:
  .xlsx          — Excel (each sheet becomes a section)
  .docx          — Word document (headings become sections)
  .pdf           — PDF (page-based chunking)
  .md            — Markdown (section-based, same as GitLab/Confluence)
  .txt           — Plain text (no headings, single section)
"""
import io
import logging
import re
import tempfile
import os

from ingestion.doc_classify import classify as _classify_doc

logger = logging.getLogger(__name__)


# ─── Format converters ─────────────────────────────────────────────────────────

def _excel_to_markdown(content: bytes, filename: str) -> str:
    """Convert Excel file to markdown text (each sheet → H2 section)."""
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl required for Excel ingestion: pip install openpyxl")

    wb = openpyxl.load_workbook(io.BytesIO(content), read_only=True, data_only=True)
    parts = [f"# {filename}\n"]

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        parts.append(f"\n## {sheet_name}\n")

        rows_text: list[str] = []
        header_row: list[str] = []

        for i, row in enumerate(ws.iter_rows(values_only=True)):
            cells = [str(c).strip() if c is not None else "" for c in row]
            # Skip completely empty rows
            if not any(cells):
                continue
            if i == 0:
                header_row = cells
                rows_text.append("| " + " | ".join(cells) + " |")
                rows_text.append("| " + " | ".join(["---"] * len(cells)) + " |")
            else:
                rows_text.append("| " + " | ".join(cells) + " |")

        if rows_text:
            parts.append("\n".join(rows_text))

    wb.close()
    return "\n".join(parts)


def _docx_to_markdown(content: bytes, filename: str) -> str:
    """Convert Word document to markdown (heading styles → # headings)."""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("python-docx required for Word ingestion: pip install python-docx")

    doc = Document(io.BytesIO(content))
    lines: list[str] = [f"# {filename}\n"]

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        style = para.style.name if para.style else ""
        if "Heading 1" in style:
            lines.append(f"\n# {text}")
        elif "Heading 2" in style:
            lines.append(f"\n## {text}")
        elif "Heading 3" in style:
            lines.append(f"\n### {text}")
        else:
            lines.append(text)

    return "\n".join(lines)


def _pdf_to_text(content: bytes, filename: str) -> str:
    """Convert PDF to text (page-by-page, each page → H2 section)."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("pdfplumber required for PDF ingestion: pip install pdfplumber")

    parts = [f"# {filename}\n"]

    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                parts.append(f"\n## Page {i}\n\n{text}")

    return "\n".join(parts)


def _text_to_markdown(content: bytes, filename: str) -> str:
    """Plain text / markdown — decode and prepend title heading if not present."""
    text = content.decode("utf-8", errors="replace").strip()
    if not text.startswith("#"):
        return f"# {filename}\n\n{text}"
    return text


# ─── Main entry ───────────────────────────────────────────────────────────────

def convert_file_to_markdown(
    filename: str,
    content: bytes,
) -> str:
    """
    Convert an uploaded file to markdown/plain text for chunking.

    Args:
        filename: original filename (used to detect format and as doc title)
        content:  raw file bytes

    Returns:
        Markdown string ready for chunk_document()

    Raises:
        ValueError if format is unsupported
        ImportError if required library is not installed
    """
    ext = os.path.splitext(filename.lower())[1]
    # Clean up filename for use as title
    title = re.sub(r'\.[^.]+$', '', filename)  # strip extension

    if ext == ".xls":
        raise ValueError(
            "Legacy .xls format is not supported. Please convert to .xlsx (Excel 2007+) and re-upload."
        )
    if ext == ".xlsx":
        return _excel_to_markdown(content, title)
    elif ext == ".docx":
        return _docx_to_markdown(content, title)
    elif ext == ".pdf":
        return _pdf_to_text(content, title)
    elif ext in (".md", ".txt", ".text"):
        return _text_to_markdown(content, title)
    else:
        raise ValueError(
            f"Unsupported file format: {ext!r}. "
            "Supported: .xlsx, .xls, .docx, .pdf, .md, .txt"
        )


def ingest_file(filename: str, content: bytes, source_label: str = "", embed_fn=None) -> list[dict]:
    """
    Full ingestion flow for a single uploaded file.

    Converts file to markdown, then chunks it properly to stay within
    the embedding model's token limits.

    Args:
        filename:     original filename (determines format)
        content:      raw file bytes
        source_label: optional human label; defaults to filename without extension
    """
    from ingestion.chunker import chunk_document

    label = source_label or re.sub(r'\.[^.]+$', '', filename)
    # Namespace uploads so user-controlled labels cannot collide with confluence:/gitlab: IDs
    safe = re.sub(r"[^\w\-. ]", "_", label.strip())[:200] or "document"
    source_id = f"file:upload:{safe}"
    ext       = os.path.splitext(filename.lower())[1].lstrip(".")

    full_text = convert_file_to_markdown(filename, content)

    # Chunk the document properly instead of returning as one oversized block
    raw_chunks = chunk_document(full_text, source_id, embed_fn=embed_fn)

    if not raw_chunks:
        logger.warning(f"File {filename!r} produced no chunks after processing")
        return []

    logger.info(f"File {filename!r} → {len(raw_chunks)} chunks (source_id={source_id!r})")
    return [
        {
            "source_id":       source_id,
            "source_type":     f"file_{ext}" if ext else "file",
            "source_version":  "uploaded",
            "doc_title":       label,
            "doc_url":         None,
            "section_heading": c.get("section_heading", label),
            "chunk_text":      c["chunk_text"],
            "parent_text":     c.get("parent_text"),
            "doc_type":        _classify_doc(label),
            "chunk_index":     i,
        }
        for i, c in enumerate(raw_chunks)
    ]
