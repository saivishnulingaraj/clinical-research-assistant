"""Load source documents from disk into :class:`Document` objects.

Supports Markdown (``.md``) and plain text (``.txt``). Markdown files may carry
a small YAML front-matter block delimited by ``---`` lines providing ``doc_id``,
``title`` and ``source_url``. Section structure is derived from ``#``/``##``
headings; files without headings become a single untitled section.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from cra.ingest.models import Document

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# Matches ATX headings (# .. ######) capturing the heading text.
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$", re.MULTILINE)


def _parse_front_matter(raw: str) -> tuple[dict, str]:
    """Split optional YAML front-matter from the document body."""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta = yaml.safe_load(match.group(1)) or {}
    body = raw[match.end() :]
    return meta, body


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Split a Markdown body into (heading, text) pairs.

    Text before the first heading is attached to an empty-heading section. The
    top-level ``# Title`` heading is treated as a section like any other so its
    prose (if any) is retained, but typically the title repeats the front-matter.
    """
    headings = list(_HEADING_RE.finditer(body))
    if not headings:
        text = body.strip()
        return [("", text)] if text else []

    sections: list[tuple[str, str]] = []

    # Preamble before the first heading.
    preamble = body[: headings[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for i, h in enumerate(headings):
        heading_text = h.group(1).strip()
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        section_text = body[start:end].strip()
        if section_text:
            sections.append((heading_text, section_text))
    return sections


def load_document(path: Path) -> Document:
    """Parse a single file into a :class:`Document`."""
    raw = path.read_text(encoding="utf-8")
    meta, body = _parse_front_matter(raw)

    doc_id = str(meta.get("doc_id") or path.stem)
    sections = _split_sections(body)

    # Derive a title: front-matter wins, else first heading, else filename.
    title = meta.get("title")
    if not title:
        heading_match = _HEADING_RE.search(body)
        title = heading_match.group(1).strip() if heading_match else path.stem
    source_url = str(meta.get("source_url") or f"file://{path}")
    synthetic = bool(meta.get("synthetic", True))

    return Document(
        doc_id=doc_id,
        title=str(title),
        source_url=source_url,
        sections=sections,
        synthetic=synthetic,
    )


def load_documents(source_dir: Path) -> list[Document]:
    """Load every ``.md``/``.txt`` file in ``source_dir`` (recursively).

    ``README.md`` files are skipped so corpus documentation is not indexed.
    """
    docs: list[Document] = []
    for path in sorted(source_dir.rglob("*")):
        if path.suffix.lower() not in {".md", ".txt"}:
            continue
        if path.name.lower() == "readme.md":
            continue
        docs.append(load_document(path))
    return docs
