import re
from typing import Dict, List, Optional

from app.models.document import DocumentChunk


TARGET_CHARS = 3000
OVERLAP_CHARS = 400
MIN_CHARS = 2500
HEADINGS = {
    "summary",
    "impact",
    "timeline",
    "root cause",
    "resolution",
    "mitigation",
    "lessons learned",
    "deployment",
    "symptoms",
    "investigation",
}


def detect_heading(line: str) -> Optional[str]:
    normalized = line.strip().strip(":").lower()
    if normalized in HEADINGS:
        return line.strip().strip(":")
    return None


def _find_split(text: str, start: int, target_end: int) -> int:
    if target_end >= len(text):
        return len(text)

    window_start = max(start + MIN_CHARS, target_end - 500)
    for pattern in (r"\n\n", r"\. ", r"\n", r" "):
        matches = list(re.finditer(pattern, text[window_start:target_end + 500]))
        if matches:
            return window_start + matches[-1].end()

    return target_end


def _section_for_position(lines_with_offsets: List[tuple[int, str]], position: int) -> Optional[str]:
    section = None
    for offset, line in lines_with_offsets:
        if offset > position:
            break
        heading = detect_heading(line)
        if heading:
            section = heading
    return section


def chunk_document(
    document_id: str,
    text: str,
    metadata: Dict,
    target_chars: int = TARGET_CHARS,
    overlap_chars: int = OVERLAP_CHARS,
) -> List[DocumentChunk]:
    if not text:
        return []

    lines_with_offsets = []
    offset = 0
    for line in text.splitlines():
        lines_with_offsets.append((offset, line))
        offset += len(line) + 1

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = _find_split(text, start, start + target_chars)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunk_id = f"{document_id}-C{chunk_index + 1:03d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk_index,
                    text=chunk_text,
                    section=_section_for_position(lines_with_offsets, start),
                    metadata=metadata,
                )
            )
            chunk_index += 1

        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
        while start < len(text) and text[start].isalnum() and text[start - 1].isalnum():
            start += 1

    return chunks
