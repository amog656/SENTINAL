import json
import os
from typing import Tuple, Optional, Dict, Any
from pathlib import Path

import pymupdf
from docx import Document


SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.json'}
MAX_FILE_SIZE = 25 * 1024 * 1024


class ParseError(Exception):
    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(message)


def validate_file(file_path: str) -> Tuple[str, int]:
    path = Path(file_path)
    
    if not path.exists():
        raise ParseError("File not found", f"Path does not exist: {file_path}")
    
    if path.stat().st_size == 0:
        raise ParseError("Empty file", "Uploaded file has zero bytes")
    
    if path.stat().st_size > MAX_FILE_SIZE:
        raise ParseError(
            "File too large",
            f"File size {path.stat().st_size} bytes exceeds maximum of {MAX_FILE_SIZE} bytes"
        )
    
    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ParseError(
            "Unsupported file type",
            f"Extension '{extension}' not supported. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    return extension, path.stat().st_size


def parse_pdf(file_path: str) -> Tuple[str, int, Dict[str, Any]]:
    try:
        doc = pymupdf.open(file_path)
    except Exception as e:
        raise ParseError("Corrupted or invalid PDF", str(e))
    
    page_count = doc.page_count
    if page_count == 0:
        doc.close()
        raise ParseError("Empty PDF", "PDF has no pages")
    
    text_parts = []
    for page_num in range(page_count):
        page = doc[page_num]
        page_text = page.get_text()
        if page_text.strip():
            text_parts.append(f"PAGE {page_num + 1}\n{page_text}")
    
    doc.close()
    
    full_text = "\n\n".join(text_parts)
    if not full_text.strip():
        raise ParseError(
            "No extractable text",
            "PDF appears to be scanned or image-based. OCR is not supported in this phase."
        )
    
    return full_text, page_count, {}


def parse_docx(file_path: str) -> Tuple[str, Optional[int], Dict[str, Any]]:
    try:
        doc = Document(file_path)
    except Exception as e:
        raise ParseError("Corrupted or invalid DOCX", str(e))
    
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    
    full_text = "\n\n".join(paragraphs)
    
    if not full_text.strip():
        raise ParseError("Empty DOCX", "Document contains no readable text")
    
    return full_text, None, {}


def parse_txt(file_path: str) -> Tuple[str, Optional[int], Dict[str, Any]]:
    encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
            return text, None, {"encoding_used": encoding}
        except UnicodeDecodeError:
            continue
    
    raise ParseError("Encoding error", "Could not decode text file with common encodings")


def parse_json(file_path: str) -> Tuple[str, Optional[int], Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ParseError("Invalid JSON", f"JSON parse error: {str(e)}")
    except UnicodeDecodeError:
        raise ParseError("Encoding error", "JSON file must be UTF-8 encoded")
    
    if not isinstance(data, dict):
        raise ParseError("Invalid JSON structure", "Root element must be a JSON object")
    
    document_id = data.get('document_id')
    
    text_fields = ['text', 'content', 'description', 'body', 'details']
    extracted_text = None
    
    for field in text_fields:
        if field in data and isinstance(data[field], str) and data[field].strip():
            extracted_text = data[field].strip()
            break
    
    if extracted_text:
        metadata = {k: v for k, v in data.items() if k != field}
        metadata["original_json"] = data
        full_text = extracted_text
    else:
        full_text = json.dumps(data, indent=2, ensure_ascii=False)
        metadata = {"original_json": data, **data}
    
    return full_text, None, {"document_id": document_id, "original_json": data, **metadata}


def parse_document(file_path: str) -> Tuple[str, str, int, Optional[int], Dict[str, Any], str]:
    extension, file_size = validate_file(file_path)
    
    parsers = {
        '.pdf': parse_pdf,
        '.docx': parse_docx,
        '.txt': parse_txt,
        '.json': parse_json,
    }
    
    parser = parsers[extension]
    text, page_count, metadata = parser(file_path)
    
    document_type = extension[1:]
    
    return text, document_type, file_size, page_count, metadata, extension


def get_document_type_from_extension(extension: str) -> str:
    return extension.lower().lstrip('.')
