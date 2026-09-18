from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.ingestion.chunker import chunk_document
from app.ingestion.cleaner import clean_text_preserve_structure
from app.ingestion.metadata import extract_metadata
from app.ingestion.parser import parse_document
from app.models.document import DocumentData, generate_document_id
from app.storage.documents import save_document_with_chunks


def ingest_document(file_path: str, filename: Optional[str] = None, source: str = "upload") -> DocumentData:
    path = Path(file_path)
    original_filename = filename or path.name

    raw_text, document_type, file_size, page_count, metadata, _ = parse_document(file_path)
    cleaned_text = clean_text_preserve_structure(raw_text)

    if not cleaned_text:
        raise ValueError("Document contains no extractable text after cleaning")

    content = path.read_bytes()
    existing_id = metadata.get("document_id") if metadata else None
    document_id = generate_document_id(content, original_filename, document_type, existing_id=existing_id)

    return DocumentData(
        document_id=document_id,
        filename=original_filename,
        document_type=document_type,
        text=cleaned_text,
        source=source,
        file_size=file_size,
        page_count=page_count,
        character_count=len(cleaned_text),
        metadata=metadata or {},
    )


def ingest_and_persist_document(file_path: str, db: Session, filename: Optional[str] = None, source: str = "upload"):
    document = ingest_document(file_path, filename=filename, source=source)
    extracted_metadata = extract_metadata(document.filename, document.text, document.metadata)
    chunks = chunk_document(
        document_id=document.document_id,
        text=document.text,
        metadata=extracted_metadata.model_dump(),
    )
    stored_document = save_document_with_chunks(db, document, extracted_metadata, chunks)
    return stored_document
