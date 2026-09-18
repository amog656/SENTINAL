from sqlalchemy.orm import Session, selectinload

from app.models.document import DocumentChunk, DocumentData, ExtractedMetadata
from app.storage.models import StoredChunk, StoredDocument


def save_document_with_chunks(
    db: Session,
    document: DocumentData,
    metadata: ExtractedMetadata,
    chunks: list[DocumentChunk],
) -> StoredDocument:
    existing = (
        db.query(StoredDocument)
        .options(selectinload(StoredDocument.chunks))
        .filter(StoredDocument.document_id == document.document_id)
        .one_or_none()
    )

    metadata_dict = metadata.model_dump()
    source_metadata = dict(document.metadata)
    source_metadata["file_document_type"] = document.document_type

    if existing:
        stored_document = existing
        stored_document.filename = document.filename
        stored_document.document_type = metadata.document_type
        stored_document.source = document.source
        stored_document.file_size = document.file_size
        stored_document.page_count = document.page_count
        stored_document.character_count = document.character_count
        stored_document.text = document.text
        stored_document.metadata_json = source_metadata
        stored_document.services = metadata.services
        stored_document.versions = metadata.versions
        stored_document.dates = metadata.dates
        stored_document.incident_ids = metadata.incident_ids
        stored_document.deployment_ids = metadata.deployment_ids
        stored_document.technical_entities = metadata.technical_entities
        db.query(StoredChunk).filter(StoredChunk.document_id == document.document_id).delete()
    else:
        stored_document = StoredDocument(
            document_id=document.document_id,
            filename=document.filename,
            document_type=metadata.document_type,
            source=document.source,
            file_size=document.file_size,
            page_count=document.page_count,
            character_count=document.character_count,
            text=document.text,
            metadata_json=source_metadata,
            services=metadata.services,
            versions=metadata.versions,
            dates=metadata.dates,
            incident_ids=metadata.incident_ids,
            deployment_ids=metadata.deployment_ids,
            technical_entities=metadata.technical_entities,
        )
        db.add(stored_document)

    for chunk in chunks:
        db.add(
            StoredChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                section=chunk.section,
                metadata_json=chunk.metadata,
            )
        )

    db.commit()
    db.refresh(stored_document)
    return stored_document


def get_document_by_id(db: Session, document_id: str) -> StoredDocument | None:
    return (
        db.query(StoredDocument)
        .options(selectinload(StoredDocument.chunks))
        .filter(StoredDocument.document_id == document_id)
        .one_or_none()
    )


def list_documents_with_chunks(db: Session) -> list[StoredDocument]:
    return (
        db.query(StoredDocument)
        .options(selectinload(StoredDocument.chunks))
        .order_by(StoredDocument.id)
        .all()
    )


def serialize_document(stored_document: StoredDocument, include_chunks: bool = True) -> dict:
    data = {
        "document_id": stored_document.document_id,
        "filename": stored_document.filename,
        "document_type": stored_document.document_type,
        "source": stored_document.source,
        "file_size": stored_document.file_size,
        "page_count": stored_document.page_count,
        "character_count": stored_document.character_count,
        "services": stored_document.services,
        "versions": stored_document.versions,
        "dates": stored_document.dates,
        "incident_ids": stored_document.incident_ids,
        "deployment_ids": stored_document.deployment_ids,
        "technical_entities": stored_document.technical_entities,
        "metadata": stored_document.metadata_json,
        "chunk_count": len(stored_document.chunks),
        "created_at": stored_document.created_at.isoformat() if stored_document.created_at else None,
    }

    if include_chunks:
        data["chunks"] = [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "section": chunk.section,
                "character_count": len(chunk.text),
                "text_preview": chunk.text[:500],
                "metadata": chunk.metadata_json,
            }
            for chunk in sorted(stored_document.chunks, key=lambda item: item.chunk_index)
        ]

    return data
