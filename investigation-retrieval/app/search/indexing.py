import logging
from typing import Dict

from sqlalchemy.orm import Session

from app.embeddings.encoder import encode_texts
from app.storage.documents import list_documents_with_chunks
from app.storage.models import StoredDocument
from app.storage.qdrant import delete_document_vectors, upsert_chunk_vectors


logger = logging.getLogger(__name__)


def index_document_chunks(document: StoredDocument) -> Dict[str, object]:
    chunks = sorted(document.chunks, key=lambda chunk: chunk.chunk_index)
    if not chunks:
        return {"vector_indexed": True, "indexed_chunks": 0, "indexing_error": None}

    try:
        embeddings = encode_texts([chunk.text for chunk in chunks])
        delete_document_vectors(document.document_id)
        indexed_chunks = upsert_chunk_vectors(document, chunks, embeddings)
        return {"vector_indexed": True, "indexed_chunks": indexed_chunks, "indexing_error": None}
    except Exception as exc:
        logger.exception("Vector indexing failed for document %s", document.document_id)
        return {"vector_indexed": False, "indexed_chunks": 0, "indexing_error": str(exc)}


def reindex_all_documents(db: Session) -> Dict[str, object]:
    documents = list_documents_with_chunks(db)
    indexed_documents = 0
    indexed_chunks = 0
    failures = []

    for document in documents:
        result = index_document_chunks(document)
        if result["vector_indexed"]:
            indexed_documents += 1
            indexed_chunks += int(result["indexed_chunks"])
        else:
            failures.append({"document_id": document.document_id, "error": result["indexing_error"]})

    return {
        "documents_seen": len(documents),
        "indexed_documents": indexed_documents,
        "indexed_chunks": indexed_chunks,
        "failures": failures,
    }
