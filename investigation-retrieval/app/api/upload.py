import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.ingestion.parser import MAX_FILE_SIZE, SUPPORTED_EXTENSIONS, ParseError
from app.ingestion.service import ingest_and_persist_document
from app.search.bm25 import rebuild_bm25_index
from app.search.indexing import index_document_chunks
from app.storage.documents import get_document_by_id, serialize_document
from app.storage.postgres import get_db


logger = logging.getLogger(__name__)
router = APIRouter(tags=["document ingestion"])


def _preview_text(text: str, max_chars: int = 1000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


@router.post(
    "/upload",
    summary="Upload and parse a document",
    description="Validates, parses, cleans, and returns structured document data for PDF, DOCX, TXT, and JSON uploads.",
)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = file.filename or "uploaded-document"
    extension = Path(filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension or 'none'}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / f"upload{extension}"
            size = 0

            with temp_path.open("wb") as output:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_FILE_SIZE:
                        raise HTTPException(
                            status_code=400,
                            detail=f"File exceeds maximum upload size of {MAX_FILE_SIZE} bytes",
                        )
                    output.write(chunk)

            if size == 0:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")

            stored_document = ingest_and_persist_document(str(temp_path), db=db, filename=filename)
            indexing_status = index_document_chunks(stored_document)
            bm25_status = rebuild_bm25_index(db)

        document_data = serialize_document(stored_document, include_chunks=False)
        document_data.update(indexing_status)
        document_data["bm25_indexed_chunks"] = bm25_status["indexed_chunks"]
        return {
            "success": True,
            "document": document_data,
        }
    except HTTPException:
        raise
    except ParseError as exc:
        logger.exception("Document parsing failed: %s", filename)
        raise HTTPException(status_code=400, detail=exc.message if not exc.detail else f"{exc.message}: {exc.detail}") from exc
    except ValueError as exc:
        logger.exception("Document ingestion failed: %s", filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected upload failure: %s", filename)
        raise HTTPException(status_code=500, detail="Unexpected error while processing upload") from exc


@router.get(
    "/documents/{document_id}",
    summary="Get stored document metadata and chunks",
    description="Returns stored document metadata and chunk previews for a previously uploaded document.",
)
async def get_document(document_id: str, db: Session = Depends(get_db)):
    stored_document = get_document_by_id(db, document_id)
    if not stored_document:
        raise HTTPException(status_code=404, detail="Document not found")
    return {
        "success": True,
        "document": serialize_document(stored_document, include_chunks=True),
    }
