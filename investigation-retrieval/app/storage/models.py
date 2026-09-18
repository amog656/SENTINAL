from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, JSON, String, Text, func
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class StoredDocument(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(String(255), unique=True, nullable=False, index=True)
    filename = Column(String(1024), nullable=False)
    document_type = Column(String(100), nullable=False, index=True)
    source = Column(String(100), nullable=False, default="upload")
    file_size = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=True)
    character_count = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)
    services = Column(JSON, nullable=False, default=list)
    versions = Column(JSON, nullable=False, default=list)
    dates = Column(JSON, nullable=False, default=list)
    incident_ids = Column(JSON, nullable=False, default=list)
    deployment_ids = Column(JSON, nullable=False, default=list)
    technical_entities = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    chunks = relationship("StoredChunk", back_populates="document", cascade="all, delete-orphan")


class StoredChunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, index=True)
    chunk_id = Column(String(255), unique=True, nullable=False, index=True)
    document_id = Column(String(255), ForeignKey("documents.document_id", ondelete="CASCADE"), nullable=False, index=True)
    chunk_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    section = Column(String(255), nullable=True)
    metadata_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    document = relationship("StoredDocument", back_populates="chunks")


Index("ix_documents_created_at", StoredDocument.created_at)
Index("ix_chunks_document_chunk_index", StoredChunk.document_id, StoredChunk.chunk_index)
