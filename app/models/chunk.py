import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime, ForeignKey, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.core.database import Base
from app.core.config import settings

class DeviceChunk(Base):
    __tablename__ = "device_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    device_id = Column(UUID(as_uuid=True), ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True)
    
    category = Column(String, nullable=False, index=True)
    content = Column(String, nullable=False)
    source_url = Column(String, nullable=True)
    metadata_json = Column(JSONB, nullable=False, default=dict)
    
    content_hash = Column(String, nullable=False, unique=True, index=True)
    embedding = Column(Vector(settings.EMBEDDING_DIMENSIONS), nullable=False)
    
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    device = relationship("Device", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("device_id", "content_hash", name="uq_device_chunk_hash"),
        Index(
            "ix_device_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
