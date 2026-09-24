import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, Integer, Numeric, DateTime, Index
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.core.database import Base


class Device(Base):
    """Canonical device record.

    Populated exclusively by the structured ETL pipeline in
    app/db/dynamic_ingest.py — never by naive text chunking. `specs_json`
    is the single source of truth for every downstream consumer (the
    LangGraph judge, the chat endpoint, the API responses).
    """

    __tablename__ = "devices"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    brand = Column(String, nullable=False, index=True)

    # Empirical, official display name (e.g. "Xiaomi Redmi Note 10S")
    model_name = Column(String, nullable=False)

    # Indexed, unique lookup key — see app/core/normalize.py. Enables exact-match
    # canonical resolution without computing regexp_replace() per row per query.
    normalized_name = Column(String, nullable=False, unique=True, index=True)

    release_year = Column(Integer, index=True)
    base_price_usd = Column(Numeric)

    specs_json = Column(JSONB, nullable=False)
    source_urls = Column(JSONB, nullable=False, default=list)

    image_url = Column(String)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    chunks = relationship("DeviceChunk", back_populates="device", cascade="all, delete-orphan")

Index("ix_devices_specs_json_gin", Device.specs_json, postgresql_using="gin")