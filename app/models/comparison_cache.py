import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, String, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB

from app.core.database import Base


class ComparisonCache(Base):
    """Caches a finished LangGraph verdict for a given unordered pair (or set)
    of canonical devices, keyed by `pair_key` (sorted, normalized names joined
    with '_'). A cache hit skips the LangGraph run entirely.
    """

    __tablename__ = "comparison_cache"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    pair_key = Column(String, nullable=False, unique=True, index=True)
    verdict_json = Column(JSONB, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))