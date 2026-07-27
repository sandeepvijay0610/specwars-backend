import uuid

from sqlalchemy import Column, String
from sqlalchemy.dialects.postgresql import UUID

from app.core.database import Base


class PhoneDirectory(Base):
    """Lightweight lookup table of every known phone name, used to power
    autocomplete for phones that haven't been fully ingested into `devices`
    yet. Kept in sync with `devices` at ingestion time (see dynamic_ingest.py).
    """

    __tablename__ = "phone_directory"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False, unique=True, index=True)
    normalized_name = Column(String, nullable=False, unique=True, index=True)