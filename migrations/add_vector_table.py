"""
Idempotent migration script to add the device_chunks table.
Safe to run on both local and AWS RDS PostgreSQL.
"""
import asyncio
import asyncpg
from app.core.config import settings
from app.core.database import engine, Base
import app.models.device
import app.models.chunk

async def migrate():
    print("Connecting to database and creating vector extension if missing...")
    conn_str = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
    conn = await asyncpg.connect(conn_str)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
    finally:
        await conn.close()

    print("Creating device_chunks table and indexes if they do not exist...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("Migration complete. device_chunks is ready.")

if __name__ == "__main__":
    asyncio.run(migrate())
