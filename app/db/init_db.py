import asyncio
import sys
import asyncpg
from app.core.config import settings
from app.core.database import engine, Base
import app.models.device  # to ensure models are imported
import app.models.chunk
import app.models.comparison_cache
import app.models.phone_directory
from app.db.seed_directory import seed_directory

async def create_extensions():
    conn_str = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgres://")
    conn = await asyncpg.connect(conn_str)
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS btree_gin;")
        print("Extensions created.")
    finally:
        await conn.close()

async def reset_database():
    if "--reset" in sys.argv:
        print("Reset flag detected. Dropping all tables...")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        print("All tables dropped.")

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tables created.")

async def main():
    await create_extensions()
    await reset_database()
    await create_tables()
    
    if "--reset" in sys.argv:
        print("Re-seeding master phone directory...")
        await seed_directory()

if __name__ == "__main__":
    asyncio.run(main())
