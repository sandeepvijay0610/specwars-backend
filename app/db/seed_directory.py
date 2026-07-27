import asyncio
import json
import uuid

from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.core.database import Base, engine, async_session
from app.core.normalize import normalize_name
from app.models.phone_directory import PhoneDirectory


async def create_table():
    """Ensure the phone_directory table exists."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def seed_directory():
    # Step 1: Create table if it doesn't exist
    await create_table()

    # Step 2: Read the phones list JSON from the project root
    with open("phones_list.json", "r", encoding="utf-8-sig") as f:
        phone_names: list[str] = json.load(f)

    if not phone_names:
        print("phones_list.json is empty. Nothing to seed.")
        return

    # Step 3: Bulk UPSERT — insert and skip on conflict (unique model_name)
    async with async_session() as session:
        rows = []
        seen_normalized = set()
        for entry in phone_names:
            if isinstance(entry, dict):
                # New format: {"brand": "Apple", "model": "iPhone 15 Pro"}
                brand = entry.get("brand", "").strip()
                model = entry.get("model", "").strip()
                full_name = f"{brand} {model}".strip() if brand else model
            else:
                # Legacy flat string format: "Apple iPhone 15 Pro"
                full_name = str(entry).strip()

            if full_name:
                norm_name = normalize_name(full_name)
                if norm_name not in seen_normalized:
                    seen_normalized.add(norm_name)
                    rows.append({
                        "id": uuid.uuid4(),
                        "model_name": full_name,
                        "normalized_name": norm_name
                    })

        print(f"Seeding {len(rows)} phones into phone_directory...")

        stmt = pg_insert(PhoneDirectory).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["model_name"])

        await session.execute(stmt)
        await session.commit()

    print(f"Done! {len(rows)} records upserted (duplicates safely ignored).")


if __name__ == "__main__":
    asyncio.run(seed_directory())
