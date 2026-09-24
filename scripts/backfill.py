"""
Backfill script for generating missing embeddings for existing devices.
Idempotent and safe to run multiple times.
"""
import os
import json
import asyncio
import uuid
import traceback

from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from dotenv import load_dotenv

from app.core.database import async_session
from app.core.embeddings import get_embedding
from app.core.chunking import generate_chunks_from_specs, generate_content_hash, ChunkResult
from app.models.device import Device
from app.models.chunk import DeviceChunk
from app.main import resolve_canonical_name

load_dotenv()

async def process_device(session, device: Device) -> dict:
    """Generate chunks and insert for a single device."""
    chunks = generate_chunks_from_specs(device.model_name, device.specs_json)
    
    if not chunks:
        return {"inserted": 0, "skipped": 0, "failed": 0}

    chunk_dicts = []
    failed = 0
    
    # Process sequentially or with gather. We'll use gather for speed.
    async def prepare_chunk(c: ChunkResult):
        try:
            emb = await get_embedding(c.content)
            return {
                "id": uuid.uuid4(),
                "device_id": device.id,
                "category": c.category,
                "content": c.content,
                "source_url": c.source_url,
                "metadata_json": {},
                "content_hash": c.content_hash,
                "embedding": emb,
            }
        except Exception as e:
            print(f"Error embedding chunk for {device.model_name}: {e}")
            return None

    results = await asyncio.gather(*[prepare_chunk(c) for c in chunks])
    
    for r in results:
        if r is None:
            failed += 1
        else:
            chunk_dicts.append(r)

    inserted = 0
    skipped = 0
    
    if chunk_dicts:
        # PostgreSQL ON CONFLICT DO NOTHING doesn't return the number of inserted rows directly easily
        # but we can do a returning id or just rely on a separate query if needed.
        # For simplicity, we just insert and assume all were inserted if they weren't skipped.
        stmt = pg_insert(DeviceChunk).values(chunk_dicts).on_conflict_do_nothing(
            index_elements=["device_id", "content_hash"]
        ).returning(DeviceChunk.id)
        
        result = await session.execute(stmt)
        inserted_ids = result.scalars().all()
        await session.commit()
        
        inserted = len(inserted_ids)
        skipped = len(chunk_dicts) - inserted

    return {"inserted": inserted, "skipped": skipped, "failed": failed}

async def backfill_sample_reviews(session):
    """Backfill external sample reviews if the device exists."""
    reviews_path = os.path.join("data", "sample_reviews.json")
    if not os.path.exists(reviews_path):
        return {"inserted": 0, "skipped": 0, "failed": 0}
        
    with open(reviews_path, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    inserted = 0
    skipped = 0
    failed = 0
    
    for review in reviews:
        model_name = review["model_name"]
        content = review["chunk_text"]
        source_url = review["source_url"]
        
        canonical = await resolve_canonical_name(model_name)
        if not canonical:
            # Skip if device not found
            continue
            
        device_id = await session.scalar(select(Device.id).where(Device.model_name == canonical))
        if not device_id:
            continue
            
        content_text = f"Phone: {canonical}\nCategory: Review\nSentiment: {content}"
        content_hash = generate_content_hash(content_text)
        
        # Check if already exists to skip embedding
        exists = await session.scalar(
            select(func.count(DeviceChunk.id))
            .where(DeviceChunk.device_id == device_id, DeviceChunk.content_hash == content_hash)
        )
        if exists:
            skipped += 1
            continue
            
        try:
            emb = await get_embedding(content_text)
            stmt = pg_insert(DeviceChunk).values(
                id=uuid.uuid4(),
                device_id=device_id,
                category="Review",
                content=content_text,
                source_url=source_url,
                metadata_json={},
                content_hash=content_hash,
                embedding=emb,
            ).on_conflict_do_nothing(index_elements=["device_id", "content_hash"]).returning(DeviceChunk.id)
            
            res = await session.execute(stmt)
            if res.scalar():
                inserted += 1
            else:
                skipped += 1
            await session.commit()
        except Exception as e:
            failed += 1
            print(f"Error embedding sample review for {model_name}: {e}")

    return {"inserted": inserted, "skipped": skipped, "failed": failed}

async def main():
    print("Starting backfill process...")
    
    async with async_session() as session:
        devices = await session.scalars(select(Device))
        devices = devices.all()
        
    print(f"Found {len(devices)} devices in the database.")
    
    total_inserted = 0
    total_skipped = 0
    total_failed = 0
    
    # Process devices one by one to avoid rate limits
    for device in devices:
        print(f"Processing '{device.model_name}'...")
        async with async_session() as session:
            stats = await process_device(session, device)
            total_inserted += stats["inserted"]
            total_skipped += stats["skipped"]
            total_failed += stats["failed"]
            
    print(f"\nProcessing sample reviews...")
    async with async_session() as session:
        review_stats = await backfill_sample_reviews(session)
        total_inserted += review_stats["inserted"]
        total_skipped += review_stats["skipped"]
        total_failed += review_stats["failed"]

    print("\nBackfill Complete.")
    print(f"Total Chunks Inserted: {total_inserted}")
    print(f"Total Chunks Skipped (already exist): {total_skipped}")
    print(f"Total Chunks Failed: {total_failed}")

if __name__ == "__main__":
    asyncio.run(main())
