import asyncio
import httpx
import pytest
import time
from sqlalchemy.future import select
from sqlalchemy import func
from app.core.database import async_session
from app.models.device import Device
from app.models.chunk import DeviceChunk

@pytest.mark.asyncio
async def test_e2e_comparison_flow():
    async with httpx.AsyncClient(timeout=120.0) as client:
        # 1. Test full retrieval and graph execution
        print("Sending first request (Expect long ingestion & RAG generation)...")
        start_time = time.time()
        resp = await client.post(
            "http://127.0.0.1:8000/api/compare",
            json={"phones": ["iPhone 15 Pro", "Samsung Galaxy S24 Ultra"]}
        )
        first_duration = time.time() - start_time
        
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}. Error: {resp.text}"
        data = resp.json()
        assert "overall_winner" in data
        assert "category_winners" in data
        assert "battery" in data["category_winners"]
        
        # 1.5. Prove vectors were used by checking database directly
        async with async_session() as session:
            # Check if devices were created
            devices_stmt = select(Device).where(Device.model_name.in_(["Apple iPhone 15 Pro", "Samsung Galaxy S24 Ultra"]))
            devices_result = await session.execute(devices_stmt)
            db_devices = devices_result.scalars().all()
            assert len(db_devices) == 2, "Expected 2 devices in the database"
            device_ids = [d.id for d in db_devices]
            
            # Check if vectors were inserted
            chunks_stmt = select(func.count(DeviceChunk.id)).where(DeviceChunk.device_id.in_(device_ids))
            chunks_result = await session.execute(chunks_stmt)
            chunk_count = chunks_result.scalar()
            assert chunk_count > 0, "No vector chunks found! Embeddings failed or were bypassed."
            
        assert "citations" in data
        assert len(data["citations"]) > 0, "Expected citations from retrieved chunks"
        
        # 2. Test Cache Hit
        print("Sending second request (Expect instant cache hit)...")
        start_time = time.time()
        resp2 = await client.post(
            "http://127.0.0.1:8000/api/compare",
            json={"phones": ["iPhone 15 Pro", "Samsung Galaxy S24 Ultra"]}
        )
        second_duration = time.time() - start_time
        
        assert resp2.status_code == 200
        assert second_duration < 2.0, "Cache hit should be extremely fast, skipping LLM evaluation"
        
        data2 = resp2.json()
        assert data == data2, "Cached response should perfectly match original response"

@pytest.mark.asyncio
async def test_api_chat():
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "http://127.0.0.1:8000/api/chat",
            json={"phones": ["iPhone 15 Pro", "Samsung Galaxy S24 Ultra"], "question": "Which one has a bigger battery?"}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 10
