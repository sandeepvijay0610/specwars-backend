from typing import List, Dict, Any
from sqlalchemy import select
from app.core.database import async_session
from app.core.embeddings import get_embedding
from app.models.device import Device
from app.models.chunk import DeviceChunk
async def retrieve_relevant_chunks(phone_names: List[str], query: str, top_k_per_phone: int = 3) -> List[Dict[str, Any]]:
    from app.main import resolve_canonical_name
    """
    Retrieves the most semantically relevant chunks for a given query,
    balanced evenly across the specified phone names.
    """
    # 1. Resolve names to Device IDs
    device_map = {} # canonical_name -> device_id
    
    async with async_session() as session:
        for name in phone_names:
            canonical = await resolve_canonical_name(name)
            if canonical:
                device_id = await session.scalar(select(Device.id).where(Device.model_name == canonical))
                if device_id:
                    device_map[canonical] = device_id
                    
    if not device_map:
        return []

    # 2. Embed the query
    query_embedding = await get_embedding(query)
    
    results = []
    
    # 3. Retrieve balanced chunks per device
    async with async_session() as session:
        for canonical_name, device_id in device_map.items():
            # Use pgvector cosine distance: embedding.cosine_distance(query_embedding)
            stmt = (
                select(DeviceChunk, DeviceChunk.embedding.cosine_distance(query_embedding).label("distance"))
                .where(DeviceChunk.device_id == device_id)
                .order_by(DeviceChunk.embedding.cosine_distance(query_embedding))
                .limit(top_k_per_phone)
            )
            
            rows = await session.execute(stmt)
            for chunk, distance in rows:
                results.append({
                    "phone_name": canonical_name,
                    "category": chunk.category,
                    "content": chunk.content,
                    "source_url": chunk.source_url,
                    "distance": float(distance)
                })
                
    # Sort the combined results by distance (closest first)
    results.sort(key=lambda x: x["distance"])
    
    return results
