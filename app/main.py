import os
import json
from typing import Dict, List, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import select, func
from langchain_openai import AzureChatOpenAI

from app.core.database import async_session
from app.core.normalize import normalize_name
from app.models.device import Device
from app.models.comparison_cache import ComparisonCache
from app.models.phone_directory import PhoneDirectory
from app.db.dynamic_ingest import ingest_new_phone
from app.agent.graph import specwars_agent

load_dotenv()

app = FastAPI(title="SpecWars API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CompareRequest(BaseModel):
    phones: List[str]


class ChatRequest(BaseModel):
    question: str
    phones: List[str]


# ---------------------------------------------------------------------------
# Canonical Name Resolution
# ---------------------------------------------------------------------------
async def resolve_canonical_name(phone: str) -> Optional[str]:
    """Resolves a raw user-typed phone name to an already-ingested canonical
    Device name, or None if it isn't in the DB yet.

    Tries an exact normalized match first (cheap, indexed), then falls back
    to a fuzzy ILIKE match so spacing/casing/typos don't force a needless
    re-ingestion.
    """
    normalized = normalize_name(phone)

    async with async_session() as session:
        exact = await session.scalar(
            select(Device.model_name).where(Device.normalized_name == normalized)
        )
        if exact:
            return exact

        fuzzy = await session.scalar(
            select(Device.model_name)
            .where(func.replace(Device.model_name, " ", "").ilike(f"%{phone.replace(' ', '')}%"))
            .limit(1)
        )
        return fuzzy


async def ensure_phone_cached(phone: str) -> str:
    """Resolves `phone` to a canonical name, triggering dynamic ingestion if
    it isn't in the DB yet. Always returns a canonical name."""
    canonical = await resolve_canonical_name(phone)
    if canonical:
        print(f"[Pre-flight] Cache hit for '{phone}' -> '{canonical}'")
        return canonical

    print(f"[Pre-flight] '{phone}' not found — triggering dynamic ingestion...")
    canonical = await ingest_new_phone(phone)
    print(f"[Pre-flight] '{phone}' ingested as '{canonical}'.")
    return canonical


# ---------------------------------------------------------------------------
# GET /api/autocomplete
# ---------------------------------------------------------------------------
@app.get("/api/autocomplete")
async def autocomplete(q: str = Query(..., min_length=1)):
    cached_results = []
    cached_normalized = set()

    async with async_session() as session:
        stmt = select(Device).where(Device.model_name.ilike(f"%{q}%")).limit(10)
        result = await session.execute(stmt)
        devices = result.scalars().all()

        for d in devices:
            cached_results.append({
                "model_name": d.model_name,
                "cached": True,
                "image_url": d.image_url,
            })
            cached_normalized.add(d.normalized_name)

        remaining = 10 - len(cached_results)
        uncached_results = []
        if remaining > 0:
            dir_stmt = (
                select(PhoneDirectory)
                .where(PhoneDirectory.model_name.ilike(f"%{q}%"))
                .limit(remaining * 3)
            )
            dir_result = await session.execute(dir_stmt)
            directory_phones = dir_result.scalars().all()

            for p in directory_phones:
                if p.normalized_name not in cached_normalized:
                    uncached_results.append({
                        "model_name": p.model_name,
                        "cached": False,
                        "image_url": None,
                    })
                if len(uncached_results) >= remaining:
                    break

    return cached_results + uncached_results


# ---------------------------------------------------------------------------
# GET /api/device/{name} — Dynamically resolves shorthand/uncached queries
# ---------------------------------------------------------------------------
@app.get("/api/device/{name}")
async def get_device(name: str):
    canonical = await ensure_phone_cached(name)

    async with async_session() as session:
        device = await session.scalar(
            select(Device).where(Device.model_name == canonical)
        )

    if not device:
        raise HTTPException(status_code=404, detail="Device not found")

    return {
        "model_name": device.model_name,
        "brand": device.brand,
        "release_year": device.release_year,
        "base_price_usd": float(device.base_price_usd) if device.base_price_usd else None,
        "specs": device.specs_json,
        "image_url": device.image_url,
    }


# ---------------------------------------------------------------------------
# POST /api/compare
# ---------------------------------------------------------------------------
@app.post("/api/compare")
async def compare_phones(request: CompareRequest):
    # Pre-flight: canonicalize + ingest any missing phones
    canonical_names: Dict[str, str] = {}
    canonical_list: List[str] = []

    for phone in request.phones:
        canonical = await ensure_phone_cached(phone)
        canonical_names[phone] = canonical
        canonical_list.append(canonical)

    pair_key = "_".join(sorted(normalize_name(n) for n in canonical_list))

    # Cache check
    async with async_session() as session:
        cached = await session.scalar(
            select(ComparisonCache).where(ComparisonCache.pair_key == pair_key)
        )
        if cached:
            payload = dict(cached.verdict_json)
            payload["canonical_names"] = canonical_names
            return payload

    # Cache miss — run the LangGraph judge
    phones_str = " vs ".join(canonical_list)
    query = f"Compare the camera, display, battery, performance, and build of: {phones_str}"
    initial_state = {"user_query": query, "phones_list": canonical_list}

    agent_result = await specwars_agent.ainvoke(initial_state)
    payload = agent_result.get("final_response", {})

    async with async_session() as session:
        session.add(ComparisonCache(pair_key=pair_key, verdict_json=payload))
        await session.commit()

    payload["canonical_names"] = canonical_names
    return payload


# ---------------------------------------------------------------------------
# POST /api/chat — grounded strictly in structured specs_json, no vector search
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def mini_chat(request: ChatRequest):
    canonical_names = []
    for phone in request.phones:
        canonical = await resolve_canonical_name(phone)
        if canonical:
            canonical_names.append(canonical)

    async with async_session() as session:
        stmt = select(Device.model_name, Device.specs_json).where(
            Device.model_name.in_(canonical_names)
        )
        result = await session.execute(stmt)
        rows = result.all()

    context_text = "\n\n".join(
        f"[{model_name}] {json.dumps(specs_json)}" for model_name, specs_json in rows
    ) or "No structured data found for these devices."

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.0,
    )

    messages = [
        ("system", f"""You are a concise smartphone assistant. Answer the user's question in
1-2 sentences ONLY. Use exclusively the structured specs below. If the answer isn't in the
data, say "I don't have that information."

Structured Specs:
{context_text}"""),
        ("user", request.question),
    ]

    response = await llm.ainvoke(messages)
    return {"answer": response.content}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)