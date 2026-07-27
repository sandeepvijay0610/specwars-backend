import os
import asyncio
import uuid
from typing import List, Tuple, Optional
from dotenv import load_dotenv

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from langchain_openai import AzureChatOpenAI
from tavily import TavilyClient

from app.core.database import async_session
from app.core.normalize import normalize_name
from app.models.device import Device
from app.models.phone_directory import PhoneDirectory

load_dotenv()

# ---------------------------------------------------------------------------
# 1. Scrubber LLM
# ---------------------------------------------------------------------------
# Small/cheap model used exactly once per ingestion to turn messy web text
# into strict structured data. Deployment name is env-configured (not
# hardcoded) so it can be swapped without touching code. This is a separate
# deployment from the judge model used in app/agent/graph.py.
scrubber_llm = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_SCRUBBER_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    temperature=0.0,
)


# ---------------------------------------------------------------------------
# 2. Web Search (Tavily)
# ---------------------------------------------------------------------------
def search_web_for_phone(phone_model: str) -> Tuple[str, Optional[str], List[str]]:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    print(f"[Tavily] Searching for '{phone_model}'...")
    response = tavily_client.search(
        query=f"{phone_model} full specifications and camera review",
        search_depth="advanced",
        include_raw_content=False,
        include_images=True,
        include_domains=["gsmarena.com", "91mobiles.com"] # Restricts search to trusted spec sheets
    )

    combined_content = []
    source_urls = []
    for result in response.get("results", []):
        combined_content.append(result.get("content", ""))
        source_urls.append(result.get("url", ""))

    images = response.get("images", [])
    first_image_url = images[0] if images else None

    return "\n\n".join(combined_content), first_image_url, source_urls


# ---------------------------------------------------------------------------
# 3. Strict Structured Extraction — "Token-is-Gold"
# ---------------------------------------------------------------------------
class PhoneDataSchema(BaseModel):
    brand: str = Field(description="Official brand name, e.g. 'Xiaomi'")
    canonical_name: str = Field(
        description="The full, official, properly capitalized model name, "
        "e.g. 'Xiaomi Redmi Note 10S' (not a shorthand or nickname)."
    )
    release_year: int = Field(description="Release year")
    base_price_usd: int = Field(description="Launch price in USD; use 0 if genuinely unknown")

    processor: str = Field(description="Chipset / SoC, e.g. 'Snapdragon 8 Gen 3'")
    display: str = Field(description="Panel type, size, resolution, and refresh rate")
    ram: str = Field(description="RAM configuration(s) offered")
    storage: str = Field(description="Storage configuration(s) and standard, e.g. 'UFS 4.0'")
    battery: str = Field(description="Battery capacity in mAh and charging speed in W")
    camera: str = Field(description="Main + auxiliary camera resolutions and key sensor details")
    build: str = Field(description="Build material, IP rating, weight")

    review_sentiments: List[str] = Field(
        description="Exactly 3 short, factual review sentiments about camera, battery, "
        "or performance, grounded strictly in the source text.",
        min_length=3,
        max_length=3,
    )


async def extract_phone_data(raw_text: str) -> PhoneDataSchema:
    """Passes raw web text through the scrubber model exactly once."""
    structured_llm = scrubber_llm.with_structured_output(PhoneDataSchema, method="function_calling")

    prompt = f"""You are a strict data-extraction assistant. Parse the following messy web
search results into clean structured data. Do not invent values — if a base price is not
found, estimate conservatively based on the tier described, or use 0.

Raw Web Data:
{raw_text}
"""
    print("[Scrubber] Extracting structured specs (single pass)...")
    return await structured_llm.ainvoke(prompt)


# ---------------------------------------------------------------------------
# 4. Ingestion
# ---------------------------------------------------------------------------
async def ingest_new_phone(phone_model: str) -> str:
    """Ensures `phone_model` exists as a canonical Device row.

    Returns the canonical (empirical) model name — whether it was already
    cached or freshly ingested — so callers never need a second lookup.
    """
    normalized = normalize_name(phone_model)

    async with async_session() as session:
        existing = await session.scalar(
            select(Device.model_name).where(Device.normalized_name == normalized)
        )
        if existing:
            print(f"[Ingest] Cache hit for '{phone_model}' -> '{existing}'")
            return existing

    print(f"[Ingest] Cache miss for '{phone_model}'. Starting structured ETL...")
    raw_text, image_url, source_urls = await asyncio.to_thread(search_web_for_phone, phone_model)
    extracted = await extract_phone_data(raw_text)

    canonical_normalized = normalize_name(extracted.canonical_name)
    specs_json = {
        "processor": extracted.processor,
        "display": extracted.display,
        "ram": extracted.ram,
        "storage": extracted.storage,
        "battery": extracted.battery,
        "camera": extracted.camera,
        "build": extracted.build,
        "review_sentiments": extracted.review_sentiments,
    }

    async with async_session() as session:
        # ON CONFLICT guards against a race where two requests ingest the same
        # phone concurrently, or the LLM resolves to an already-known canonical name.
        insert_stmt = pg_insert(Device).values(
            id=uuid.uuid4(),
            brand=extracted.brand,
            model_name=extracted.canonical_name,
            normalized_name=canonical_normalized,
            release_year=extracted.release_year,
            base_price_usd=extracted.base_price_usd,
            specs_json=specs_json,
            source_urls=source_urls,
            image_url=image_url,
        ).on_conflict_do_nothing(index_elements=["normalized_name"])
        await session.execute(insert_stmt)

        await session.execute(
            pg_insert(PhoneDirectory)
            .values(id=uuid.uuid4(), model_name=extracted.canonical_name, normalized_name=canonical_normalized)
            .on_conflict_do_nothing(index_elements=["normalized_name"])
        )

        await session.commit()

        canonical = await session.scalar(
            select(Device.model_name).where(Device.normalized_name == canonical_normalized)
        )

    print(f"[Ingest] '{phone_model}' resolved to '{canonical}'.")
    return canonical


# ---------------------------------------------------------------------------
# 5. Manual Test Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(ingest_new_phone("Google Pixel 9 Pro"))