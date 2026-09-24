import os
import json
import asyncio
from typing import TypedDict, List, Dict, Any

from dotenv import load_dotenv
from sqlalchemy import text
from pydantic import BaseModel, Field

from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END

from app.core.database import async_session
from app.core.retrieval import retrieve_relevant_chunks

load_dotenv()


# ---------------------------------------------------------------------------
# 1. State Schema
# ---------------------------------------------------------------------------
class GraphState(TypedDict):
    user_query: str
    phones_list: List[str]          # already-canonical model names (resolved upstream in main.py)
    device_specs: Dict[str, Any]    # {model_name: {specs, base_price_usd, source_urls}}
    retrieved_chunks: List[Dict[str, Any]]
    final_response: Dict[str, Any]


# ---------------------------------------------------------------------------
# 2. Retrieval Node — pulls ONLY structured specs_json, never raw text chunks
# ---------------------------------------------------------------------------
async def retrieve_context(state: GraphState):
    phones_list = state.get("phones_list", [])
    device_specs: Dict[str, Any] = {}

    async with async_session() as session:
        sql = text("""
            SELECT model_name, base_price_usd, specs_json, source_urls
            FROM devices
            WHERE model_name = ANY(:phones_list);
        """)
        result = await session.execute(sql, {"phones_list": phones_list})
        rows = result.fetchall()

        for row in rows:
            device_specs[row.model_name] = {
                "base_price_usd": float(row.base_price_usd) if row.base_price_usd else None,
                "specs": row.specs_json,
                "source_urls": row.source_urls or [],
            }

    # Fetch relevant chunks
    chunks = await retrieve_relevant_chunks(phones_list, state["user_query"], top_k_per_phone=4)

    return {"device_specs": device_specs, "retrieved_chunks": chunks}


# ---------------------------------------------------------------------------
# 3. Structured Verdict Schema — Strict Object Structure (No Lazy Ties)
# ---------------------------------------------------------------------------
class CategoryVerdict(BaseModel):
    winner: str = Field(
        description="Exact model name of the definitive winner for this category. "
        "Use 'Tie' ONLY if the underlying spec values are literally identical "
        "(e.g. both phones use the exact same chipset)."
    )
    justification: str = Field(
        description="1-2 sentences citing the concrete numerical or spec difference that "
        "decided this category (e.g. exact mAh figures, chipset generation, refresh rate "
        "in Hz, storage standard)."
    )


class CategoryWinners(BaseModel):
    performance: CategoryVerdict
    camera: CategoryVerdict
    display: CategoryVerdict
    battery: CategoryVerdict
    build: CategoryVerdict


class SpecWarsVerdict(BaseModel):
    overall_winner: str = Field(
        description="Exact model name of the phone that wins the most categories. "
        "'Tie' is only valid if category wins are split evenly AND the hardware is "
        "identical across the board."
    )
    overall_summary: str = Field(description="1-2 sentence bottom-line verdict.")
    category_winners: CategoryWinners


# ---------------------------------------------------------------------------
# 4. Generation Node
# ---------------------------------------------------------------------------
async def generate_response(state: GraphState):
    user_query = state["user_query"]
    device_specs = state.get("device_specs", {})
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    chunks_text = "\n\n".join(
        [f"[Phone: {c['phone_name']} | Category: {c['category']} | Source: {c['source_url'] or 'DB'}]\n{c['content']}" for c in retrieved_chunks]
    )

    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_LLM_DEPLOYMENT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        temperature=0.1,
    )
    structured_llm = llm.with_structured_output(SpecWarsVerdict, method="function_calling")

    system_prompt = f"""You are an empirical hardware judge for smartphone comparisons. You decide
every category on numerical and spec superiority — you are not a diplomat.

Structured Specs (from database, already verified — do not add outside facts):
{json.dumps(device_specs, indent=2)}

Relevant Context Chunks (Web Search & Reviews):
{chunks_text}

STRICT RULES:
1. Ground every claim ONLY in the specs and context chunks above. No outside knowledge, no guessing.
2. Ban Lazy Ties: you may declare a category "Tie" ONLY if the compared spec values are
   literally identical strings (e.g. both list "Snapdragon 8 Gen 3"). A close call is NOT
   a tie — pick the phone with the higher number, newer standard, or higher tier.
3. Every justification must reference the actual figures from the specs above
   (e.g. "5000mAh vs 4500mAh", "UFS 4.0 vs UFS 3.1", "120Hz vs 90Hz").
4. Fill ALL 5 categories: performance, camera, display, battery, build.
5. Keep every summary/justification to 1-2 sentences.
6. Synthesize insights from the Context Chunks to provide deeper evaluations (e.g. thermal throttling, photo quality), using the provided Source URLs to back them up if requested.
"""

    messages = [
        ("system", system_prompt),
        ("user", user_query),
    ]

    response: SpecWarsVerdict = await structured_llm.ainvoke(messages)
    payload = response.model_dump()

    # Citations are derived deterministically from the DB, never from the LLM,
    # so a hallucinated source URL is structurally impossible.
    all_sources: List[str] = []
    for info in device_specs.values():
        for url in info.get("source_urls", []):
            if url and url not in all_sources:
                all_sources.append(url)
                
    for chunk in retrieved_chunks:
        url = chunk.get("source_url")
        if url and url not in all_sources:
            all_sources.append(url)

    payload["citations"] = all_sources

    return {"final_response": payload}


# ---------------------------------------------------------------------------
# 5. Graph Builder
# ---------------------------------------------------------------------------
builder = StateGraph(GraphState)
builder.add_node("retrieve_context", retrieve_context)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "retrieve_context")
builder.add_edge("retrieve_context", "generate_response")
builder.add_edge("generate_response", END)

specwars_agent = builder.compile()


# ---------------------------------------------------------------------------
# 6. Manual Test Execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    async def main():
        initial_state = {
            "user_query": "Compare the camera and performance of the Google Pixel 9 Pro and the OnePlus Nord CE5",
            "phones_list": ["Google Pixel 9 Pro", "OnePlus Nord CE5"],
        }
        print(f"Running agent for query: '{initial_state['user_query']}'...")
        result = await specwars_agent.ainvoke(initial_state)
        print("\n--- Final JSON Payload ---")
        print(json.dumps(result["final_response"], indent=2))

    asyncio.run(main())