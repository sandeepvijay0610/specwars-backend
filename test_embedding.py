import asyncio
from app.core.embeddings import get_embedding

async def main():
    emb = await get_embedding("Hello world")
    print(f"Dimension: {len(emb)}")

if __name__ == "__main__":
    asyncio.run(main())
