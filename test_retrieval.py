import asyncio
import pprint
from app.core.retrieval import retrieve_relevant_chunks

async def main():
    res = await retrieve_relevant_chunks(['Pixel 9'], 'Which phone has better battery and fast charging?', 3)
    pprint.pprint(res)

if __name__ == "__main__":
    asyncio.run(main())
