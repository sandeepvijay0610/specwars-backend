import asyncio
import httpx
import time
import pprint

async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        print("Sending first request (Expect long ingestion & RAG generation)...")
        start_time = time.time()
        resp = await client.post(
            "http://127.0.0.1:8000/api/compare",
            json={"phones": ["iPhone 15 Pro", "Samsung Galaxy S24 Ultra"]}
        )
        first_duration = time.time() - start_time
        print(f"Status Code: {resp.status_code}")
        print(f"Time Taken: {first_duration:.2f} seconds")
        if resp.status_code == 200:
            print("Response preview:")
            pprint.pprint(resp.json())
        else:
            print(f"Error payload: {resp.text}")
        print("\n" + "-"*50 + "\n")
        
        print("Sending second request (Expect instant cache hit)...")
        start_time = time.time()
        resp = await client.post(
            "http://127.0.0.1:8000/api/compare",
            json={"phones": ["iPhone 15 Pro", "Samsung Galaxy S24 Ultra"]}
        )
        second_duration = time.time() - start_time
        print(f"Status Code: {resp.status_code}")
        print(f"Time Taken: {second_duration:.2f} seconds")
        print("Cache Test Passed!" if second_duration < 2.0 else "Cache Test Failed!")

if __name__ == "__main__":
    asyncio.run(main())
