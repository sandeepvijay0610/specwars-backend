import httpx
import asyncio

async def test():
    async with httpx.AsyncClient(timeout=120.0) as client:
        print('Testing Uncached Comparison (OnePlus 12 vs Sony Xperia 1 VI)...')
        resp = await client.post('http://127.0.0.1:8000/api/compare', json={'phones': ['OnePlus 12', 'Sony Xperia 1 VI']})
        print(f'Comparison Status: {resp.status_code}')
        
        print('\nTesting Custom Chat Question...')
        resp2 = await client.post('http://127.0.0.1:8000/api/chat', json={'phones': ['Samsung Galaxy S24 Ultra', 'Apple iPhone 15 Pro'], 'question': 'Which phone has better zoom capabilities?'})
        print(f'Chat Status: {resp2.status_code}')
        print(resp2.json())

asyncio.run(test())
