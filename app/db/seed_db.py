import asyncio
import json
from app.core.database import async_session
from app.core.normalize import normalize_name
from app.models.device import Device

async def seed_data():
    with open("seed_data.json", "r") as f:
        data = json.load(f)
    
    async with async_session() as session:
        for item in data:
            if "specs" in item:
                item["specs_json"] = item.pop("specs")
            item["normalized_name"] = normalize_name(item["model_name"])
            device = Device(**item)
            session.add(device)
        await session.commit()
    print("Seeded data successfully.")

if __name__ == "__main__":
    asyncio.run(seed_data())
