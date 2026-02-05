
import httpx
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(".env")

async def t():
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    print(f"Key: {key[:10]}...{key[-5:]} (len: {len(key)})")
    url = "https://api.siliconflow.cn/v1/user/info"
    headers = {"Authorization": f"Bearer {key}"}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers)
        print(f"Status: {r.status_code}")
        print(f"Body: {r.text}")

if __name__ == "__main__":
    asyncio.run(t())
