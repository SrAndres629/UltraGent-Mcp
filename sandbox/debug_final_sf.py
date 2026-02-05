
import asyncio
import httpx
import os
from dotenv import load_dotenv

load_dotenv(".env")

async def test_final():
    key = os.getenv("SILICONFLOW_API_KEY", "").strip()
    print(f"Testing with Key: {key[:10]}...{key[-5:]}")
    print(f"Key length: {len(key)}")
    
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json"
    }
    
    # Probando con el modelo que aparece en tu captura/manual
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [{"role": "user", "content": "Responder con 'SISTEMA ONLINE'"}],
        "temperature": 0.7
    }
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, headers=headers, timeout=20.0)
            print(f"Status: {resp.status_code}")
            print(f"Headers: {resp.headers.get('request-id', 'no-id')}")
            print(f"Body: {resp.text}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_final())
