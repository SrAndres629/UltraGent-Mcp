
import asyncio
import httpx
import os
import sys
from dotenv import load_dotenv

# Setup
sys.path.append(os.getcwd())
load_dotenv(".env")

async def test_siliconflow():
    api_key = os.getenv("SILICONFLOW_API_KEY", "")
    print(f"Testing SiliconFlow API...")
    print(f"API Key present: {bool(api_key)}")
    print(f"API Key starts with: {api_key[:10]}...")
    
    url = "https://api.siliconflow.cn/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "messages": [{"role": "user", "content": "Hello! Reply with 'OK'."}],
        "stream": False
    }
    
    print("\nSending request to SiliconFlow...")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            print(f"Status Code: {response.status_code}")
            if response.status_code == 200:
                print("SUCCESS!")
                print(f"Response: {response.json()['choices'][0]['message']['content']}")
            else:
                print(f"FAILURE!")
                print(f"Response Body: {response.text}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_siliconflow())
