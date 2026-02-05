
import asyncio
import sys
import os
import logging
from dotenv import load_dotenv

# Setup paths
sys.path.append(os.getcwd())
load_dotenv(".env")

from router import GroqProvider

async def test_groq():
    print("Initializing Groq Provider...")
    provider = GroqProvider()
    print(f"Provider: {provider.name}")
    print(f"Model: {provider.model}")
    print(f"API Key present: {bool(provider.api_key)}")
    
    if not provider.api_key:
        print("ERROR: No API Key found")
        return

    messages = [{"role": "user", "content": "Hello! Reply with 'OK'."}]
    
    print("\nSending request...")
    try:
        content, tokens = await provider.complete(messages)
        print("\nSUCCESS!")
        print(f"Response: {content}")
        print(f"Tokens: {tokens}")
    except Exception as e:
        print(f"\nFAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_groq())
