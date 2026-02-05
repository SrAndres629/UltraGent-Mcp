
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

MODELS_TO_TEST = [
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-2.0-pro-exp"
]

print("--- Testing Top Tier Models ---")
for m in MODELS_TO_TEST:
    print(f"Testing {m}...")
    try:
        model = genai.GenerativeModel(m)
        response = model.generate_content("Ping")
        print(f"✅ SUCCESS: {m}")
    except Exception as e:
        print(f"❌ FAIL: {m} ({e})")
