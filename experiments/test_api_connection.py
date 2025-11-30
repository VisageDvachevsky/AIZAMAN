#!/usr/bin/env python3
"""Quick API connection test"""

from openai import OpenAI

API_KEY = "sk-C4Ju9Yy2-EKOf6SHs-jBPA"
BASE_URL = "https://api.artemox.com/v1"

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print("Testing API connection...")
try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'Hello'"}],
        max_tokens=10
    )
    print(f"✅ API works! Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ API error: {e}")
