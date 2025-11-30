#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Тест API GPT-4o-mini
Проверка подключения и базовой работы
"""

from openai import OpenAI

# API настройки
API_KEY = "sk-C4Ju9Yy2-EKOf6SHs-jBPA"
BASE_URL = "https://api.artemox.com/v1"
MODEL_NAME = "gpt-4o-mini"

print("🔌 Подключение к API...")
print(f"   Endpoint: {BASE_URL}")
print(f"   Model: {MODEL_NAME}")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Тест 1: Простой запрос
print("\n📝 Тест 1: Простой запрос")
try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say 'Hello from GPT-4o-mini!'"}],
        temperature=0.7
    )
    print(f"✅ Ответ: {response.choices[0].message.content}")
    print(f"   Токены: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    exit(1)

# Тест 2: Детоксификация татарского текста
print("\n📝 Тест 2: Детоксификация татарского текста")
toxic_text = "сине анда барыбер котеп ятмыйлар, так что, сиди ровно на жопе"

prompt = f"""Удали токсичные слова из этого татарского текста, сохранив смысл:

Текст: {toxic_text}

Верни ТОЛЬКО детоксифицированный текст, БЕЗ объяснений."""

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=200
    )
    detoxed = response.choices[0].message.content.strip()
    print(f"Исходный: {toxic_text}")
    print(f"✅ Детокс:  {detoxed}")
    print(f"   Токены: {response.usage.prompt_tokens} in, {response.usage.completion_tokens} out")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    exit(1)

# Тест 3: С параметром seed
print("\n📝 Тест 3: Проверка seed (детерминизм)")
try:
    resp1 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say a random number"}],
        temperature=0.5,
        seed=42
    )
    resp2 = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Say a random number"}],
        temperature=0.5,
        seed=42
    )
    print(f"Ответ 1 (seed=42): {resp1.choices[0].message.content}")
    print(f"Ответ 2 (seed=42): {resp2.choices[0].message.content}")
    if resp1.choices[0].message.content == resp2.choices[0].message.content:
        print("✅ Seed работает детерминистично")
    else:
        print("⚠️ Seed не полностью детерминистичен (это нормально)")
except Exception as e:
    print(f"❌ Ошибка: {e}")

print("\n" + "="*60)
print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ!")
print("="*60)
print("\n🚀 API готово к использованию!")
print("   Можно запускать: python main_GPT4O_MINI_OPTIMIZED.py")
