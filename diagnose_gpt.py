#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Диагностика: почему GPT пропускает токсичные слова
"""

import os
from openai import OpenAI

API_KEY = os.getenv("OPENAI_API_KEY", "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")
MODEL_NAME = "gpt-4o-2024-11-20"
client = OpenAI(api_key=API_KEY)

# Проблемные примеры
test_cases = [
    "@user, син кил давай, кутак сырлама!",
    "Заипали, нэрсэ булэк итим, нишлим диеп!",
    "маржага син",
    "Эй, хули комментны удалил?",
    "ни хрена да булышмады",
]

def test_simple_prompt(text):
    """Тест с ОЧЕНЬ простым промптом"""
    prompt = f"""Удали маты и грубые слова из татарского текста. Сохрани орфографию.

Текст: {text}

Результат (только текст, без пояснений):"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )

    return resp.choices[0].message.content.strip()


def test_explicit_prompt(text):
    """Тест с явным перечислением"""
    prompt = f"""Твоя задача: убрать ВСЕ токсичные слова.

УДАЛИ эти слова если встретишь:
- кутак
- маржага
- Заипали (это мат)
- хули
- хрена
- @user

Текст: {text}

Результат:"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=150
    )

    return resp.choices[0].message.content.strip()


def test_current_prompt(text):
    """Тест с текущим промптом из main_BALANCED"""
    prompt = f"""Детоксифицируй татарский текст. Удали ТОЛЬКО явные маты и грубую лексику.

🎯 КРИТИЧНО - СОХРАНИ ОРФОГРАФИЮ:
- НЕ меняй татарские буквы: э, ә, җ, ө, ү, ң, һ, ғ, қ

📋 ЧТО УДАЛЯТЬ:
✗ Явные маты: бля*, хуй*, жоп*, кут*, ебать*, сука*, пизд*, хрен*, хули*
✗ Грубые оскорбления: ахмак*, ангыра*, дебил*, идиот*

ТЕКСТ: {text}

Результат:"""

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=150
    )

    return resp.choices[0].message.content.strip()


print("="*70)
print("🔬 ДИАГНОСТИКА GPT")
print("="*70)

for i, text in enumerate(test_cases, 1):
    print(f"\n{'='*70}")
    print(f"[{i}] Оригинал: {text}")
    print(f"{'='*70}")

    simple = test_simple_prompt(text)
    print(f"  Простой промпт:  {simple}")

    explicit = test_explicit_prompt(text)
    print(f"  Явный список:    {explicit}")

    current = test_current_prompt(text)
    print(f"  Текущий промпт:  {current}")

    # Проверка
    issues = []
    if "кутак" in simple.lower():
        issues.append("кутак остался (простой)")
    if "заипали" in explicit.lower():
        issues.append("заипали остался (явный)")
    if "маржага" in current.lower():
        issues.append("маржага остался (текущий)")
    if "@user" in explicit:
        issues.append("@user остался")
    if "хули" in current.lower():
        issues.append("хули остался")
    if "хрен" in current.lower():
        issues.append("хрена остался")

    if issues:
        print(f"  ⚠️  ПРОБЛЕМЫ: {', '.join(issues)}")
    else:
        print(f"  ✅ Все варианты удалили токсичность")

print("\n" + "="*70)
print("📊 ВЫВОДЫ:")
print("="*70)
print("Посмотри какой промпт работает лучше всего")
print("Если все плохо - значит GPT-4o не знает татарскую токсичность")
print("="*70)
