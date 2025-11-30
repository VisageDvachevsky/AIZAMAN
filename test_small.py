#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 Тест на малой выборке (10 примеров)
Проверка работы перед полной обработкой
"""

import sys
import pandas as pd

# Импортируем функции из основного скрипта
sys.path.insert(0, '.')
from main_GPT4O_MINI_OPTIMIZED import detoxify_text, TOXIC_WORDS_SET

print("="*80)
print("🧪 ТЕСТ НА 10 ПРИМЕРАХ")
print("="*80)

# Загружаем первые 10 строк
print("\n📥 Загрузка первых 10 примеров...")
df = pd.read_csv("dev_inputs.tsv", sep="\t", nrows=10)
print(f"   Загружено: {len(df)} примеров\n")

# Обрабатываем каждый пример
results = []
for idx, row in df.iterrows():
    toxic = row['tat_toxic']

    print(f"{'='*80}")
    print(f"📝 Пример {idx}")
    print(f"{'='*80}")
    print(f"Токсичный: {toxic}")

    # Определяем токсичные слова в тексте
    toxic_lower = toxic.lower()
    found_toxic = [w for w in TOXIC_WORDS_SET if w in toxic_lower]
    if found_toxic:
        print(f"🔴 Найденная токсичность: {', '.join(found_toxic)}")
    else:
        print(f"⚠️ Явной токсичности не найдено (возможно неявная)")

    # Детоксификация
    print(f"\n⏳ Детоксификация...")
    detoxed = detoxify_text(toxic)

    print(f"✅ Детокс:     {detoxed}")

    # Проверка остаточной токсичности
    detoxed_lower = detoxed.lower()
    remaining = [w for w in TOXIC_WORDS_SET if w in detoxed_lower]
    if remaining:
        print(f"⚠️ ОСТАЛАСЬ ТОКСИЧНОСТЬ: {', '.join(remaining)}")
    else:
        print(f"✓ Токсичность удалена")

    # Проверка длины
    len_change = len(detoxed) - len(toxic)
    len_percent = (len_change / len(toxic) * 100) if len(toxic) > 0 else 0
    print(f"📏 Изменение длины: {len_change:+d} символов ({len_percent:+.1f}%)")

    results.append({
        'ID': idx,
        'toxic': toxic,
        'detoxed': detoxed,
        'found_toxic': found_toxic,
        'remaining': remaining,
        'len_change_pct': len_percent
    })
    print()

# Итоговая статистика
print("="*80)
print("📊 ИТОГОВАЯ СТАТИСТИКА")
print("="*80)

total = len(results)
fully_cleaned = sum(1 for r in results if not r['remaining'])
partially_cleaned = total - fully_cleaned

print(f"\n✅ Полностью очищено: {fully_cleaned}/{total} ({fully_cleaned/total*100:.1f}%)")
if partially_cleaned > 0:
    print(f"⚠️ Частично очищено: {partially_cleaned}/{total}")

avg_len_change = sum(r['len_change_pct'] for r in results) / total
print(f"\n📏 Средний Δ длины: {avg_len_change:+.1f}%")

# Проблемные примеры
if partially_cleaned > 0:
    print(f"\n⚠️ ПРИМЕРЫ С ОСТАТОЧНОЙ ТОКСИЧНОСТЬЮ:")
    for r in results:
        if r['remaining']:
            print(f"   [{r['ID']}] Осталось: {', '.join(r['remaining'])}")
            print(f"       {r['detoxed'][:70]}")

print("\n" + "="*80)
print("✅ ТЕСТ ЗАВЕРШЁН")
print("="*80)
print("\n💡 Если результаты хорошие, запускайте полную обработку:")
print("   .venv/bin/python main_GPT4O_MINI_OPTIMIZED.py")
