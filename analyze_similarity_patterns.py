#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Анализ ПАТТЕРНОВ изменений, которые снижают SIM"""

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

print("🔧 Загрузка LaBSE...")
model = SentenceTransformer('sentence-transformers/LaBSE')

df = pd.read_csv('submission.tsv', sep='\t')

print('\n🔍 Анализ паттернов низкого SIM:\n')

# Вычисляем SIM для всех
low_sim_examples = []

for idx, row in df.iterrows():
    orig = str(row['tat_toxic'])
    detox = str(row['tat_detox1'])

    if orig != detox:  # Только измененные
        emb1 = model.encode([orig], convert_to_numpy=True)
        emb2 = model.encode([detox], convert_to_numpy=True)

        sim = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))

        if sim < 0.75:  # Низкий SIM
            len_change = abs(len(detox) - len(orig)) / len(orig) * 100
            low_sim_examples.append((idx, sim, len_change, orig, detox))

print(f'📊 Найдено {len(low_sim_examples)} примеров с SIM < 0.75\n')

# Сортируем по SIM
low_sim_examples.sort(key=lambda x: x[1])

print('🎯 ПАТТЕРНЫ низкого SIM (худшие 10):')
print('   (анализируй ЧТО изменено и ПОЧЕМУ это плохо)\n')

for idx, sim, len_change, orig, detox in low_sim_examples[:10]:
    print(f'[{idx}] SIM={sim:.2f}, ΔLen={len_change:.0f}%')
    print(f'  Orig:  {orig[:70]}')
    print(f'  Detox: {detox[:70]}')

    # Простой анализ типа изменения
    if len_change > 50:
        print(f'  ⚠️ ПАТТЕРН: Слишком много удалено ({len_change:.0f}%)')

    elif '@user' in orig and '@user' not in detox:
        print(f'  ℹ️ ПАТТЕРН: Удален @user')

    # Проверка орфографии / регистра
    if orig.lower() != orig or detox.lower() != detox:
        orig_clean = orig.replace('э', 'е').replace('ә', 'а')
        if orig_clean != orig and detox != orig:
            print(f'  ⚠️ ПАТТЕРН: Возможно изменена орфография')

    print()

print('✅ Готово! Используй эти ПАТТЕРНЫ:')
print('   - НЕ удаляй слишком много текста')
print('   - НЕ меняй орфографию')
print('   - Минимальные изменения = высокий SIM')
