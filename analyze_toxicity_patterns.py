#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Анализ ПАТТЕРНОВ остаточной токсичности"""

import pandas as pd
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter

print("🔧 Загрузка модели токсичности...")
model_name = 'textdetox/xlmr-large-toxicity-classifier-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

df = pd.read_csv('submission.tsv', sep='\t')

print('\n🔍 Анализ паттернов остаточной токсичности:\n')

# Находим ВСЕ примеры где детокс токсичен
toxic_detox = []
word_patterns = Counter()

for idx, row in df.iterrows():
    detox = str(row['tat_detox1'])

    inputs = tokenizer(detox, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        toxic_prob = probs[0][1].item()

    if toxic_prob > 0.3:  # Токсичный
        toxic_detox.append((idx, toxic_prob, detox))
        # Извлекаем все слова
        words = re.findall(r'[а-яәөүңғқһёa-zәөүңғқһ]+', detox.lower())
        word_patterns.update(words)

print(f'📊 Найдено {len(toxic_detox)} токсичных детоксов из {len(df)}')
print(f'   Процент: {len(toxic_detox)/len(df)*100:.1f}%\n')

# Топ слов в токсичных детоксах
print('🎯 Топ-40 слов в ТОКСИЧНЫХ детоксах (паттерны):')
print('   (эти слова часто остаются и модель их считает токсичными)\n')

for word, count in word_patterns.most_common(40):
    if len(word) >= 3:  # Пропускаем предлоги
        print(f'  {word}: {count} раз')

print('\n📋 Примеры токсичных детоксов (первые 10):')
for idx, prob, text in toxic_detox[:10]:
    print(f'  [{idx}] P={prob:.2f}: {text[:80]}')

print("\n✅ Готово! Используй эти ПАТТЕРНЫ для улучшения.")
