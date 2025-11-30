#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 ФИНАЛЬНОЕ РЕШЕНИЕ - Баланс STA × SIM × FL

СТРАТЕГИЯ:
1. Rule-based удаление матов (90% случаев)
2. GPT ТОЛЬКО для обрубков с ЖЕСТКИМИ ограничениями
3. Приоритет: SIM > STA (лучше оставить немного токсичности но высокий SIM)

КЛЮЧЕВОЕ:
- НЕ менять орфографию (э, ә, җ остаются как есть!)
- НЕ переписывать предложения
- МИНИМУМ изменений
"""

import re
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
from functools import lru_cache

API_KEY = os.getenv("OPENAI_API_KEY", "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")
MODEL_NAME = "gpt-4o-2024-11-20"
client = OpenAI(api_key=API_KEY)

INPUT_FILE = "dev_inputs.tsv"
OUTPUT_FILE = "submission.tsv"

total_api_calls = 0

# ============================================
# РАСШИРЕННЫЙ СЛОВАРЬ ТОКСИЧНОСТИ
# ============================================

TOXIC_PATTERNS = [
    # Русские маты
    (r'\b[бБ]ля[дяеюиьть]?\w*', ''),
    (r'\b[бБ]л[еэи][нт]\w*', ''),
    (r'\b[хХ]уй\w*', ''),
    (r'\b[хХ][уУ][йЙиИеЕяЯ]\w*', ''),
    (r'\b[хХ]ер\w*', ''),
    (r'\b[жЖ]оп\w*', ''),
    (r'\b[еЕ]б\w+', ''),
    (r'\b[зЗ]аеб\w+', ''),
    (r'\b[сС]ука\w*', ''),
    (r'\b[пП]изд\w+', ''),
    (r'\b[нН]ахуй\w*', ''),

    # Татарские маты
    (r'\b[кК][уүУҮ][тТ][а-яәөүңғқһёъьA-ZӘӨҮҢҒҚҺ.\-]*', ''),  # кут*
    (r'\b[аА][хХ]мак\w*', ''),
    (r'\b[аА]нгыра\w*', ''),
    (r'\b[тТ]инт[әэ]к\w*', ''),
    (r'\b[хХ]айван\w*', ''),
    (r'\b[чЧ]учка\w*', ''),
    (r'\b[дД]урак\w*', ''),
    (r'\b[дД]ебил\w*', ''),
    (r'\b[иИ]диот\w*', ''),
    (r'\b[сС]волочь\w*', ''),
    (r'\b[мМ]ар[жҗ]а\w*', ''),
    (r'\b[тТ]иле[лк]?\w*', ''),
    (r'\b[сС]осо[пм]\w*', ''),
    (r'\b[зЗ]аипа\w*', ''),
    (r'\b[зЗ]айп\w*', ''),
    (r'\b[аА]хуел\w*', ''),
    (r'\b[тТ]вар[ьъ]?\w*', ''),
    (r'\b[долД][оО][лЛ]ба[йя]\w*', ''),  # долбаящер и т.д.
]

# Контекстные фразы
PHRASES = [
    (r'на\s+жоп\w*', ''),
    (r'в\s+жоп\w*', ''),
]


def rule_based_detox(text):
    """
    ГЛАВНАЯ детоксификация - rule-based

    КРИТИЧНО: ТОЛЬКО удаляем маты, НЕ меняем орфографию!
    """
    if not isinstance(text, str) or not text.strip():
        return text

    result = text

    # @user - можем удалить безопасно
    result = re.sub(r'@\w+[,\s]*', '', result)

    # Фразы
    for pattern, repl in PHRASES:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)

    # Маты
    for pattern, repl in TOXIC_PATTERNS:
        result = re.sub(pattern, repl, result, flags=re.IGNORECASE)

    # Очистка пробелов (ОСТОРОЖНО - сохраняем пунктуацию)
    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\s+([,!?.;:])', r'\1', result)
    result = re.sub(r'([,!?.;:])\s+([,!?.;:])', r'\1\2', result)  # Двойная пунктуация
    result = result.strip()

    # Валидация: не слишком короткий
    if len(result) < len(text) * 0.3:
        # Слишком много удалили - вернем оригинал (лучше токсично но высокий SIM)
        return text

    return result if result else text


def has_truncation(text):
    """Проверка на обрубки в конце"""
    words = text.strip().split()
    if not words:
        return False

    last_word = words[-1].lower().rstrip('.,!?;:')

    # Обрубок если заканчивается на предлог/союз
    truncation_markers = [
        'на', 'в', 'с', 'к', 'по', 'о', 'за', 'от', 'у',
        'и', 'а', 'но', 'да', 'или', 'что', 'как', 'это',
        'очен', 'белән', 'белэн', 'дэ', 'да'
    ]

    return last_word in truncation_markers


@lru_cache(maxsize=500)
def gpt_fix_truncation(orig, cleaned):
    """
    GPT ТОЛЬКО для обрубков

    ЖЕСТКИЕ ограничения:
    - НЕ менять орфографию (э, ә, җ, ө, ү, ң, һ как в оригинале!)
    - НЕ переписывать предложение
    - ТОЛЬКО убрать обрубок минимальными изменениями
    """

    if not has_truncation(cleaned):
        return cleaned

    # Быстрый fix без GPT
    words = cleaned.split()
    if words and len(words) > 1 and words[-1].lower() in ['на', 'в', 'с', 'к', 'очен', 'белән', 'белэн', 'дэ', 'да']:
        return ' '.join(words[:-1])

    prompt = f"""Убери обрубок в конце. МИНИМАЛЬНЫЕ изменения!

КРИТИЧНО:
- НЕ меняй орфографию (э, ә, җ, ө, ү, ң, һ остаются КАК ЕСТЬ!)
- НЕ переписывай предложение
- ТОЛЬКО убери последний предлог ИЛИ добавь 1 короткое слово для грамматики

Оригинал: {orig}
С обрубком: {cleaned}

Верни ТОЛЬКО исправленный текст, БЕЗ пояснений:"""

    global total_api_calls

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15,
            max_tokens=150,
            seed=42
        )

        total_api_calls += 1
        result = resp.choices[0].message.content.strip().strip('"\'`')

        # Валидация: не слишком длинный
        if len(result) > len(cleaned) * 1.3:
            return cleaned

        return result if result else cleaned

    except:
        return cleaned


def detox_pipeline(text):
    """
    Pipeline:
    1. Rule-based (главное)
    2. GPT только для обрубков
    """
    # Step 1: Rule-based
    cleaned = rule_based_detox(text)

    # Step 2: Fix обрубков (редко)
    if has_truncation(cleaned) and len(cleaned) > 10:
        fixed = gpt_fix_truncation(text, cleaned)
        return fixed

    return cleaned


def main():
    print("="*70)
    print("🏆 ФИНАЛЬНОЕ РЕШЕНИЕ - Баланс STA×SIM×FL")
    print("="*70)

    print(f"\n📥 Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"   Samples: {len(df)}")

    print(f"\n⚡ Strategy:")
    print(f"   [1] Rule-based detox (главное)")
    print(f"   [2] GPT ТОЛЬКО для обрубков (редко)")
    print(f"   [3] Приоритет: SIM > STA")

    print("\n🚀 Processing...\n")

    tqdm.pandas(desc="🎯 Detox")
    df["tat_detox1"] = df["tat_toxic"].progress_apply(detox_pipeline)

    # Валидация
    df["tat_detox1"] = df["tat_detox1"].fillna(df["tat_toxic"])
    empty_mask = df["tat_detox1"].isna() | (df["tat_detox1"].str.strip() == "")
    if empty_mask.any():
        df.loc[empty_mask, "tat_detox1"] = df.loc[empty_mask, "tat_toxic"]

    # Статистика
    changed = (df["tat_toxic"] != df["tat_detox1"]).sum()

    length_diffs = []
    for idx in range(len(df)):
        orig = df.iloc[idx]["tat_toxic"]
        detox = df.iloc[idx]["tat_detox1"]
        if len(orig) > 0:
            diff = abs(len(detox) - len(orig)) / len(orig)
            length_diffs.append(diff)

    avg_diff = sum(length_diffs) / len(length_diffs) * 100 if length_diffs else 0

    print(f"\n📊 Statistics:")
    print(f"   Changed: {changed}/{len(df)} ({changed/len(df)*100:.1f}%)")
    print(f"   Avg length Δ: {avg_diff:.1f}%")
    print(f"   GPT calls: {total_api_calls}")

    print(f"\n📦 Saving: {OUTPUT_FILE}")
    df[["ID", "tat_toxic", "tat_detox1"]].to_csv(OUTPUT_FILE, sep="\t", index=False)

    print("\n" + "="*70)
    print("✅ ГОТОВО!")
    print("="*70)

    print(f"\n🎯 Ожидания:")
    print(f"   STA: 0.80-0.85 (жертвуем немного)")
    print(f"   SIM: 0.92-0.95 (приоритет!)")
    print(f"   FL:  0.96-0.98 (минимум изменений)")
    print(f"   J:   0.70-0.75 → ПОБЕДА!")

    print(f"\n📊 Запусти: .venv/bin/python evaluate_j_score.py")


if __name__ == "__main__":
    main()
