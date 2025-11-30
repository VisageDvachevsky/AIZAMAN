#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🏆 BALANCED SOLUTION - Баланс детоксификации и сохранения

СТРАТЕГИЯ:
1. GPT для детоксификации (видит контекст)
2. ЖЕСТКИЙ контроль орфографии через промпт
3. Двухпроходная проверка: если изменений >25% → откат
4. Баланс: STA=0.82+, SIM=0.90+

КЛЮЧЕВОЕ:
- НЕ менять орфографию (э, ә, җ, ө, ү, ң, һ)
- Минимальные изменения структуры
- Удалять маты + агрессивные обороты
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

total_input_tokens = 0
total_output_tokens = 0
total_api_calls = 0


@lru_cache(maxsize=2000)
def balanced_detox(text: str) -> str:
    """
    GPT детоксификация с ЖЕСТКИМ контролем орфографии
    """

    prompt = f"""Детоксифицируй татарский текст. Удали ТОЛЬКО явные маты и грубую лексику.

🎯 КРИТИЧНО - СОХРАНИ ОРФОГРАФИЮ:
- НЕ меняй татарские буквы: э, ә, җ, ө, ү, ң, һ, ғ, қ
- НЕ исправляй опечатки в оригинале
- Сохрани ВСЕ буквы КАК ЕСТЬ (если "дэ" - оставь "дэ", НЕ меняй на "да")
- Сохрани пунктуацию и смайлики

📋 ЧТО УДАЛЯТЬ:
✗ Явные маты (русские и татарские): бля*, хуй*, жоп*, кут*, ебать*, сука*, пизд*, хрен*, хули*
✗ Грубые оскорбления: ахмак*, ангыра*, дебил*, идиот*, долба*
✗ Пренебрежительные слова: чтоль/чтоли можно оставить

📋 ЧТО СОХРАНЯТЬ:
✓ Обычные татарские слова (син, мин, бит, инде, баш, кит и т.д.)
✓ Контекст и структуру предложения
✓ Нейтральные выражения

📋 ПРИМЕРЫ (обрати внимание на сохранение орфографии):

Вход: Купме ашарга була инде? Симереп чучка буласыз бит
Выход: Купме ашарга була инде? Симереп буласыз бит

Вход: уф бля.сектегез уже.шундый кутак баш чтоли ты?сама решай!
Выход: уф. сектегез уже. шундый баш чтоли ты? сама решай!

Вход: Эй, хули комментны удалил? немес блят! Атып утермэкче идем инде сине
Выход: Эй, комментны удалил? немес! Атып утермэкче идем инде сине

Вход: ни хрена да булышмады синен счинакомын, тамак авырта белэт!:) хуйня какая-то
Выход: ни да булышмады синен счинакомын, тамак авырта белэт!:)

Вход: Кызлар сез тормозлар, шул томозлык аркасында шундый заебатый егетлэр югалтасыз!
Выход: Кызлар сез тормозлар, шул томозлык аркасында шундый егетлэр югалтасыз!

Вход: татарча яз, кутенне корт чаккыры!!!
Выход: татарча яз!!!

ТЕКСТ ДЛЯ ДЕТОКСИФИКАЦИИ:
{text}

ВАЖНО: Верни ТОЛЬКО детоксифицированный текст, БЕЗ объяснений!"""

    global total_input_tokens, total_output_tokens, total_api_calls

    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Низкая для консистентности
                max_tokens=300,
                top_p=0.9,
                seed=42
            )

            if hasattr(resp, 'usage') and resp.usage:
                total_input_tokens += resp.usage.prompt_tokens
                total_output_tokens += resp.usage.completion_tokens
                total_api_calls += 1

            result = resp.choices[0].message.content.strip()

            # Очистка от артефактов
            lines = result.split('\n')
            clean_lines = []
            for line in lines:
                if any(marker in line for marker in ['Вход:', 'Выход:', 'ТЕКСТ:', 'ВАЖНО:']):
                    continue
                if line.strip():
                    clean_lines.append(line)

            if clean_lines:
                result = clean_lines[0].strip()

            result = result.strip('"\'`')

            # Валидация длины
            if len(result) < len(text) * 0.35:
                return text

            if len(result) > len(text) * 1.4:
                return text

            # КРИТИЧНО: Проверка изменений
            # Если изменили >25% длины - слишком много, откат
            len_change = abs(len(result) - len(text)) / len(text)
            if len_change > 0.25:
                # Слишком много изменений - вернем оригинал
                # Лучше немного токсично но высокий SIM
                return text

            if not result or result.isspace():
                return text

            return result

        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return text

    return text


def main():
    print("="*70)
    print("🏆 BALANCED SOLUTION - Баланс STA×SIM×FL")
    print("="*70)

    print(f"\n📥 Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"   Samples: {len(df)}")

    print(f"\n⚡ Strategy:")
    print(f"   [1] GPT детоксификация (видит контекст)")
    print(f"   [2] ЖЕСТКИЙ контроль орфографии в промпте")
    print(f"   [3] Откат если изменений >25%")
    print(f"   [4] Баланс: STA=0.82+, SIM=0.90+")

    print("\n🚀 Processing...\n")

    tqdm.pandas(desc="🎯 Balanced detox")
    df["tat_detox1"] = df["tat_toxic"].progress_apply(balanced_detox)

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
    print(f"   Expected SIM: ~{max(85, 100-avg_diff):.0f}%")

    print(f"\n📦 Saving: {OUTPUT_FILE}")
    df[["ID", "tat_toxic", "tat_detox1"]].to_csv(OUTPUT_FILE, sep="\t", index=False)

    print("\n" + "="*70)
    print("✅ ГОТОВО!")
    print("="*70)

    print(f"\n💰 Tokens:")
    print(f"   API calls: {total_api_calls}")
    print(f"   Input: {total_input_tokens:,}")
    print(f"   Output: {total_output_tokens:,}")
    cost = (total_input_tokens / 1_000_000) * 2.5 + (total_output_tokens / 1_000_000) * 10
    print(f"   Cost: ${cost:.2f}")

    print(f"\n🎯 Ожидания:")
    print(f"   STA: 0.82-0.86 (GPT видит контекст)")
    print(f"   SIM: 0.88-0.92 (контроль орфографии + откат)")
    print(f"   FL:  0.96-0.98 (грамматика)")
    print(f"   J:   0.68-0.73 → Цель 0.70+!")

    print(f"\n📊 Запусти оценку: .venv/bin/python evaluate_j_score.py")


if __name__ == "__main__":
    main()
