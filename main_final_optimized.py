#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 FINAL OPTIMIZED SOLUTION for Tatar Text Detoxification
Issue #3: Optimize for J-score ≥ 0.70

Key improvements:
1. PARALLEL API calls (5-10 concurrent) to meet 30-minute time limit
2. OPTIMIZED prompt (~1000 tokens instead of ~1500-2000)
3. BETTER disguised profanity detection (Заипали, etc.)
4. MORPHOLOGICAL variant handling (кут/кутенэ/кутен)
5. PRESERVE @user mentions to maintain SIM score
6. Two-pass detoxification for problematic texts

Target metrics:
- STA (detoxification): ≥0.88
- SIM (similarity): ≥0.94
- FL (fluency): ≥0.94
- J-score: ≥0.70

Time constraint: ≤30 minutes for 600 examples
Speed target: <3 seconds per example

Author: AI Issue Solver
Date: 2025-11-30
"""

import re
import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ═══════════════════════════════════════════════════════════
# API CONFIGURATION
# ═══════════════════════════════════════════════════════════

API_KEY = "sk-C4Ju9Yy2-EKOf6SHs-jBPA"
BASE_URL = "https://api.artemox.com/v1"
MODEL_NAME = "gpt-4o-mini"

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Settings
INPUT_FILE = "dev_inputs.tsv"
OUTPUT_FILE = "submission_final.tsv"
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 8  # Parallel API calls

# Counters (thread-safe)
counter_lock = threading.Lock()
total_api_calls = 0
total_input_tokens = 0
total_output_tokens = 0

# ═══════════════════════════════════════════════════════════
# EXPANDED TOXIC LEXICON WITH MORPHOLOGICAL VARIANTS
# ═══════════════════════════════════════════════════════════

TOXIC_PATTERNS = {
    # === DISGUISED PROFANITY (CRITICAL - often missed) ===
    'disguised': [
        ('заипал', r'заипал[иоа]'),        # заебали disguised
        ('нахуй', r'нах[уе]й'),            # нахуй
        ('похуй', r'пох[уе]й'),            # похуй
        ('бляяя', r'бля{2,}'),             # extended бля
        ('хуйй', r'ху+й+'),                # extended хуй
    ],

    # === RUSSIAN STRONG PROFANITY ===
    'russian_strong': [
        ('бля', r'\bбля[тдь]?\b'),
        ('хуй', r'\bху[йёея]\b'),
        ('хуйня', r'хуйн[яуе]'),
        ('пизд', r'пизд[аеуой]?'),
        ('ебан', r'[её]бан[аыое]?'),
        ('ебать', r'[её]ба[тл][ьи]?'),
        ('сука', r'\bсук[аи]\b'),
        ('пидор', r'пидор|пидар'),
        ('блят', r'блят[ьи]?'),
    ],

    # === RUSSIAN VULGAR ===
    'russian_vulgar': [
        ('жоп', r'жоп[аеуойы]?'),
        ('срать', r'сра[тлн]|сру'),
        ('говно', r'говн[оауе]'),
        ('дерьмо', r'дерьм[оау]'),
    ],

    # === RUSSIAN MILD (less aggressive removal) ===
    'russian_mild': [
        ('блин', r'\bблин\b'),
        ('хрен', r'\bхрен[аоуы]?\b'),
        ('черт', r'\bчерт[аоу]?\b'),
    ],

    # === TATAR OFFENSIVE (ALL MORPHOLOGICAL VARIANTS) ===
    'tatar_offensive': [
        ('кутак', r'кута[ккг]'),           # кутак = ass
        ('кут', r'\bкут[ене]?\b'),         # кут = ass (root)
        ('кутен', r'кутен[эеа]'),          # кутене, кутена
        ('чучка', r'чучк[ауоеи]'),         # свинья
        ('дунгыз', r'дунгыз'),             # свинья
        ('тиле', r'\bтиле\b'),             # сумасшедший
        ('ангыра', r'ангыр[ауы]'),         # идиот
        ('сосоп', r'сосоп'),               # сосать (vulgar)
        ('тинтэк', r'тинт[эе]к'),          # fool
        ('кутлак', r'кутла[ккг]'),         # ass-related
        ('куттак', r'кутта[ккг]'),         # ass-related variant
    ],

    # === CODE-SWITCHING CONSTRUCTIONS ===
    'code_switching': [
        ('на хуй', r'на\s*х[уе][йр]'),
        ('пошол', r'пошол\s*на'),
        ('иди на', r'иди\s*на\s*х'),
        ('дохуя', r'дох[уе][йя]'),
    ],

    # === OFFENSIVE RUSSIAN WORDS ===
    'offensive': [
        ('сволочь', r'сволоч[ьи]'),
        ('идиот', r'идиот'),
        ('дебил', r'дебил'),
        ('урод', r'урод'),
        ('тварь', r'твар[ьи]'),
        ('козел', r'козел|козёл'),
        ('мудак', r'мудак'),
        ('придурок', r'придур[оа]к'),
        ('тупой', r'тупо[йе]'),
        ('дурак', r'дурак|дура\b'),
    ],
}

def get_all_toxic_patterns():
    """Compile all toxic patterns for detection"""
    patterns = []
    for category, items in TOXIC_PATTERNS.items():
        for name, pattern in items:
            patterns.append((name, re.compile(pattern, re.IGNORECASE)))
    return patterns

COMPILED_PATTERNS = get_all_toxic_patterns()


def check_remaining_toxicity(text: str) -> list:
    """Check for remaining toxic patterns in text"""
    text_lower = text.lower()
    remaining = []
    for name, pattern in COMPILED_PATTERNS:
        if pattern.search(text_lower):
            remaining.append(name)
    return remaining


# ═══════════════════════════════════════════════════════════
# OPTIMIZED PROMPT (~1000 tokens instead of ~1500-2000)
# ═══════════════════════════════════════════════════════════

def create_optimized_prompt(text: str) -> str:
    """
    Create optimized prompt (~1000 tokens) with focus on:
    1. Disguised profanity (Заипали)
    2. Morphological variants (кут/кутенэ)
    3. Minimal changes (preserve SIM)
    4. Keep @user mentions
    """

    prompt = f"""Детоксифицируй татарский текст. УДАЛИ токсичную лексику, СОХРАНИ смысл.

⚠️ ОБЯЗАТЕЛЬНО УДАЛИТЬ:
• Русские маты: бля, хуй, пизд, ебан, сука, жопа, срать, говно
• Замаскированные: Заипали (=заебали), бляяя, нахуй
• Татарские: кутак, кут, кутенэ, чучка, дунгыз, тиле, ангыра, сосоп
• Оскорбления: сволочь, идиот, дебил, придурок, тупой
• Слабые: блин, хрен (удалять если возможно)

⚠️ СОХРАНИТЬ:
• @упоминания (@user) - НЕ токсичны
• Смысл, эмоции, структуру
• Эмодзи, пунктуацию :) =D ))

📋 ПРИМЕРЫ:
Токс: сине анда котеп ятмыйлар, сиди на жопе
Детокс: сине анда котеп ятмыйлар, сиди

Токс: Заипали, нэрсэ булэк итим
Детокс: Надоело, нэрсэ булэк итим

Токс: @user, син кил давай, кутак сырлама!
Детокс: @user, син кил давай, сырлама!

Токс: Уз кутенэ приключение эзлэгэнсен
Детокс: Узе приключение эзлэгэнсен

Токс: Пашол нахуй куттакбаш
Детокс: Бар инде

ТЕКСТ: {text}

Детокс (ТОЛЬКО результат):"""

    return prompt


def create_refinement_prompt(text: str, first_attempt: str, remaining: list) -> str:
    """Create focused refinement prompt for second pass"""
    return f"""⚠️ ТОКСИЧНОСТЬ ОСТАЛАСЬ! Исправь:

Оригинал: {text}
Попытка 1: {first_attempt}
Осталось: {', '.join(remaining)}

УДАЛИ эти слова, сохрани смысл. Верни ТОЛЬКО исправленный текст:"""


# ═══════════════════════════════════════════════════════════
# PREPROCESSING - MINIMAL (preserve @user for SIM)
# ═══════════════════════════════════════════════════════════

def preprocess(text: str) -> str:
    """
    MINIMAL preprocessing - do NOT remove @user
    This preserves similarity score
    """
    if not isinstance(text, str):
        return str(text) if text else ""

    # Only normalize whitespace
    result = re.sub(r'\s+', ' ', text).strip()

    return result if result else text


# ═══════════════════════════════════════════════════════════
# POSTPROCESSING
# ═══════════════════════════════════════════════════════════

def postprocess(text: str, original: str) -> str:
    """
    Postprocess GPT output with validation
    """
    if not text:
        return original

    # Remove quotes
    text = text.strip('"\'`')

    # Remove GPT prefixes
    prefixes = [
        'детокс:', 'детоксифицированный текст:',
        'результат:', 'ответ:', 'output:',
        'детоксифицированный:', 'исправленный текст:',
    ]
    text_lower = text.lower()
    for prefix in prefixes:
        if text_lower.startswith(prefix):
            text = text[len(prefix):].strip()
            break

    # If multiple lines, take first meaningful line
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        # Skip metadata lines
        if any(marker in line.lower() for marker in ['токсичный:', 'детокс:', 'шаг ', '━━━', 'оригинал:', 'попытка']):
            continue
        if line.strip():
            clean_lines.append(line.strip())

    if clean_lines:
        text = clean_lines[0]

    # Validate length (not shorter than 20% of original)
    if len(text) < len(original) * 0.20:
        return original

    # Validate not empty
    if not text.strip():
        return original

    # Check for truncation (ends with preposition/conjunction)
    words = text.strip().split()
    if words and len(words) > 2:
        last_word = words[-1].lower().rstrip('.,!?;:')
        bad_endings = ['на', 'в', 'с', 'к', 'по', 'о', 'за', 'от', 'у', 'и', 'а', 'но', 'да', 'ли', 'э', 'э']
        if last_word in bad_endings:
            # Truncation detected - return original
            return original

    return text


# ═══════════════════════════════════════════════════════════
# MAIN DETOXIFICATION FUNCTION
# ═══════════════════════════════════════════════════════════

def detoxify_single(text: str) -> str:
    """
    Detoxify a single text with two-pass approach if needed
    """
    global total_api_calls, total_input_tokens, total_output_tokens

    # Validate input
    if not isinstance(text, str) or not text.strip():
        return text if text else ""

    # Preprocess
    preprocessed = preprocess(text)

    # First pass
    prompt = create_optimized_prompt(preprocessed)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Lower for more deterministic output
                max_tokens=300,   # Reduced for speed
                top_p=0.9,
                seed=42
            )

            with counter_lock:
                total_api_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens

            result = response.choices[0].message.content.strip()
            result = postprocess(result, text)

            # Check for remaining toxicity
            remaining = check_remaining_toxicity(result)

            if not remaining:
                return result

            # Second pass if toxicity remains
            refinement_prompt = create_refinement_prompt(text, result, remaining)

            response2 = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": refinement_prompt}],
                temperature=0.1,  # Very low for accuracy
                max_tokens=300,
                seed=42
            )

            with counter_lock:
                total_api_calls += 1
                if hasattr(response2, 'usage') and response2.usage:
                    total_input_tokens += response2.usage.prompt_tokens
                    total_output_tokens += response2.usage.completion_tokens

            result2 = response2.choices[0].message.content.strip()
            result2 = postprocess(result2, text)

            # Return better result
            remaining2 = check_remaining_toxicity(result2)
            if len(remaining2) < len(remaining):
                return result2
            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print(f"⚠️ API error: {e}")
                return text

    return text


def detoxify_batch_parallel(texts: list, max_workers: int = MAX_WORKERS) -> list:
    """
    Detoxify texts in parallel using ThreadPoolExecutor
    """
    results = [None] * len(texts)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(detoxify_single, text): idx
            for idx, text in enumerate(texts)
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(texts), desc="🎯 Детоксификация") as pbar:
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    print(f"⚠️ Error for idx {idx}: {e}")
                    results[idx] = texts[idx]  # Fallback to original
                pbar.update(1)

    return results


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("🎯 FINAL OPTIMIZED SOLUTION - Tatar Text Detoxification")
    print("   Issue #3: J-score ≥ 0.70 within 30 minutes")
    print("=" * 80)

    print(f"\n📥 Reading: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep="\t")
    print(f"   Examples: {len(df)}")

    print(f"\n⚡ Configuration:")
    print(f"   Model: {MODEL_NAME}")
    print(f"   API: {BASE_URL}")
    print(f"   Parallel workers: {MAX_WORKERS}")
    print(f"   Temperature: 0.2 (low for accuracy)")
    print(f"   Max tokens: 300 (optimized for speed)")

    print(f"\n✅ Key improvements:")
    print(f"   • Parallel API calls ({MAX_WORKERS} workers)")
    print(f"   • Optimized prompt (~1000 tokens)")
    print(f"   • Disguised profanity handling (Заипали)")
    print(f"   • Morphological variants (кут/кутенэ)")
    print(f"   • Preserve @user for SIM score")
    print(f"   • Two-pass detoxification when needed")

    print(f"\n🎯 Target metrics:")
    print(f"   STA (detoxification): ≥0.88")
    print(f"   SIM (similarity): ≥0.94")
    print(f"   FL (fluency): ≥0.94")
    print(f"   J-score: ≥0.70")

    print(f"\n⏱️ Time limit: 30 minutes")
    print(f"   Target speed: <3 sec/example")

    start_time = time.time()

    print("\n🚀 Processing...\n")

    # Get all texts
    texts = df["tat_toxic"].tolist()

    # Process in parallel
    results = detoxify_batch_parallel(texts, max_workers=MAX_WORKERS)

    # Assign results
    df["tat_detox1"] = results

    # Final validation
    df["tat_detox1"] = df["tat_detox1"].fillna(df["tat_toxic"])
    empty_mask = df["tat_detox1"].isna() | (df["tat_detox1"].str.strip() == "")
    if empty_mask.any():
        print(f"   Fixing empty: {empty_mask.sum()}")
        df.loc[empty_mask, "tat_detox1"] = df.loc[empty_mask, "tat_toxic"]

    end_time = time.time()
    elapsed = end_time - start_time

    # Statistics
    changed = (df["tat_toxic"] != df["tat_detox1"]).sum()

    length_diffs = []
    for idx in range(len(df)):
        orig = df.iloc[idx]["tat_toxic"]
        detox = df.iloc[idx]["tat_detox1"]
        if len(orig) > 0:
            diff = abs(len(detox) - len(orig)) / len(orig)
            length_diffs.append(diff)

    avg_diff = sum(length_diffs) / len(length_diffs) * 100 if length_diffs else 0

    # Check remaining toxicity
    remaining_toxic_count = 0
    for idx in range(len(df)):
        detox = df.iloc[idx]["tat_detox1"]
        if check_remaining_toxicity(detox):
            remaining_toxic_count += 1

    print(f"\n📊 Statistics:")
    print(f"   Total: {len(df)}")
    print(f"   Changed: {changed} ({changed/len(df)*100:.1f}%)")
    print(f"   Mean Δ length: {avg_diff:.1f}%")
    print(f"   Remaining toxic: {remaining_toxic_count}")
    print(f"   API calls: {total_api_calls}")
    print(f"   Tokens (input): {total_input_tokens:,}")
    print(f"   Tokens (output): {total_output_tokens:,}")
    print(f"   Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"   Speed: {elapsed/len(df):.2f} sec/example")

    # Cost (GPT-4o-mini)
    cost_input = (total_input_tokens / 1_000_000) * 0.15
    cost_output = (total_output_tokens / 1_000_000) * 0.60
    total_cost = cost_input + cost_output
    print(f"   Estimated cost: ${total_cost:.4f}")

    print(f"\n📦 Saving: {OUTPUT_FILE}")
    df[["ID", "tat_toxic", "tat_detox1"]].to_csv(OUTPUT_FILE, sep="\t", index=False)

    print("\n" + "=" * 80)
    print("✅ DONE!")
    print("=" * 80)

    # Show examples
    print("\n📋 Example changes (first 10):\n")
    shown = 0
    for idx in range(len(df)):
        orig = df.iloc[idx]["tat_toxic"]
        detox = df.iloc[idx]["tat_detox1"]

        if orig != detox and shown < 10:
            diff_pct = abs(len(detox) - len(orig)) / len(orig) * 100 if len(orig) > 0 else 0
            remaining = check_remaining_toxicity(detox)
            status = "⚠️" if remaining else "✅"
            print(f"[{idx}] Δ{diff_pct:.0f}% {status}")
            print(f"🔴 {orig[:70]}{'...' if len(orig) > 70 else ''}")
            print(f"🟢 {detox[:70]}{'...' if len(detox) > 70 else ''}")
            if remaining:
                print(f"   Remaining: {', '.join(remaining)}")
            print()
            shown += 1

    print(f"\n📊 Run evaluation:")
    print(f"   python evaluate_j_score.py {OUTPUT_FILE}")

    if elapsed > 1800:  # 30 minutes
        print(f"\n⚠️ WARNING: Processing took {elapsed/60:.1f} minutes (>30 min limit)")
        print(f"   Consider increasing MAX_WORKERS or reducing NUM_RETRIES")
    else:
        print(f"\n✅ Time constraint met: {elapsed/60:.1f} minutes (<30 min)")


if __name__ == "__main__":
    main()
