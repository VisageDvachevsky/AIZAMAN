#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 Локальный J Score калькулятор

Использует ТЕ ЖЕ модели что организаторы:
- SIM: LaBSE (sentence-transformers/LaBSE)
- STA: xlm-roberta-large toxicity classifier
- FL: XCOMET-lite

J = mean(STA × SIM × FL)
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

print("🔧 Загрузка моделей...")

# ================================
# 1. SIM - LaBSE
# ================================
print("   [1/3] LaBSE для similarity...")
labse_model = SentenceTransformer('sentence-transformers/LaBSE')

def compute_similarity(orig_texts, detox_texts):
    """Cosine similarity между LaBSE embeddings"""
    orig_emb = labse_model.encode(orig_texts, convert_to_numpy=True, show_progress_bar=False)
    detox_emb = labse_model.encode(detox_texts, convert_to_numpy=True, show_progress_bar=False)

    # Cosine similarity
    similarities = []
    for i in range(len(orig_emb)):
        cos_sim = np.dot(orig_emb[i], detox_emb[i]) / (
            np.linalg.norm(orig_emb[i]) * np.linalg.norm(detox_emb[i])
        )
        similarities.append(cos_sim)

    return np.array(similarities)


# ================================
# 2. STA - Toxicity Classifier
# ================================
print("   [2/3] XLM-RoBERTa toxicity classifier...")
tox_model_name = "textdetox/xlmr-large-toxicity-classifier-v2"
tox_tokenizer = AutoTokenizer.from_pretrained(tox_model_name)
tox_model = AutoModelForSequenceClassification.from_pretrained(tox_model_name)
tox_model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tox_model.to(device)

def compute_toxicity(texts, batch_size=32):
    """
    Возвращает вероятность нетоксичности (1 - toxicity)
    """
    non_toxicities = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tox_tokenizer(batch, padding=True, truncation=True,
                              max_length=512, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = tox_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            # Label 0 = non-toxic, Label 1 = toxic
            non_toxic_probs = probs[:, 0].cpu().numpy()
            non_toxicities.extend(non_toxic_probs)

    return np.array(non_toxicities)


# ================================
# 3. FL - XCOMET (упрощенная версия)
# ================================
print("   [3/3] Fluency estimator...")

# Для fluency используем proxy: длина + отсутствие обрубков
# Полная XCOMET модель слишком тяжелая, используем heuristic
def compute_fluency(orig_texts, detox_texts):
    """
    Heuristic fluency:
    - Нет резких обрубков
    - Сохранена длина
    - Нет незакрытой пунктуации
    """
    fluencies = []

    for orig, detox in zip(orig_texts, detox_texts):
        score = 1.0

        # Штраф за резкое сокращение
        len_ratio = len(detox) / len(orig) if len(orig) > 0 else 1.0
        if len_ratio < 0.5:
            score *= 0.7  # Большое сокращение
        elif len_ratio < 0.7:
            score *= 0.85

        # Штраф за обрубки (заканчивается на предлог/союз)
        detox_words = detox.strip().split()
        if detox_words:
            last_word = detox_words[-1].lower()
            if last_word in ['на', 'в', 'с', 'к', 'по', 'о', 'за', 'от', 'у', 'и', 'а', 'но', 'да']:
                score *= 0.6  # Обрубок

        # Штраф за незакрытую пунктуацию
        if detox.count('(') != detox.count(')'):
            score *= 0.9
        if detox.count('"') % 2 != 0:
            score *= 0.9

        fluencies.append(score)

    return np.array(fluencies)


# ================================
# 4. J Score
# ================================

def compute_j_score(orig_texts, detox_texts):
    """
    J = mean(STA × SIM × FL)
    """
    print("\n📊 Вычисление метрик...")

    # SIM
    print("   Similarity (LaBSE)...")
    sim_scores = compute_similarity(orig_texts, detox_texts)

    # STA (non-toxicity of detoxified)
    print("   Style Transfer (toxicity)...")
    sta_scores = compute_toxicity(detox_texts)

    # FL
    print("   Fluency...")
    fl_scores = compute_fluency(orig_texts, detox_texts)

    # J score per sample
    j_scores = sta_scores * sim_scores * fl_scores

    # Mean J score
    j_mean = np.mean(j_scores)

    return {
        'j_score': j_mean,
        'sta_mean': np.mean(sta_scores),
        'sim_mean': np.mean(sim_scores),
        'fl_mean': np.mean(fl_scores),
        'j_scores': j_scores,
        'sta_scores': sta_scores,
        'sim_scores': sim_scores,
        'fl_scores': fl_scores
    }


# ================================
# 5. Анализ
# ================================

def analyze_results(df, results):
    """Детальный анализ результатов"""

    print("\n" + "="*70)
    print("📊 РЕЗУЛЬТАТЫ ОЦЕНКИ")
    print("="*70)

    print(f"\n🎯 J Score: {results['j_score']:.4f}")
    print(f"\n   Компоненты:")
    print(f"   STA (detoxification): {results['sta_mean']:.4f}")
    print(f"   SIM (similarity):     {results['sim_mean']:.4f}")
    print(f"   FL  (fluency):        {results['fl_mean']:.4f}")

    # Находим проблемные примеры
    print(f"\n🔍 ПРОБЛЕМНЫЕ ПРИМЕРЫ:\n")

    # Низкий STA (токсичность осталась)
    low_sta = np.where(results['sta_scores'] < 0.7)[0]
    if len(low_sta) > 0:
        print(f"   ⚠️  Низкий STA (токсичность осталась): {len(low_sta)} примеров")
        for idx in low_sta[:5]:
            print(f"      [{idx}] STA={results['sta_scores'][idx]:.2f}")
            print(f"          {df.iloc[idx]['tat_detox1'][:70]}\n")

    # Низкий SIM (слишком много изменений)
    low_sim = np.where(results['sim_scores'] < 0.8)[0]
    if len(low_sim) > 0:
        print(f"   ⚠️  Низкий SIM (много изменений): {len(low_sim)} примеров")
        for idx in low_sim[:5]:
            print(f"      [{idx}] SIM={results['sim_scores'][idx]:.2f}")
            print(f"          Orig:  {df.iloc[idx]['tat_toxic'][:60]}")
            print(f"          Detox: {df.iloc[idx]['tat_detox1'][:60]}\n")

    # Низкий FL (неграмматичность)
    low_fl = np.where(results['fl_scores'] < 0.7)[0]
    if len(low_fl) > 0:
        print(f"   ⚠️  Низкий FL (обрубки): {len(low_fl)} примеров")
        for idx in low_fl[:5]:
            print(f"      [{idx}] FL={results['fl_scores'][idx]:.2f}")
            print(f"          {df.iloc[idx]['tat_detox1'][:70]}\n")

    # Лучшие примеры
    print(f"\n✅ ЛУЧШИЕ ПРИМЕРЫ (высокий J):\n")
    top_indices = np.argsort(results['j_scores'])[-5:][::-1]
    for idx in top_indices:
        j = results['j_scores'][idx]
        sta = results['sta_scores'][idx]
        sim = results['sim_scores'][idx]
        fl = results['fl_scores'][idx]

        print(f"   [{idx}] J={j:.3f} (STA={sta:.2f}, SIM={sim:.2f}, FL={fl:.2f})")
        print(f"       Orig:  {df.iloc[idx]['tat_toxic'][:65]}")
        print(f"       Detox: {df.iloc[idx]['tat_detox1'][:65]}\n")


def main():
    import sys

    submission_file = sys.argv[1] if len(sys.argv) > 1 else "submission.tsv"

    print(f"\n📥 Загрузка: {submission_file}")
    df = pd.read_csv(submission_file, sep='\t')
    print(f"   Примеров: {len(df)}")

    # Вычисляем J score
    results = compute_j_score(
        df['tat_toxic'].tolist(),
        df['tat_detox1'].tolist()
    )

    # Анализ
    analyze_results(df, results)

    print("\n" + "="*70)
    print(f"🎯 ФИНАЛЬНЫЙ J SCORE: {results['j_score']:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
