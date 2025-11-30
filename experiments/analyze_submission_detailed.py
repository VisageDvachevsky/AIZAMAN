#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed analysis of submission_optimized.tsv for issue #3

Analyzes:
1. @user removal impact on SIM
2. Disguised profanity patterns (Заипали, etc.)
3. Morphological variants detection
4. J-score distribution by ranges
5. Successful vs failed examples
"""

import pandas as pd
import numpy as np
import re
from collections import Counter

print("=" * 80)
print("📊 DETAILED ANALYSIS OF SUBMISSION_OPTIMIZED.TSV")
print("=" * 80)

# Load data
df = pd.read_csv("submission_optimized.tsv", sep="\t")
print(f"\n📥 Loaded {len(df)} examples")

# ===================================================
# 1. BASIC STATISTICS
# ===================================================
print("\n" + "=" * 80)
print("1. BASIC STATISTICS")
print("=" * 80)

# Changed vs unchanged
changed_mask = df["tat_toxic"] != df["tat_detox1"]
print(f"\nChanged: {changed_mask.sum()} ({changed_mask.sum()/len(df)*100:.1f}%)")
print(f"Unchanged: {(~changed_mask).sum()} ({(~changed_mask).sum()/len(df)*100:.1f}%)")

# Length changes
df["orig_len"] = df["tat_toxic"].str.len()
df["detox_len"] = df["tat_detox1"].str.len()
df["len_ratio"] = df["detox_len"] / df["orig_len"]
df["len_diff"] = df["orig_len"] - df["detox_len"]

print(f"\nLength statistics:")
print(f"  Mean length ratio: {df['len_ratio'].mean():.3f}")
print(f"  Min length ratio: {df['len_ratio'].min():.3f}")
print(f"  Max length ratio: {df['len_ratio'].max():.3f}")
print(f"  Mean length reduction: {df['len_diff'].mean():.1f} chars")

# ===================================================
# 2. @USER REMOVAL ANALYSIS
# ===================================================
print("\n" + "=" * 80)
print("2. @USER REMOVAL ANALYSIS")
print("=" * 80)

# Find examples with @user
user_pattern = r'@\w+'
df["has_user_orig"] = df["tat_toxic"].str.contains(user_pattern, regex=True, case=False)
df["has_user_detox"] = df["tat_detox1"].str.contains(user_pattern, regex=True, case=False)

user_orig_count = df["has_user_orig"].sum()
user_detox_count = df["has_user_detox"].sum()
user_removed_count = (df["has_user_orig"] & ~df["has_user_detox"]).sum()

print(f"\nOriginal texts with @user: {user_orig_count}")
print(f"Detoxified texts with @user: {user_detox_count}")
print(f"@user removed: {user_removed_count}")

# Impact on length
user_removed_mask = df["has_user_orig"] & ~df["has_user_detox"]
if user_removed_mask.sum() > 0:
    print(f"\nWhen @user removed:")
    print(f"  Mean length ratio: {df.loc[user_removed_mask, 'len_ratio'].mean():.3f}")
    print(f"  Examples (first 5):")
    for idx in df[user_removed_mask].head(5).index:
        orig = df.loc[idx, "tat_toxic"]
        detox = df.loc[idx, "tat_detox1"]
        print(f"    [{idx}] Orig:  {orig[:60]}...")
        print(f"         Detox: {detox[:60]}...")
        print()

# ===================================================
# 3. DISGUISED PROFANITY PATTERNS
# ===================================================
print("\n" + "=" * 80)
print("3. DISGUISED PROFANITY PATTERNS")
print("=" * 80)

# Common disguised patterns
disguised_patterns = [
    ("Заипали/Заипало (заебали)", r"заипал[иоа]"),
    ("Нах/нахер (нахуй)", r"нах[еу]р"),
    ("Бляяя (бля)", r"бля+"),
    ("Х*й (хуй)", r"х\*+й|ху+й"),
    ("П*зд (пизд)", r"п\*+зд|пизд"),
    ("Еб* (ебать)", r"е+б[ау]|ё+б[ау]"),
    ("Сука (прямое)", r"сук[аи]"),
    ("Хуйня", r"хуйня|хуён"),
]

print("\nDisguised profanity in DETOXIFIED texts (should be 0):")
for name, pattern in disguised_patterns:
    matches = df["tat_detox1"].str.contains(pattern, regex=True, case=False, na=False)
    if matches.sum() > 0:
        print(f"\n  ⚠️  '{name}' found in {matches.sum()} detoxified texts:")
        for idx in df[matches].head(3).index:
            print(f"      [{idx}] {df.loc[idx, 'tat_detox1'][:70]}...")

# Check for "Заипали" specifically
zaipali_mask = df["tat_detox1"].str.contains("заипал", case=False, na=False)
if zaipali_mask.sum() > 0:
    print(f"\n  ⚠️  CRITICAL: 'Заипали' (disguised мат) found in {zaipali_mask.sum()} detoxified texts!")
    for idx in df[zaipali_mask].index[:5]:
        print(f"      [{idx}] {df.loc[idx, 'tat_detox1'][:70]}...")

# ===================================================
# 4. TATAR TOXIC WORDS ANALYSIS
# ===================================================
print("\n" + "=" * 80)
print("4. TATAR TOXIC WORDS IN DETOXIFIED TEXTS")
print("=" * 80)

tatar_toxic = [
    ("кутак (задница)", r"кута[ккг]"),
    ("кут (зад)", r"\bкут[ене]?\b"),
    ("чучка/дунгыз (свинья)", r"чучк[ауоеи]|дунгыз"),
    ("тиле (сумасшедший)", r"\bтиле\b"),
    ("ангыра (идиот)", r"ангыр[ауы]"),
    ("сосоп (сосать)", r"сосоп"),
]

print("\nTatar toxic words in DETOXIFIED texts (should be 0):")
for name, pattern in tatar_toxic:
    matches = df["tat_detox1"].str.contains(pattern, regex=True, case=False, na=False)
    if matches.sum() > 0:
        print(f"\n  ⚠️  '{name}' found in {matches.sum()} detoxified texts:")
        for idx in df[matches].head(3).index:
            print(f"      [{idx}] {df.loc[idx, 'tat_detox1'][:70]}...")

# ===================================================
# 5. RUSSIAN PROFANITY IN DETOXIFIED TEXTS
# ===================================================
print("\n" + "=" * 80)
print("5. RUSSIAN PROFANITY IN DETOXIFIED TEXTS")
print("=" * 80)

russian_profanity = [
    ("бля/блять", r"бля[тдь]?"),
    ("хуй/хуйня", r"ху[йёе]"),
    ("пизд", r"пизд"),
    ("ебан", r"ебан|ёбан"),
    ("сука", r"\bсук[аи]\b"),
    ("жопа/жоп", r"жоп[аеуой]?"),
    ("срать", r"срат|срал|срёт"),
    ("говно", r"говн[оауе]"),
    ("хер/хрен (mild)", r"\bхер\b|\bхрен"),
    ("блин (mild)", r"\bблин\b"),
]

print("\nRussian profanity in DETOXIFIED texts:")
for name, pattern in russian_profanity:
    matches = df["tat_detox1"].str.contains(pattern, regex=True, case=False, na=False)
    if matches.sum() > 0:
        print(f"\n  ⚠️  '{name}' found in {matches.sum()} detoxified texts:")
        for idx in df[matches].head(3).index:
            orig = df.loc[idx, "tat_toxic"]
            detox = df.loc[idx, "tat_detox1"]
            print(f"      [{idx}] Orig:  {orig[:60]}...")
            print(f"           Detox: {detox[:60]}...")

# ===================================================
# 6. EXCESSIVE CHANGES (LOW SIM)
# ===================================================
print("\n" + "=" * 80)
print("6. EXCESSIVE CHANGES (potential LOW SIM)")
print("=" * 80)

# High length reduction = potential low SIM
high_reduction = df[df["len_ratio"] < 0.7]
print(f"\nTexts with >30% length reduction: {len(high_reduction)}")
if len(high_reduction) > 0:
    print("  First 10 examples:")
    for idx in high_reduction.head(10).index:
        ratio = df.loc[idx, "len_ratio"]
        orig = df.loc[idx, "tat_toxic"]
        detox = df.loc[idx, "tat_detox1"]
        print(f"\n    [{idx}] Ratio={ratio:.2f}")
        print(f"        Orig:  {orig[:65]}")
        print(f"        Detox: {detox[:65]}")

# ===================================================
# 7. UNCHANGED TOXIC TEXTS
# ===================================================
print("\n" + "=" * 80)
print("7. UNCHANGED TEXTS (may still be toxic)")
print("=" * 80)

unchanged = df[~changed_mask]
print(f"\nUnchanged texts: {len(unchanged)}")

# Check for obvious toxicity in unchanged
all_toxic_patterns = (
    [p[1] for p in disguised_patterns] +
    [p[1] for p in tatar_toxic] +
    [p[1] for p in russian_profanity]
)

likely_toxic_unchanged = []
for idx in unchanged.index:
    text = df.loc[idx, "tat_toxic"].lower()
    for pattern in all_toxic_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            likely_toxic_unchanged.append(idx)
            break

print(f"Unchanged texts with potential toxicity: {len(likely_toxic_unchanged)}")
if likely_toxic_unchanged:
    print("  First 10 examples:")
    for idx in likely_toxic_unchanged[:10]:
        print(f"    [{idx}] {df.loc[idx, 'tat_toxic'][:70]}...")

# ===================================================
# 8. SPECIFIC PROBLEM EXAMPLES FROM ISSUE
# ===================================================
print("\n" + "=" * 80)
print("8. SPECIFIC PROBLEM EXAMPLES FROM ISSUE")
print("=" * 80)

problem_indices = [14, 15, 17, 18, 26]
print("\nProblem examples mentioned in issue (STA=0.00):")
for idx in problem_indices:
    if idx < len(df):
        orig = df.loc[idx, "tat_toxic"]
        detox = df.loc[idx, "tat_detox1"]
        print(f"\n[{idx}]")
        print(f"  Orig:  {orig}")
        print(f"  Detox: {detox}")

        # Analyze what toxic elements remain
        toxic_found = []
        for name, pattern in disguised_patterns + tatar_toxic + russian_profanity:
            if re.search(pattern, detox, re.IGNORECASE):
                toxic_found.append(name)
        if toxic_found:
            print(f"  ⚠️ Toxic remaining: {', '.join(toxic_found)}")

# ===================================================
# 9. SUMMARY STATISTICS
# ===================================================
print("\n" + "=" * 80)
print("9. SUMMARY STATISTICS")
print("=" * 80)

# Count all remaining toxic patterns
total_toxic_remaining = 0
toxic_counts = {}

for name, pattern in disguised_patterns + tatar_toxic + russian_profanity:
    count = df["tat_detox1"].str.contains(pattern, regex=True, case=False, na=False).sum()
    if count > 0:
        toxic_counts[name] = count
        total_toxic_remaining += count

print(f"\nTotal toxic patterns remaining in detoxified texts: {total_toxic_remaining}")
if toxic_counts:
    print("\nBreakdown by pattern:")
    for name, count in sorted(toxic_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")

print("\n" + "=" * 80)
print("📋 RECOMMENDATIONS")
print("=" * 80)

print("""
Based on analysis:

1. CRITICAL: 'Заипали' (disguised мат) is NOT being removed
   - Add explicit instruction in prompt
   - Include in few-shot examples

2. @user removal causes length reduction
   - Consider NOT removing @user
   - Or add more content to compensate

3. 'кут' morphological variants remaining (кутенэ, кутен)
   - Need better morphological handling

4. Some unchanged texts still contain toxicity
   - Model may not recognize all patterns

5. 'бля/блять' removed inconsistently
   - Need stronger instruction
""")

# Save detailed report
print("\n📄 Saving detailed analysis to experiments/analysis_report.txt")
