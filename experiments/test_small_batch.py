#!/usr/bin/env python3
"""
Test the final optimized solution on a small batch (20 examples)
to verify it works correctly before running on full dataset.
"""

import sys
sys.path.insert(0, '/tmp/gh-issue-solver-1764499499780')

import pandas as pd
from main_final_optimized import (
    detoxify_single, check_remaining_toxicity,
    create_optimized_prompt, preprocess, postprocess
)
import time

print("=" * 80)
print("🧪 Testing main_final_optimized.py on small batch")
print("=" * 80)

# Load data
df = pd.read_csv("dev_inputs.tsv", sep="\t")
print(f"\n📥 Loaded {len(df)} examples")

# Test on first 20 examples
test_size = 20
test_df = df.head(test_size).copy()
print(f"   Testing on {test_size} examples")

# Also include the problematic examples from issue
problem_indices = [14, 15, 17, 18, 26]
for idx in problem_indices:
    if idx < len(df) and idx >= test_size:
        test_df = pd.concat([test_df, df.iloc[[idx]]])

print(f"   + Adding problem examples: {problem_indices}")
print(f"   Total test examples: {len(test_df)}")

# Test each example
print("\n" + "=" * 80)
print("🎯 Processing examples...")
print("=" * 80)

start_time = time.time()
results = []

for idx in range(len(test_df)):
    row = test_df.iloc[idx]
    orig = row["tat_toxic"]
    real_idx = row["ID"] if "ID" in row else idx

    try:
        detox = detoxify_single(orig)
        remaining = check_remaining_toxicity(detox)

        status = "⚠️" if remaining else "✅"
        print(f"\n[{real_idx}] {status}")
        print(f"  Orig:  {orig[:70]}{'...' if len(orig) > 70 else ''}")
        print(f"  Detox: {detox[:70]}{'...' if len(detox) > 70 else ''}")
        if remaining:
            print(f"  ⚠️ Remaining toxic: {', '.join(remaining)}")

        results.append({
            "idx": real_idx,
            "orig": orig,
            "detox": detox,
            "remaining": remaining,
            "changed": orig != detox
        })

    except Exception as e:
        print(f"\n[{real_idx}] ❌ Error: {e}")
        results.append({
            "idx": real_idx,
            "orig": orig,
            "detox": orig,
            "remaining": [],
            "changed": False,
            "error": str(e)
        })

elapsed = time.time() - start_time

# Summary
print("\n" + "=" * 80)
print("📊 Summary")
print("=" * 80)

total = len(results)
changed = sum(1 for r in results if r["changed"])
remaining_toxic = sum(1 for r in results if r["remaining"])
errors = sum(1 for r in results if "error" in r)

print(f"\n  Total: {total}")
print(f"  Changed: {changed} ({changed/total*100:.1f}%)")
print(f"  With remaining toxicity: {remaining_toxic}")
print(f"  Errors: {errors}")
print(f"  Time: {elapsed:.1f}s ({elapsed/total:.2f}s per example)")

# Check specific problem examples
print("\n" + "=" * 80)
print("🔍 Problem examples check")
print("=" * 80)

for r in results:
    if r["idx"] in problem_indices:
        print(f"\n[{r['idx']}]")
        print(f"  Orig:  {r['orig'][:70]}")
        print(f"  Detox: {r['detox'][:70]}")
        if r["remaining"]:
            print(f"  ❌ Still toxic: {', '.join(r['remaining'])}")
        else:
            print(f"  ✅ Detoxified successfully")

print("\n" + "=" * 80)
print("✅ Test completed!")
print("=" * 80)
