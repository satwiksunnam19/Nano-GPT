#!/usr/bin/env python3
"""
Token Counting Script for Shakespeare Dataset
Counts total tokens in train.bin and val.bin files
"""

import numpy as np
import os

def count_tokens_in_dataset(data_dir='data/shakesphere'):
    """Count tokens in train and validation binary files."""

    train_path = os.path.join(data_dir, 'train.bin')
    val_path = os.path.join(data_dir, 'val.bin')

    print("=" * 60)
    print("Shakespeare Dataset Token Analysis")
    print("=" * 60)

    # Load train tokens
    if os.path.exists(train_path):
        train_tokens = np.memmap(train_path, dtype=np.uint16, mode='r')
        train_count = len(train_tokens)
        print(f"\n📚 Training Set:")
        print(f"   Total tokens: {train_count:,} ({train_count/1e6:.2f}M)")
    else:
        print(f"\n❌ Train file not found: {train_path}")
        train_count = 0

    # Load validation tokens
    if os.path.exists(val_path):
        val_tokens = np.memmap(val_path, dtype=np.uint16, mode='r')
        val_count = len(val_tokens)
        print(f"\n📖 Validation Set:")
        print(f"   Total tokens: {val_count:,} ({val_count/1e6:.2f}M)")
    else:
        print(f"\n❌ Val file not found: {val_path}")
        val_count = 0

    total_count = train_count + val_count
    print(f"\n🔢 Total Dataset:")
    print(f"   Total tokens: {total_count:,} ({total_count/1e6:.2f}M)")

    print("\n" + "=" * 60)
    print("Training Coverage Analysis")
    print("=" * 60)

    # Calculate tokens consumed per iteration
    # From train_m4.sh: batch_size=6, block_size=384, gradient_accum_steps=1
    batch_size = 6
    block_size = 384
    grad_accum_steps = 1
    tokens_per_iter = batch_size * block_size * grad_accum_steps

    print(f"\n⚙️  Training Configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Block size: {block_size}")
    print(f"   Gradient accumulation steps: {grad_accum_steps}")
    print(f"   Tokens per iteration: {tokens_per_iter:,}")

    # Calculate epochs for different training runs
    training_configs = [
        ("train_m4.sh (8000 iters)", 8000),
        ("Short run (5000 iters)", 5000),
        ("Medium run (10000 iters)", 10000),
        ("Long run (15000 iters)", 15000),
    ]

    print(f"\n📊 Dataset Coverage for Different Training Runs:")
    print(f"{'Training Run':<30} {'Total Tokens':<15} {'Epochs':<10} {'Dataset Passes'}")
    print("-" * 80)

    for name, iters in training_configs:
        total_tokens_consumed = iters * tokens_per_iter
        epochs = total_tokens_consumed / train_count if train_count > 0 else 0
        print(f"{name:<30} {total_tokens_consumed/1e6:>8.2f}M      {epochs:>8.2f}x    {'█' * int(epochs)}")

    print("\n" + "=" * 60)
    print("Benchmark Attention Study Analysis")
    print("=" * 60)

    # From benchmark_attention.py: max_iters=5000
    benchmark_iters = 5000
    benchmark_tokens = benchmark_iters * tokens_per_iter
    benchmark_epochs = benchmark_tokens / train_count if train_count > 0 else 0

    print(f"\n🔬 Benchmark Configuration:")
    print(f"   Max iterations: {benchmark_iters:,}")
    print(f"   Total tokens consumed: {benchmark_tokens/1e6:.2f}M")
    print(f"   Epochs (dataset passes): {benchmark_epochs:.2f}x")
    print(f"   Train set utilization: {(benchmark_epochs * 100):.1f}%")

    if benchmark_epochs < 1.0:
        print(f"\n⚠️  WARNING: Training will not complete even 1 full epoch!")
        print(f"   Consider increasing max_iters to at least {int(train_count / tokens_per_iter)} for 1 epoch")
    elif benchmark_epochs < 3.0:
        print(f"\n✅ Training will see the dataset {benchmark_epochs:.1f} times")
        print(f"   This is reasonable for a quick ablation study")
    else:
        print(f"\n✅ Training will see the dataset {benchmark_epochs:.1f} times")
        print(f"   Good coverage for comparing attention mechanisms")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    count_tokens_in_dataset()
