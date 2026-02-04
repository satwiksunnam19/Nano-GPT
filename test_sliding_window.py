"""
Test sliding window attention in the model
"""

import torch
from model import GPT, GPTConfig

print("=" * 70)
print("Testing Sliding Window Attention in GPT Model")
print("=" * 70)

# Test 1: Standard causal attention (baseline)
print("\n1. Standard Causal Attention (No Sliding Window)")
print("-" * 70)
config_standard = GPTConfig(
    block_size=128,
    vocab_size=50304,
    n_layer=2,
    n_head=8,
    n_embd=512,
    dropout=0.0,
    bias=True
)
model_standard = GPT(config_standard)
print(f"✓ Model created with standard causal attention")
print(f"  - Layers: {config_standard.n_layer}, Heads: {config_standard.n_head}")

# Test with input
B, T = 2, 32
idx = torch.randint(0, config_standard.vocab_size, (B, T))
logits, loss = model_standard(idx)
print(f"✓ Forward pass successful: {logits.shape}")

# Test 2: Sliding window attention
print("\n2. Sliding Window Attention")
print("-" * 70)
config_sliding = GPTConfig(
    block_size=128,
    vocab_size=50304,
    n_layer=2,
    n_head=8,
    n_embd=512,
    dropout=0.0,
    bias=True,
    window_size=16,
    sink_size=4
)
model_sliding = GPT(config_sliding)
print(f"✓ Model created with sliding window attention")
print(f"  - Window size: {config_sliding.window_size}")
print(f"  - Sink size: {config_sliding.sink_size}")

logits_sliding, loss_sliding = model_sliding(idx)
print(f"✓ Forward pass successful: {logits_sliding.shape}")

# Test 3: GQA with sliding window
print("\n3. GQA with Sliding Window")
print("-" * 70)
config_gqa = GPTConfig(
    block_size=128,
    vocab_size=50304,
    n_layer=2,
    n_head=8,
    n_embd=512,
    n_kv_heads=2,
    dropout=0.0,
    bias=True,
    attention_type="gqa",
    window_size=16,
    sink_size=4
)
model_gqa = GPT(config_gqa)
print(f"✓ Model created with GQA + sliding window")
print(f"  - Q heads: {config_gqa.n_head}, KV heads: {config_gqa.n_kv_heads}")
print(f"  - Window size: {config_gqa.window_size}, Sink size: {config_gqa.sink_size}")

logits_gqa, loss_gqa = model_gqa(idx)
print(f"✓ Forward pass successful: {logits_gqa.shape}")

# Test 4: MLA with sliding window
print("\n4. MLA with Sliding Window")
print("-" * 70)
config_mla = GPTConfig(
    block_size=128,
    vocab_size=50304,
    n_layer=2,
    n_head=8,
    n_embd=512,
    latent_dim=128,
    dropout=0.0,
    bias=True,
    attention_type="mla",
    window_size=16,
    sink_size=4
)
model_mla = GPT(config_mla)
print(f"✓ Model created with MLA + sliding window")
print(f"  - Latent dim: {config_mla.latent_dim}")
print(f"  - Window size: {config_mla.window_size}, Sink size: {config_mla.sink_size}")

logits_mla, loss_mla = model_mla(idx)
print(f"✓ Forward pass successful: {logits_mla.shape}")

# Memory comparison
print("\n5. Memory Complexity")
print("-" * 70)
seq_len = 4096
standard_memory = seq_len * seq_len
sliding_memory = seq_len * (config_sliding.window_size + config_sliding.sink_size)

print(f"For sequence length = {seq_len}:")
print(f"  Standard attention: {standard_memory:,} elements ({standard_memory / 1e6:.2f}M)")
print(f"  Sliding window:     {sliding_memory:,} elements ({sliding_memory / 1e6:.2f}M)")
print(f"  Memory reduction:   {100 * (1 - sliding_memory / standard_memory):.1f}%")

print("\n" + "=" * 70)
print("ALL TESTS PASSED!")
print("=" * 70)
print("\nTo use sliding window in your training:")
print("  config = GPTConfig(")
print("      ...,")
print("      window_size=512,    # Recent tokens to attend to")
print("      sink_size=4,        # Initial tokens always accessible")
print("  )")
print("=" * 70)
