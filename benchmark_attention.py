"""
Benchmark different attention mechanisms on NanoGPT

Compares:
- Standard MHA (baseline)
- GQA with different n_kv_heads
- MLA with different latent_dim
- Sliding Window with different window_size
- Combinations (GQA + Sliding Window)

Metrics:
- Training speed (ms/iter)
- Memory usage (MB)
- Loss convergence (steps to reach target)
- Final validation loss
"""

import torch
import time
import os
import json
from contextlib import nullcontext
from model import GPTConfig, GPT
from datetime import datetime

# Hyperparameters (small model for fast benchmarking)
batch_size = 4
block_size = 256
max_iters = 1000
eval_interval = 200
eval_iters = 50
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# Model config (consistent across all variants) 
base_config = {
    'n_layer': 4,
    'n_head': 6,
    'n_embd': 384,
    'dropout': 0.0,
    'bias': False,
    'vocab_size': 50304,
    'block_size': block_size,
}

# Attention variants to benchmark
attention_configs = [
    # 1. Baseline: Standard MHA
    {
        'name': 'Standard MHA',
        'attention_type': 'standard',
        'description': 'Baseline - standard multi-head attention'
    },

    # 2. GQA variants
    {
        'name': 'GQA (n_kv=2)',
        'attention_type': 'gqa',
        'n_kv_heads': 2,
        'description': 'Grouped Query Attention with 2 KV heads (3x reduction)'
    },
    {
        'name': 'GQA (n_kv=3)',
        'attention_type': 'gqa',
        'n_kv_heads': 3,
        'description': 'Grouped Query Attention with 3 KV heads (2x reduction)'
    },

    # 3. MLA variants
    {
        'name': 'MLA (latent=128)',
        'attention_type': 'mla',
        'latent_dim': 128,
        'description': 'Multi-head Latent Attention with latent_dim=128'
    },
    {
        'name': 'MLA (latent=64)',
        'attention_type': 'mla',
        'latent_dim': 64,
        'description': 'Multi-head Latent Attention with latent_dim=64 (higher compression)'
    },

    # 4. Sliding Window variants
    {
        'name': 'Sliding Window (256)',
        'attention_type': 'standard',
        'window_size': 256,
        'sink_size': 4,
        'description': 'Sliding window attention, window=256, sink=4'
    },
    {
        'name': 'Sliding Window (128)',
        'attention_type': 'standard',
        'window_size': 128,
        'sink_size': 4,
        'description': 'Sliding window attention, window=128, sink=4'
    },

    # 5. Combined: GQA + Sliding Window
    {
        'name': 'GQA+SW (n_kv=2, w=128)',
        'attention_type': 'gqa',
        'n_kv_heads': 2,
        'window_size': 128,
        'sink_size': 4,
        'description': 'GQA with 2 KV heads + sliding window (128, 4)'
    },
]


def get_batch(split, data_dir='data/shakesphere'):
    """Load a batch of data"""
    import numpy as np

    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])

    if device == 'mps':
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, eval_iters):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with nullcontext():
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_memory_usage():
    """Get current memory usage in MB"""
    if device == 'mps':
        return torch.mps.current_allocated_memory() / 1024 / 1024
    elif device == 'cuda':
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        return 0.0


def benchmark_config(config_dict, config_name):
    """Benchmark a single attention configuration"""
    print(f"\n{'='*70}")
    print(f"Benchmarking: {config_name}")
    print(f"Description: {config_dict['description']}")
    print(f"{'='*70}")

    # Create model config
    model_config = {**base_config, **{k: v for k, v in config_dict.items() if k not in ['name', 'description']}}
    config = GPTConfig(**model_config)

    # Create model
    model = GPT(config)
    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Memory before training
    mem_before = get_memory_usage()

    # Training loop
    model.train()
    iter_times = []
    losses = []

    print(f"\nTraining for {max_iters} iterations...")

    for iter_num in range(max_iters):
        # Get batch
        X, Y = get_batch('train')

        # Forward pass (time this)
        t0 = time.time()
        logits, loss = model(X, Y)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        t1 = time.time()
        iter_time = (t1 - t0) * 1000  # ms

        iter_times.append(iter_time)
        losses.append(loss.item())

        # Evaluation
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            eval_results = estimate_loss(model, eval_iters)
            print(f"iter {iter_num:4d}: train loss {eval_results['train']:.4f}, "
                  f"val loss {eval_results['val']:.4f}, time {iter_time:.2f}ms")

    # Memory after training
    mem_after = get_memory_usage()
    mem_used = mem_after - mem_before

    # Calculate statistics
    avg_time = sum(iter_times) / len(iter_times)
    final_train_loss = losses[-1]

    # Final evaluation
    final_eval = estimate_loss(model, eval_iters)

    results = {
        'name': config_name,
        'description': config_dict['description'],
        'parameters': n_params,
        'avg_iter_time_ms': float(avg_time),
        'final_train_loss': float(final_train_loss),
        'final_val_loss': float(final_eval['val'].item()) if torch.is_tensor(final_eval['val']) else float(final_eval['val']),
        'memory_mb': float(mem_used),
        'total_iters': max_iters,
        'config': {k: v for k, v in config_dict.items() if k not in ['name', 'description']}
    }

    return results


def print_comparison_table(all_results, baseline_name='Standard MHA'):
    """Print a comparison table like the image you showed"""
    print("\n" + "="*100)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*100)

    # Find baseline
    baseline = next((r for r in all_results if r['name'] == baseline_name), all_results[0])
    baseline_time = baseline['avg_iter_time_ms']
    baseline_loss = baseline['final_val_loss']
    baseline_params = baseline['parameters']

    # Print header
    print(f"\n{'Attention Type':<30} | {'Params':<10} | {'Time/iter':<12} | {'Val Loss':<10} | {'vs Baseline':<15}")
    print("-" * 100)

    # Print baseline
    print(f"{baseline['name']:<30} | {baseline_params/1e6:>8.2f}M | {baseline_time:>10.2f}ms | {baseline_loss:>10.4f} | {'baseline':<15}")

    # Print others
    for result in all_results:
        if result['name'] == baseline_name:
            continue

        name = result['name']
        params = result['parameters']
        time_ms = result['avg_iter_time_ms']
        val_loss = result['final_val_loss']

        # Calculate vs baseline
        time_diff = ((time_ms - baseline_time) / baseline_time) * 100
        loss_diff = ((val_loss - baseline_loss) / baseline_loss) * 100
        param_diff = ((params - baseline_params) / baseline_params) * 100

        time_str = f"{time_diff:+.0f}% time"

        print(f"{name:<30} | {params/1e6:>8.2f}M | {time_ms:>10.2f}ms | {val_loss:>10.4f} | {time_str:<15}")

    print("\n" + "="*100)
    print("\nKey Findings:")

    # Find best in each category
    fastest = min(all_results, key=lambda x: x['avg_iter_time_ms'])
    best_loss = min(all_results, key=lambda x: x['final_val_loss'])
    smallest = min(all_results, key=lambda x: x['parameters'])

    print(f"1. Fastest: {fastest['name']} ({fastest['avg_iter_time_ms']:.2f}ms/iter)")
    print(f"2. Best Loss: {best_loss['name']} (val loss {best_loss['final_val_loss']:.4f})")
    print(f"3. Smallest: {smallest['name']} ({smallest['parameters']/1e6:.2f}M params)")

    # Memory efficiency
    if all_results[0]['memory_mb'] > 0:
        lowest_mem = min(all_results, key=lambda x: x['memory_mb'])
        print(f"4. Lowest Memory: {lowest_mem['name']} ({lowest_mem['memory_mb']:.1f}MB)")


def main():
    print("="*70)
    print("NanoGPT Attention Mechanism Benchmark")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print(f"Iterations: {max_iters}")
    print(f"Model: {base_config['n_layer']} layers, {base_config['n_head']} heads, {base_config['n_embd']} embd")

    # Run benchmarks
    all_results = []

    for config_dict in attention_configs:
        try:
            result = benchmark_config(config_dict, config_dict['name'])
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to benchmark {config_dict['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print comparison table
    if all_results:
        print_comparison_table(all_results)

        # Save results to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"benchmark_results_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'device': device,
                'config': base_config,
                'results': all_results
            }, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
    else:
        print("\n❌ No successful benchmarks to compare")


if __name__ == "__main__":
    main()
