"""
Option C: Early Signal at Full Scale Validation
Run all 4 attention mechanisms at 124M params for 500 iterations
Check if validation loss ranking holds from small-scale ablation

Expected ranking from 17M ablation:
1. MQA (best val loss)
2. GQA ratio 4
3. GQA ratio 2
4. MHA (worst val loss)

If ranking holds → confidence in small-scale results
If ranking flips → need deeper investigation
"""

import torch
import time
import os
import json
import numpy as np
from contextlib import nullcontext
from model import GPTConfig, GPT
from datetime import datetime
import wandb


def calculate_gpt_params(n_layer, n_embd, n_head, n_kv_heads, vocab_size, block_size, attention_type='standard'):
    """Calculate approximate GPT parameter count."""
    # Embeddings (shared)
    params = vocab_size * n_embd  # token embeddings
    params += block_size * n_embd  # position embeddings

    # Per-layer parameters
    for _ in range(n_layer):
        # LayerNorm 1
        params += 2 * n_embd

        # Attention projections
        if attention_type == 'standard' or attention_type == 'mha':
            # MHA: full Q, K, V
            params += 3 * n_embd * n_embd  # Q, K, V
        elif attention_type == 'gqa':
            # GQA: full Q, reduced K,V
            head_dim = n_embd // n_head
            params += n_embd * n_embd  # Q projection
            params += n_kv_heads * head_dim * n_embd  # K projection
            params += n_kv_heads * head_dim * n_embd  # V projection

        # Attention output projection
        params += n_embd * n_embd

        # LayerNorm 2
        params += 2 * n_embd

        # MLP (4x expansion)
        params += 4 * n_embd * n_embd  # up projection
        params += 4 * n_embd * n_embd  # down projection

    # Final LayerNorm
    params += 2 * n_embd

    return params


def find_optimal_layers(target_params, n_embd, n_head, n_kv_heads, vocab_size, block_size,
                       attention_type='standard', tolerance=0.02):
    """Find optimal number of layers to match target parameter count."""
    # Binary search for optimal layers
    low, high = 1, 50
    best_n_layer = low
    best_diff = float('inf')

    while low <= high:
        mid = (low + high) // 2
        params = calculate_gpt_params(mid, n_embd, n_head, n_kv_heads, vocab_size, block_size, attention_type)
        diff = abs(params - target_params)

        if diff < best_diff:
            best_diff = diff
            best_n_layer = mid

        if params < target_params:
            low = mid + 1
        else:
            high = mid - 1

    # Verify we're within tolerance
    final_params = calculate_gpt_params(best_n_layer, n_embd, n_head, n_kv_heads, vocab_size, block_size, attention_type)
    deviation = abs(final_params - target_params) / target_params

    if deviation > tolerance:
        print(f"Warning: Could not match target params within {tolerance*100}% tolerance")
        print(f"  Target: {target_params:,}, Achieved: {final_params:,}, Deviation: {deviation*100:.1f}%")

    return best_n_layer

# Hyperparameters for 124M scale
batch_size = 4  # Reduced for MPS memory safety at 124M scale
block_size = 384  # Keep smaller for MPS (original 1024 may OOM)
max_iters = 500  # EARLY SIGNAL: just 500 iterations (~10% of full run)
eval_interval = 50  # Evaluate frequently to catch early trends
eval_iters = 50
log_interval = 25
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
learning_rate = 3e-4

# 124M scale configuration (based on GPT-2 124M)
base_model_config = {
    'n_head': 12,  # GPT-2 124M standard (allows ratios: 1, 2, 3, 4, 6, 12)
    'n_embd': 768,  # GPT-2 124M standard
    'dropout': 0.2,
    'bias': False,
    'vocab_size': 50304,
    'block_size': block_size,
}

# Define baseline: Standard MHA with 12 layers (GPT-2 124M architecture)
BASELINE_N_LAYER = 12
BASELINE_ATTENTION = 'standard'
BASELINE_N_KV_HEADS = 12

# Calculate target parameter count from baseline (should be ~124M)
target_params = calculate_gpt_params(
    BASELINE_N_LAYER,
    base_model_config['n_embd'],
    base_model_config['n_head'],
    BASELINE_N_KV_HEADS,
    base_model_config['vocab_size'],
    base_model_config['block_size'],
    BASELINE_ATTENTION
)

print(f"\n{'='*80}")
print(f"OPTION C: EARLY SIGNAL AT FULL SCALE (124M params)")
print(f"{'='*80}")
print(f"Target Parameter Count: {target_params:,} ({target_params/1e6:.1f}M)")
print(f"Baseline: MHA with 12 heads, 12 layers (GPT-2 124M architecture)")
print(f"Iterations: {max_iters} (early signal validation)")
print(f"Expected Ranking from 17M ablation: MQA > GQA ratio 4 > GQA ratio 2 > MHA")
print(f"{'='*80}\n")

# Calculate optimal layers for each attention variant to match ~124M params
print("Calculating parameter-constant configurations for 124M scale...")
print(f"{'Attention Type':<25} | {'Q Heads':<8} | {'KV Heads':<9} | {'Ratio':<8} | {'Layers':<8} | {'Params':<12}")
print("-" * 90)

attention_configs = []

# 1. MHA (baseline - 12 layers)
params_mha = calculate_gpt_params(
    BASELINE_N_LAYER, base_model_config['n_embd'], base_model_config['n_head'],
    BASELINE_N_KV_HEADS, base_model_config['vocab_size'], base_model_config['block_size'], 'standard'
)
print(f"{'MHA':<25} | {base_model_config['n_head']:<8} | {base_model_config['n_head']:<9} | {1:<8} | {BASELINE_N_LAYER:<8} | {params_mha:>10,}  ← BASELINE")
attention_configs.append({
    'name': 'MHA',
    'n_layer': BASELINE_N_LAYER,
    'attention_type': 'standard',
    'description': f'Multi-Head Attention (12 KV heads, {BASELINE_N_LAYER} layers)'
})

# 2. GQA ratio 2: 6 KV heads
n_layer_gqa2 = find_optimal_layers(
    target_params, base_model_config['n_embd'], base_model_config['n_head'],
    6, base_model_config['vocab_size'], base_model_config['block_size'], 'gqa'
)
params_gqa2 = calculate_gpt_params(
    n_layer_gqa2, base_model_config['n_embd'], base_model_config['n_head'],
    6, base_model_config['vocab_size'], base_model_config['block_size'], 'gqa'
)
print(f"{'GQA (ratio 2)':<25} | {base_model_config['n_head']:<8} | {6:<9} | {base_model_config['n_head']//6:<8} | {n_layer_gqa2:<8} | {params_gqa2:>10,}")
attention_configs.append({
    'name': 'GQA (ratio 2)',
    'n_layer': n_layer_gqa2,
    'attention_type': 'gqa',
    'n_kv_heads': 6,
    'description': f'GQA with 6 KV heads (ratio 2, {n_layer_gqa2} layers)'
})

# 3. GQA ratio 4: 3 KV heads
n_layer_gqa4 = find_optimal_layers(
    target_params, base_model_config['n_embd'], base_model_config['n_head'],
    3, base_model_config['vocab_size'], base_model_config['block_size'], 'gqa'
)
params_gqa4 = calculate_gpt_params(
    n_layer_gqa4, base_model_config['n_embd'], base_model_config['n_head'],
    3, base_model_config['vocab_size'], base_model_config['block_size'], 'gqa'
)
print(f"{'GQA (ratio 4)':<25} | {base_model_config['n_head']:<8} | {3:<9} | {base_model_config['n_head']//3:<8} | {n_layer_gqa4:<8} | {params_gqa4:>10,}")
attention_configs.append({
    'name': 'GQA (ratio 4)',
    'n_layer': n_layer_gqa4,
    'attention_type': 'gqa',
    'n_kv_heads': 3,
    'description': f'GQA with 3 KV heads (ratio 4, {n_layer_gqa4} layers)'
})

# 4. MQA: 1 KV head
n_layer_mqa = find_optimal_layers(
    target_params, base_model_config['n_embd'], base_model_config['n_head'],
    1, base_model_config['vocab_size'], base_model_config['block_size'], 'gqa'
)
params_mqa = calculate_gpt_params(
    n_layer_mqa, base_model_config['n_embd'], base_model_config['n_head'],
    1, base_model_config['vocab_size'], base_model_config['block_size'], 'gqa'
)
print(f"{'MQA':<25} | {base_model_config['n_head']:<8} | {1:<9} | {base_model_config['n_head']//1:<8} | {n_layer_mqa:<8} | {params_mqa:>10,}")
attention_configs.append({
    'name': 'MQA',
    'n_layer': n_layer_mqa,
    'attention_type': 'gqa',
    'n_kv_heads': 1,
    'description': f'Multi-Query Attention (1 KV head, {n_layer_mqa} layers)'
})

print("-" * 90)
print()


def get_batch(split, data_dir='data/shakesphere'):
    """Load a batch of data"""
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


def benchmark_config(config_dict, config_name, base_config):
    """Benchmark a single attention configuration with W&B tracking"""
    n_layer = config_dict.get('n_layer', 12)
    n_kv_heads = config_dict.get('n_kv_heads', base_config['n_head'])
    ratio = base_config['n_head'] // n_kv_heads if n_kv_heads > 0 else 1

    print(f"\n{'='*80}")
    print(f"Benchmarking: {config_name}")
    print(f"Description: {config_dict['description']}")
    print(f"Layers: {n_layer}, KV Heads: {n_kv_heads}, Ratio: {ratio}")
    print(f"{'='*80}")

    # Create model config
    full_config = {**base_config, **{k: v for k, v in config_dict.items() if k not in ['name', 'description']}}
    config = GPTConfig(**full_config)

    # Create model
    model = GPT(config)
    model.to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Initialize W&B run
    run = wandb.init(
        project="nanogpt-124m-early-signal",
        name=f"{config_name} (124M, 500 iters)",
        config={
            "attention_type": config_dict.get('attention_type', 'standard'),
            "n_params": n_params,
            "n_layer": full_config['n_layer'],
            "n_head": full_config['n_head'],
            "n_embd": full_config['n_embd'],
            "batch_size": batch_size,
            "block_size": block_size,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "validation_type": "early_signal_full_scale",
            **{k: v for k, v in config_dict.items() if k not in ['name', 'description']}
        },
        reinit=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Memory before training
    mem_before = get_memory_usage()

    # Training loop
    model.train()
    iter_times = []
    losses = []
    tokens_consumed = 0

    print(f"\nTraining for {max_iters} iterations (early signal validation)...")

    for iter_num in range(max_iters):
        # Get batch
        X, Y = get_batch('train')

        # Forward pass
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

        # Track tokens consumed
        tokens_consumed += batch_size * block_size

        # Evaluation and W&B logging
        if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
            eval_results = estimate_loss(model, eval_iters)
            train_loss = eval_results['train'].item() if torch.is_tensor(eval_results['train']) else eval_results['train']
            val_loss = eval_results['val'].item() if torch.is_tensor(eval_results['val']) else eval_results['val']

            # Log to W&B
            wandb.log({
                'iter': iter_num,
                'tokens_millions': tokens_consumed / 1e6,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'generalization_gap': val_loss - train_loss,
                'iter_time_ms': iter_time,
                'memory_mb': get_memory_usage() - mem_before
            })

            print(f"iter {iter_num:4d}: train loss {train_loss:.4f}, "
                  f"val loss {val_loss:.4f}, gap {val_loss - train_loss:.4f}, tokens {tokens_consumed/1e6:.2f}M")

        # Log training loss more frequently
        elif iter_num % log_interval == 0:
            wandb.log({
                'iter': iter_num,
                'tokens_millions': tokens_consumed / 1e6,
                'train_loss_step': loss.item(),
                'iter_time_ms': iter_time
            })

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
        'final_train_loss': float(final_eval['train'].item()) if torch.is_tensor(final_eval['train']) else float(final_eval['train']),
        'final_val_loss': float(final_eval['val'].item()) if torch.is_tensor(final_eval['val']) else float(final_eval['val']),
        'generalization_gap': float(final_eval['val'].item() - final_eval['train'].item()),
        'memory_mb': float(mem_used),
        'total_iters': max_iters,
        'tokens_consumed': tokens_consumed,
        'config': {k: v for k, v in config_dict.items() if k not in ['name', 'description']}
    }

    # Log final summary
    wandb.summary['final_train_loss'] = results['final_train_loss']
    wandb.summary['final_val_loss'] = results['final_val_loss']
    wandb.summary['generalization_gap'] = results['generalization_gap']
    wandb.summary['avg_iter_time_ms'] = results['avg_iter_time_ms']
    wandb.summary['total_tokens_millions'] = tokens_consumed / 1e6

    # Finish W&B run
    wandb.finish()

    return results


def print_comparison_table(all_results):
    """Print comparison table for early signal validation"""
    print("\n" + "="*130)
    print("OPTION C: EARLY SIGNAL AT FULL SCALE (124M params, 500 iterations)")
    print("Validation of 17M ablation results at full scale")
    print("="*130)

    print(f"\n{'Attention Type':<20} | {'Layers':<7} | {'KV Heads':<9} | {'Params':<11} | {'Val Loss':<10} | {'Gen Gap':<10} | {'Time/iter':<11}")
    print("-" * 130)

    # Sort by validation loss (best to worst)
    for result in sorted(all_results, key=lambda x: x['final_val_loss']):
        n_layer = result['config'].get('n_layer', 'N/A')
        n_kv_heads = result['config'].get('n_kv_heads', 12)

        print(f"{result['name']:<20} | {n_layer:>6} | {n_kv_heads:>8} | {result['parameters']/1e6:>9.1f}M | "
              f"{result['final_val_loss']:>10.4f} | {result['generalization_gap']:>10.4f} | "
              f"{result['avg_iter_time_ms']:>9.2f}ms")

    print("\n" + "="*130)
    print("\n🔍 Ranking Analysis:")

    # Check if ranking matches 17M ablation
    sorted_by_val = sorted(all_results, key=lambda x: x['final_val_loss'])
    ranking = [r['name'] for r in sorted_by_val]

    print(f"\nActual ranking at 124M scale: {' > '.join(ranking)}")
    print(f"Expected from 17M ablation: MQA > GQA (ratio 4) > GQA (ratio 2) > MHA")

    # Check if MQA still wins
    if ranking[0] == 'MQA':
        print("\n✅ RANKING HOLDS: MQA still achieves best validation loss at 124M scale!")
        print("   → Small-scale ablation successfully predicts full-scale performance")
        print("   → Proceed with MQA for production implementation")
    else:
        print(f"\n⚠️  RANKING CHANGED: {ranking[0]} wins at 124M scale (not MQA)")
        print("   → Small-scale results did NOT transfer to full scale")
        print("   → Need deeper investigation before production decision")


def main():
    print("\n" + "="*80)
    print("OPTION C: EARLY SIGNAL AT FULL SCALE VALIDATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Batch size: {batch_size} (reduced for 124M scale)")
    print(f"Block size: {block_size}")
    print(f"Iterations: {max_iters} (10% of full run)")
    print(f"Learning rate: {learning_rate}")
    print(f"\n🎯 Goal: Validate if 17M ablation ranking transfers to 124M scale")
    print(f"Expected: MQA > GQA ratio 4 > GQA ratio 2 > MHA")
    print()

    # Run ablation study across all attention mechanisms
    all_results = []
    total_runs = len(attention_configs)
    current_run = 0

    for config_dict in attention_configs:
        current_run += 1
        print(f"\n{'='*80}")
        print(f"Run {current_run}/{total_runs}: {config_dict['name']}")
        print(f"{'='*80}")

        try:
            result = benchmark_config(
                config_dict,
                config_dict['name'],
                base_model_config
            )
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
        output_file = f"early_signal_124m_{timestamp}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'device': device,
                'validation_type': 'early_signal_full_scale',
                'target_params': target_params,
                'base_model_config': base_model_config,
                'attention_configs': attention_configs,
                'results': all_results,
                'expected_ranking': 'MQA > GQA (ratio 4) > GQA (ratio 2) > MHA'
            }, f, indent=2)

        print(f"\n✅ Results saved to: {output_file}")
        print(f"📊 View results on W&B: https://wandb.ai/[your-username]/nanogpt-124m-early-signal")
    else:
        print("\n❌ No successful benchmarks to compare")


if __name__ == "__main__":
    main()
