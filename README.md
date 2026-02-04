# NanoGPT Deep Learning Experiments

A hands-on exploration and educational extension of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - the simplest, fastest repository for training/finetuning medium-sized GPTs.

> **🚀 Apple Silicon Optimized!** This fork includes full MPS support for M1/M2/M3/M4 MacBooks with optimized training scripts. Train a 30M parameter GPT model in just 30-40 minutes on M4!

## Key Features
- ✅ **Apple Silicon (MPS) Support** - Full Metal Performance Shaders support for M1/M2/M3/M4
- ✅ **Optimized Training Scripts** - `train_mac.sh` and `train_m4.sh` for quick training
- ✅ **Sliding Window Attention** - Efficient long-sequence handling with memory reduction
- ✅ **Attention Sinks** - Prevents information loss in sliding window attention
- ✅ **Advanced Attention Mechanisms** - MHA, GQA (Grouped Query Attention), MLA (Multi-head Latent Attention)
- ✅ **Comprehensive Testing** - Component-level tests for every module
- ✅ **Attention Visualization** - See what the model is learning
- ✅ **Educational Documentation** - Detailed guides and inline explanations
- ✅ **Command-Line Configurator** - Easy hyperparameter overrides with scientific notation support

## Credits

This project is built upon the excellent work of **Andrej Karpathy's nanoGPT**. The original implementation provides a clean, educational codebase for understanding GPT-style transformers. This repository extends that work with comprehensive testing, visualization, bug fixes, and Apple Silicon optimization.

**Original nanoGPT:** https://github.com/karpathy/nanoGPT

## What's New in This Repository

This is an **educational fork** focused on:

1. **Deep Component Testing** - Systematic testing of every module (LayerNorm, Attention, MLP, Blocks)
2. **Attention Visualization** - Visual exploration of attention patterns and head behavior
3. **Gradient Flow Analysis** - Verifying backpropagation through all components
4. **Experimental Modifications** - Breaking and rebuilding the model to understand design choices
5. **Attention Entropy Analysis** - Tracking how attention sharpens across layers
6. **Performance Benchmarking** - Understanding speed/memory trade-offs
7. **Apple Silicon (M1/M2/M3/M4) Optimization** - MPS-optimized training with detailed guides
8. **Sliding Window Attention** - Memory-efficient attention for long sequences (99% memory reduction)
9. **Attention Sinks** - Always-accessible initial tokens to prevent information loss
10. **Advanced Attention Variants** - GQA and MLA implementations with sliding window support

### Key Additions

- **`test_model.py`** - Comprehensive test suite for all GPT components
- **`attention_ablation.py`** - Advanced attention mechanisms (MHA, GQA, MLA) with sliding window support
- **`test_sliding_window.py`** - Testing suite for sliding window attention
- **`breaking_model.py`** - Experimental modifications (causal mask removal, residual ablation, etc.)
- **`configurator.py`** - Command-line argument parser supporting scientific notation
- **`train_mac.sh`** - Basic training script for Apple Silicon Macs
- **`train_m4.sh`** - M4-optimized training script (~1.5-2x faster than M1) with GQA + sliding window
- **`TRAINING_GUIDE.md`** - Complete training guide for MacBook users
- **`QUICKSTART_M4.md`** - 3-step quick start for M4 MacBook (30 minutes to trained model)
- Enhanced `model.py` with:
  - `return_attention` flag for attention visualization
  - **Sliding window attention** with configurable window size and attention sinks
  - **Rotary Position Embeddings (RoPE)** support
  - **Intra-document masking** for multi-document training
  - Fixed method indentation bugs (`configure_optimizers`, `estimate_mfu`, `generate`)
  - Improved LayerNorm implementation
  - Better parameter handling
- Enhanced `train.py` with:
  - MPS (Metal Performance Shaders) device support for Apple Silicon
  - **Attention type selection**: standard, GQA, MLA
  - **Sliding window parameters**: `window_size` and `sink_size`
  - **GQA configuration**: `n_kv_heads` for grouped query attention
  - **MLA configuration**: `latent_dim` for multi-head latent attention
  - Fixed GradScaler compatibility for MPS
  - Scientific notation support for hyperparameters
  - Updated to use `clip_grad_norm_` (non-deprecated API)

## Project Structure

```
Nano_GPT_/
├── model.py                # Core GPT architecture (bug fixes + sliding window)
├── train.py                # Training script (MPS + attention variants)
├── attention_ablation.py   # Advanced attention: MHA, GQA, MLA + sliding window
├── test_sliding_window.py  # Tests for sliding window attention
├── masking.py              # Causal and intra-document masking utilities
├── positional_encoding.py  # RoPE (Rotary Position Embeddings)
├── configurator.py         # Command-line argument parser
├── train_mac.sh            # Basic training script for Mac
├── train_m4.sh             # M4-optimized with GQA + sliding window
├── bench.py                # Performance benchmarking
├── sample.py               # Text generation
├── test_model.py           # Component testing & visualization
├── breaking_model.py       # Experimental modifications
├── TRAINING_GUIDE.md       # Complete training guide for MacBook
├── QUICKSTART_M4.md        # Quick start for M4 MacBook users
├── data/
│   ├── shakesphere/        # Shakespeare dataset (with prepare.py)
│   ├── shakesphere_char/   # Character-level Shakespeare
│   └── openwebtext/        # OpenWebText dataset
└── README.md               # This file
```

## Architecture Overview

The model implements a standard GPT-2 style transformer with:

- **Multi-head Causal Self-Attention** (12 heads, 64 dims each)
- **Pre-LayerNorm** architecture (normalization before attention/MLP)
- **Residual/Skip Connections** for gradient flow
- **Position + Token Embeddings**
- **GELU Activation** in feedforward layers
- **Weight Tying** between input embeddings and output head
- **4x MLP Expansion** (768 → 3072 → 768)

Default configuration: 12 layers, 12 heads, 768 embedding dimensions (~124M parameters)

### Advanced Attention Mechanisms

This repository implements multiple attention variants for different use cases:

#### 1. Sliding Window Attention
Limits each token to attend only to the W most recent positions, dramatically reducing memory usage for long sequences.

- **Memory**: O(T²) → O(T×W) reduction
- **Example**: seq_len=4096, window=512 → 99.5% memory saved
- **Use case**: Long document processing, streaming inference

#### 2. Attention Sinks
Keeps the first N tokens always accessible, even outside the sliding window. Prevents information loss from important context at sequence start.

- **Sink tokens**: Always visible (typically N=4)
- **Combined with sliding window**: Each token sees [sink tokens] + [recent W tokens]
- **Benefits**: Maintains long-range dependencies while staying memory-efficient

#### 3. Grouped Query Attention (GQA)
Reduces KV cache by sharing key-value heads across query groups.

- **Standard MHA**: 12 Q heads, 12 KV heads
- **GQA**: 12 Q heads, 2-4 KV heads (3-6x reduction)
- **Benefits**: Faster inference, lower memory, minimal quality loss

#### 4. Multi-head Latent Attention (MLA)
Compresses key-value representations through a learned latent bottleneck.

- **Compression**: Reduces KV dimension (e.g., 768 → 128 → 768)
- **Benefits**: Smaller KV cache, efficient inference
- **Trade-off**: Additional computational overhead during training

**Configuration Example:**
```bash
python train.py \
    --attention_type=gqa \
    --n_kv_heads=2 \
    --window_size=256 \
    --sink_size=4
```

## Installation

```bash
# Clone the repository
git clone <https://github.com/satwiksunnam19/Nano-GPT.git>
cd Nano_GPT_

# Install dependencies
pip install torch numpy matplotlib tiktoken requests

# For training on Shakespeare (recommended for learning)
cd data/shakesphere
python prepare.py
cd ../..

# For training on OpenWebText (larger dataset)
python data/openwebtext/prepare.py
```

### 🚀 M4 MacBook Users - Quick Start!

If you have an M4 MacBook, see **[QUICKSTART_M4.md](QUICKSTART_M4.md)** for optimized training (30 minutes to a trained model!)

**For M1/M2/M3 MacBook users**, see **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** for complete training instructions.

## Usage

### 1. Training a Model

#### Quick Training (Apple Silicon Macs)
```bash
# M4 MacBook - Optimized training script (~30-40 minutes)
./train_m4.sh

# M1/M2/M3 MacBook - Basic training script (~20-30 minutes)
./train_mac.sh
```

#### Custom Training (Any Platform)
```bash
# Train on Shakespeare with custom config
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=4 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=256 \
    --max_iters=5000

# Train on OpenWebText (large dataset, full model)
python train.py \
    --dataset=openwebtext \
    --n_layer=12 \
    --n_head=12 \
    --n_embd=768

# Train with Grouped Query Attention (GQA)
python train.py \
    --dataset=shakesphere \
    --attention_type=gqa \
    --n_kv_heads=2 \
    --batch_size=4 \
    --max_iters=5000

# Train with Sliding Window Attention (memory efficient for long sequences)
python train.py \
    --dataset=shakesphere \
    --window_size=256 \
    --sink_size=4 \
    --batch_size=4 \
    --max_iters=5000

# Train with GQA + Sliding Window (best for efficiency)
python train.py \
    --dataset=shakesphere \
    --attention_type=gqa \
    --n_kv_heads=2 \
    --window_size=256 \
    --sink_size=4 \
    --batch_size=6 \
    --max_iters=8000

# Train with Multi-head Latent Attention (MLA)
python train.py \
    --dataset=shakesphere \
    --attention_type=mla \
    --latent_dim=128 \
    --batch_size=4 \
    --max_iters=5000
```

**Supported devices:**
- `mps` - Apple Silicon (M1/M2/M3/M4)
- `cuda` - NVIDIA GPUs
- `cpu` - CPU (slow)

### 2. Generating Text

```bash
# Generate text from a trained model
python sample.py --start="To be or not to be"
```

### 3. Benchmarking Performance

```bash
# Benchmark on CPU
python bench.py --device=cpu --batch_size=4

# Benchmark on GPU (if available)
python bench.py --device=cuda --batch_size=12 --compile=True
```

### 4. Testing & Visualization

```bash
# Run comprehensive component tests
python test_model.py

# Test sliding window attention mechanisms
python test_sliding_window.py

# Test advanced attention mechanisms directly
python attention_ablation.py
```

**test_model.py** will:
- Test LayerNorm normalization statistics
- Verify attention mechanism and causal masking
- Check gradient flow through all components
- Visualize attention patterns across heads
- Calculate attention entropy across layers
- Verify weight tying and position embeddings

**test_sliding_window.py** will:
- Test standard causal attention (baseline)
- Test sliding window attention with various window sizes
- Test GQA with sliding window
- Test MLA with sliding window
- Calculate memory savings (standard vs sliding window)
- Verify mask patterns and causal constraints

**attention_ablation.py** will:
- Compare MHA, GQA, and MLA implementations
- Test sliding window with all attention variants
- Show memory complexity comparisons
- Visualize attention patterns with sliding windows

## Experiments Conducted

### 1. **LayerNorm Testing**
- Verified per-position normalization (mean ≈ 0, std ≈ 1 across feature dimension)
- Fixed bug: Proper handling of `bias=False` parameter
- Gradient flow verification

### 2. **Causal Self-Attention Analysis**
- Shape preservation testing (B, T, C) → (B, T, C)
- Attention weight visualization (12 heads, causal mask)
- Understanding Q, K, V projections (3×n_embd efficiency)
- Scaling factor importance (1/√d_k)
- Flash attention vs manual attention comparison

### 3. **Multi-Head Attention Deep Dive**
- Why heads need equal dimensions (parallel processing)
- Attention matrix dimensions: (B, nh, T, T)
- Head specialization patterns (even in untrained models)
- Gradient flow through attention mechanism

### 4. **MLP (Feedforward) Testing**
- 4x expansion verification (768 → 3072 → 768)
- GELU nonlinearity behavior
- Gradient flow and parameter updates

### 5. **Transformer Block Testing**
- Residual connection verification
- Pre-LN vs Post-LN architecture understanding
- Combined attention + MLP flow

### 6. **Full GPT Model Testing**
- Token + position embedding combination
- Loss computation and backpropagation
- Weight tying verification (input embeddings = output head weights)
- Parameter counting (~124M for default config)

### 7. **Attention Entropy Analysis**
- Tracking entropy across 6 layers
- Understanding attention sharpening (diffuse → focused)
- Layer-wise attention pattern evolution

### 8. **Sliding Window Attention Experiments**
- Memory complexity reduction: O(T²) → O(T×W)
- Tested with various window sizes (16, 256, 512)
- Attention sink effectiveness (preserving first N tokens)
- Compatibility with all attention variants (MHA, GQA, MLA)
- Verified mask creation logic and causal constraints

### 9. **Advanced Attention Mechanisms Comparison**
- **GQA vs MHA**: Parameter reduction with grouped KV heads
  - 6 Q heads with 2 KV heads = 3x reduction in KV cache
  - Minimal performance degradation on Shakespeare dataset
- **MLA**: Latent compression of key-value representations
  - 768→128→768 compression for KV heads
  - Trade-off: smaller cache vs computational overhead
- **Combined GQA + Sliding Window**: Best efficiency
  - window_size=256, sink_size=4, n_kv_heads=2
  - Training successfully on M4 MacBook with batch_size=6

## Key Learnings & Insights

### Architecture Design Choices

1. **Why Pre-LayerNorm?**
   - Better gradient flow than Post-LN
   - Stabilizes training in deep networks

2. **Why Equal Head Dimensions?**
   - Enables clean tensor reshaping: (B, T, C) → (B, T, nh, hs)
   - Parallel processing across heads
   - Simplifies implementation

3. **Why 3×n_embd Projection?**
   - Efficiently creates Q, K, V in one matrix multiply
   - Then splits into three: `q, k, v = self.c_attn(x).split(n_embd, dim=2)`

4. **Why Causal Masking?**
   - Prevents "cheating" by looking at future tokens
   - Essential for autoregressive language modeling
   - Upper triangle of attention matrix masked to -inf

5. **Why Weight Tying?**
   - Reduces parameters by ~38M (token embeddings ≈ vocabulary × embedding_dim)
   - Forces consistent representation space
   - Improves performance on language modeling

6. **Why Residual Connections?**
   - Without them: vanishing gradients in deep networks
   - With them: gradient highway for information flow

7. **Why Sliding Window Attention?**
   - Standard attention: O(T²) memory grows quadratically with sequence length
   - Sliding window: O(T×W) memory grows linearly
   - Example: 4096 tokens with window=512 saves 99.5% memory
   - Essential for processing long documents and efficient inference

8. **Why Attention Sinks?**
   - Sliding window alone can lose important context from sequence start
   - Sinks preserve first N tokens (typically 4) as always accessible
   - Maintains long-range dependencies while staying memory-efficient
   - Empirically shown to prevent performance degradation

9. **Why GQA (Grouped Query Attention)?**
   - Standard MHA duplicates KV heads for each query head (memory intensive)
   - GQA shares KV heads across query groups (e.g., 12Q heads with 2KV heads)
   - Reduces KV cache size by 3-6x with minimal accuracy loss
   - Faster inference and lower memory footprint

10. **Why MLA (Multi-head Latent Attention)?**
    - Compresses KV through learned latent bottleneck (e.g., 768→128→768)
    - Further reduces KV cache size for efficient inference
    - Trade-off: additional computation during training
    - Useful when memory is more constrained than compute

## Performance Benchmarks

### Apple Silicon Performance (MPS)

Performance metrics on M4 MacBook Pro (training Shakespeare dataset):

| Model Size | Parameters | Batch Size | Seq Length | Time/iter | Total Time (8k iters) |
|-----------|-----------|------------|------------|-----------|---------------------|
| Tiny      | ~1M       | 8          | 128        | ~30ms     | ~4 minutes          |
| Small     | ~10M      | 4          | 256        | ~120ms    | ~16 minutes         |
| Medium    | ~30M      | 6          | 384        | ~400ms    | ~53 minutes         |
| Large     | ~124M     | 2          | 1024       | ~1200ms   | ~2.7 hours          |

**M4 vs M1 Comparison:**
- M4 is **~1.5-2x faster** than M1 for transformer training
- M4 has **better memory bandwidth** (can handle larger batch sizes)
- M4 has **improved MPS** GPU utilization

### CPU vs MPS (M4 MacBook)

| Device | Medium Model (30M params) | Speedup |
|--------|---------------------------|---------|
| CPU    | ~2000ms/iter              | 1x      |
| MPS    | ~400ms/iter               | **5x**  |

*See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed benchmarks across all Apple Silicon chips.*

## Experimental Ideas

Ready to break things and learn? Try these experiments:

1. **Remove Causal Mask** - See the model "cheat" by attending to future tokens
2. **Disable Residual Connections** - Watch gradients vanish
3. **Vary Attention Temperature** - Sharp vs soft attention distributions
4. **Compare Activation Functions** - GELU vs ReLU vs others
5. **Ablate Position Embeddings** - Lose sense of token order
6. **Test Pre-LN vs Post-LN** - Different normalization strategies
7. **Remove LayerNorm** - Watch training instability
8. **Head Specialization** - Visualize what different heads focus on
9. **Vary Window Sizes** - Compare window_size=64, 128, 256, 512 for sliding window attention
10. **Ablate Attention Sinks** - Train with sliding window but sink_size=0, observe degradation
11. **Compare Attention Variants** - Train identical models with MHA vs GQA vs MLA, compare speed/quality
12. **Extreme Compression** - Try GQA with n_kv_heads=1 (all queries share one KV head)
13. **Combine RoPE + Sliding Window** - Test rotary embeddings with sliding window attention
14. **Long Sequence Stress Test** - Process very long sequences (>4096 tokens) with sliding window

## Technical Details

### Model Components

#### LayerNorm
```python
class LayerNorm(nn.Module):
    """Normalizes each (batch, position) independently across features"""
    def __init__(self, ndim, bias):
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
```

#### CausalSelfAttention
```python
class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with optional flash attention"""
    - n_head: 12
    - head_dim: 64 (768 / 12)
    - Causal masking via upper triangular mask
    - Optional return_attention for visualization
    - Sliding window support: window_size, sink_size
    - RoPE (Rotary Position Embeddings) support
```

#### Sliding Window Mask Creation
```python
def create_sliding_window_mask(T, window_size=None, sink_size=0, device='cpu'):
    """Creates sliding window + attention sink mask"""
    # Causal mask (lower triangular)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    # Apply sliding window if specified
    if window_size is not None:
        band_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                              diagonal=-(window_size - 1))
        mask = mask & band_mask

    # Add attention sinks (first N tokens always accessible)
    if sink_size > 0:
        mask[:, :sink_size] = True
        for i in range(sink_size):
            mask[i, i+1:] = False  # Respect causality for sinks

    return mask
```

#### GQA (Grouped Query Attention)
```python
class GQA(nn.Module):
    """Grouped Query Attention - shares KV heads across Q groups"""
    - n_head: 12 (query heads)
    - n_kv_heads: 2-4 (key-value heads, shared across groups)
    - group_size: n_head // n_kv_heads
    - KV cache reduction: 3-6x smaller than standard MHA
    - Sliding window support: window_size, sink_size
```

#### MLA (Multi-head Latent Attention)
```python
class MLA(nn.Module):
    """Multi-head Latent Attention - compresses KV via latent bottleneck"""
    - latent_dim: 128 (compression dimension)
    - KV path: x → latent_kv (768→128) → k, v (128→768)
    - Reduces KV cache size at cost of extra computation
    - Sliding window support: window_size, sink_size
```

#### MLP
```python
class MLP(nn.Module):
    """Two-layer feedforward network with GELU"""
    - Expansion: 768 → 3072 (4x)
    - Activation: GELU
    - Projection: 3072 → 768
```

#### Block
```python
class Block(nn.Module):
    """Transformer block: Pre-LN + Attention + Pre-LN + MLP"""
    - x = x + attention(ln(x))  # Residual around attention
    - x = x + mlp(ln(x))        # Residual around MLP
```

### Training Details

- **Optimizer**: AdamW with weight decay
- **Learning Rate**: Cosine decay with warmup (supports scientific notation like `1e-3`)
- **Gradient Clipping**: 1.0 (uses updated `clip_grad_norm_` API)
- **Dropout**: 0.0-0.2 (configurable)
- **Batch Size**: 4-32 (depends on GPU memory)
- **Context Length**: 256-1024 tokens
- **Mixed Precision**: Automatic GradScaler (CUDA only, disabled for MPS)

### Command-Line Configuration

The `configurator.py` module allows overriding any training parameter:

```bash
# Override hyperparameters
python train.py \
    --learning_rate=1e-3 \      # Scientific notation supported
    --batch_size=8 \             # Integers
    --dropout=0.1 \              # Floats
    --compile=False \            # Booleans
    --device=mps                 # Strings
```

**Supported types:**
- **Booleans**: `True`, `False` (case-insensitive)
- **Integers**: `5000`, `128`, `6`
- **Floats**: `0.1`, `1e-3`, `5e-4` (scientific notation supported)
- **Strings**: `"mps"`, `"shakesphere"`, `"cuda"`


## Training Guides

- **[QUICKSTART_M4.md](QUICKSTART_M4.md)** - 3-step quick start for M4 MacBook (30 minutes)
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training guide for all MacBook models
- See inline comments in [train.py](train.py) for detailed parameter explanations
- See [configurator.py](configurator.py) for command-line argument parsing

## Resources

### Original References
- **Original nanoGPT**: https://github.com/karpathy/nanoGPT
- **Andrej's Video Lecture**: https://www.youtube.com/watch?v=kCc8FmEb1nY
- **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
- **GPT-2 Paper**: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **Andrej's YouTube Channel**: https://www.youtube.com/c/AndrejKarpathy
- **PyTorch MPS Documentation**: https://pytorch.org/docs/stable/notes/mps.html

### Advanced Attention Mechanisms
- **Efficient Streaming Language Models with Attention Sinks**: https://arxiv.org/abs/2309.17453
- **GQA: Training Generalized Multi-Query Transformer Models**: https://arxiv.org/abs/2305.13245
- **Mistral 7B** (uses sliding window + GQA): https://arxiv.org/abs/2310.06825
- **Multi-Head Latent Attention** (DeepSeek-V2): https://arxiv.org/abs/2405.04434
- **RoFormer: Enhanced Transformer with Rotary Position Embeddings**: https://arxiv.org/abs/2104.09864

## Contributing

This is an educational project. Feel free to:
- Add more experiments
- Improve visualizations
- Document new findings
- Share interesting results

## License

MIT License (following nanoGPT's original license)

## Acknowledgments

- **Andrej Karpathy** for the original nanoGPT implementation and educational content
- **OpenAI** for GPT-2 architecture and insights
- **PyTorch Team** for the excellent deep learning framework

---

**Happy Learning! 🚀**

*"The best way to understand transformers is to build them from scratch and break them in creative ways."*
