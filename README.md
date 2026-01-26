# NanoGPT Deep Learning Experiments

A hands-on exploration and educational extension of [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - the simplest, fastest repository for training/finetuning medium-sized GPTs.

> **🚀 Apple Silicon Optimized!** This fork includes full MPS support for M1/M2/M3/M4 MacBooks with optimized training scripts. Train a 30M parameter GPT model in just 30-40 minutes on M4!

## Key Features

- ✅ **Multiple Critical Bug Fixes** - Fixed indentation errors, path handling, GradScaler compatibility
- ✅ **Apple Silicon (MPS) Support** - Full Metal Performance Shaders support for M1/M2/M3/M4
- ✅ **Optimized Training Scripts** - `train_mac.sh` and `train_m4.sh` for quick training
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

### Key Additions

- **`test_model.py`** - Comprehensive test suite for all GPT components
- **`breaking_model.py`** - Experimental modifications (causal mask removal, residual ablation, etc.)
- **`configurator.py`** - Command-line argument parser supporting scientific notation
- **`train_mac.sh`** - Basic training script for Apple Silicon Macs
- **`train_m4.sh`** - M4-optimized training script (~1.5-2x faster than M1)
- **`TRAINING_GUIDE.md`** - Complete training guide for MacBook users
- **`QUICKSTART_M4.md`** - 3-step quick start for M4 MacBook (30 minutes to trained model)
- Enhanced `model.py` with:
  - `return_attention` flag for attention visualization
  - Fixed method indentation bugs (`configure_optimizers`, `estimate_mfu`, `generate`)
  - Improved LayerNorm implementation
  - Better parameter handling
- Enhanced `train.py` with:
  - MPS (Metal Performance Shaders) device support for Apple Silicon
  - Fixed GradScaler compatibility for MPS
  - Scientific notation support for hyperparameters
  - Updated to use `clip_grad_norm_` (non-deprecated API)

## Project Structure

```
Nano_GPT_/
├── model.py                # Core GPT architecture (bug fixes applied)
├── train.py                # Training script (MPS support added)
├── configurator.py         # Command-line argument parser
├── train_mac.sh            # Basic training script for Mac
├── train_m4.sh             # M4-optimized training script
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

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
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
```

This will:
- Test LayerNorm normalization statistics
- Verify attention mechanism and causal masking
- Check gradient flow through all components
- Visualize attention patterns across heads
- Calculate attention entropy across layers
- Verify weight tying and position embeddings

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

## Bug Fixes Applied

This repository includes fixes for several critical bugs found in the original implementation:

### 1. **Method Indentation Errors** (model.py)
- Fixed `configure_optimizers` - was at module level instead of class method
- Fixed `estimate_mfu` - was at module level instead of class method
- Fixed `generate` - was incorrectly indented (typo: `genrate` → `generate`)

### 2. **Path Handling** (train.py:125)
```python
# Before: data_dir= os.path('data',dataset)  # TypeError!
# After:  data_dir= os.path.join('data',dataset)
```

### 3. **Typo in Variable Name** (train.py:64, 214)
```python
# Before: weigh_decay= 1e-1
# After:  weight_decay= 1e-1
```

### 4. **GradScaler MPS Compatibility** (train.py:211-213)
```python
# Before: scaler= torch.cuda.amp.GradScaler(...)  # Only works on CUDA
# After:  scaler= torch.amp.GradScaler(device_type, enabled=(dtype=='float16' and device_type=='cuda'))
```

### 5. **Device Type Detection** (train.py:120)
```python
# Before: device_type='cuda' if 'cuda' in device else 'cpu'  # Didn't handle MPS
# After:  device_type='cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
```

### 6. **Deprecated Gradient Clipping** (train.py:327)
```python
# Before: torch.nn.utils.clip_grad_norm(...)  # Deprecated
# After:  torch.nn.utils.clip_grad_norm_(...)  # Correct API
```

### 7. **Scientific Notation Parsing** (configurator.py)
- Added proper parsing for scientific notation like `1e-3`, `5e-4`
- Handles integers, floats, booleans, and strings

## Common Issues & Solutions

### 1. LayerNorm Mean/Std Confusion
**Issue**: Computing global stats instead of per-position
```python
# Wrong: Global statistics
out.mean()  # Returns single value

# Correct: Per-position statistics
out.mean(dim=-1)  # Returns (B, T) tensor
```

### 2. Attention Visualization
**Issue**: Flash attention doesn't return weights
```python
# Force manual attention
attn.flash = False
output, att_weights = attn(x, return_attention=True)
```

### 3. Block Size Mismatch
**Issue**: `RuntimeError: size of tensor a (4) must match size b (16)`
```python
# Ensure sequence length <= block_size
config = Config(block_size=16)  # Must be >= sequence length
idx = torch.randint(0, vocab_size, (batch, 16))
```

### 4. MPS Out of Memory (Apple Silicon)
**Issue**: `RuntimeError: MPS backend out of memory`
```bash
# Solution 1: Reduce batch size
python train.py --batch_size=2  # or even 1

# Solution 2: Reduce model size
python train.py --n_layer=4 --n_embd=256

# Solution 3: Reduce sequence length
python train.py --block_size=256
```

### 5. Training Very Slow on Mac
**Issue**: Not using MPS acceleration
```bash
# Check if MPS is available
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"

# Should print: MPS: True
# If False, reinstall PyTorch with MPS support
conda install pytorch torchvision torchaudio -c pytorch
```

## Training Guides

- **[QUICKSTART_M4.md](QUICKSTART_M4.md)** - 3-step quick start for M4 MacBook (30 minutes)
- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - Complete training guide for all MacBook models
- See inline comments in [train.py](train.py) for detailed parameter explanations
- See [configurator.py](configurator.py) for command-line argument parsing

## Resources

- **Original nanoGPT**: https://github.com/karpathy/nanoGPT
- **Andrej's Video Lecture**: https://www.youtube.com/watch?v=kCc8FmEb1nY
- **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
- **GPT-2 Paper**: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- **Andrej's YouTube Channel**: https://www.youtube.com/c/AndrejKarpathy
- **PyTorch MPS Documentation**: https://pytorch.org/docs/stable/notes/mps.html

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
