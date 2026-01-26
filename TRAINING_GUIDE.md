# Training NanoGPT on MacBook

Complete guide for training NanoGPT on MacBook with Apple Silicon (M1/M2/M3/M4).

## Your System: M4 MacBook 🚀

**You have the latest M4 chip!** This means:
- ⚡ **~1.5-2x faster** than M1 for deep learning
- 🧠 **Better Neural Engine** for optimized operations
- 💾 **Improved memory bandwidth** - can train larger models
- 🎯 **Better MPS (Metal Performance Shaders)** support

**Recommendation:** You can train **medium-sized models** comfortably that would struggle on M1!

## Prerequisites

✅ You already have:
- PyTorch 2.9.1 with MPS support
- Python 3.10.16 in conda environment `PyTorch`
- M4 MacBook (excellent for training!)
- All necessary dependencies

## Quick Start (5 minutes)

### Step 1: Prepare the Shakespeare dataset

```bash
# Activate your conda environment
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate PyTorch

# Download and prepare Shakespeare data
cd data/shakesphere
python prepare.py
```

This will:
- Download tiny shakespeare dataset (~1MB)
- Tokenize using GPT-2 BPE tokenizer
- Create `train.bin` (~300k tokens) and `val.bin` (~36k tokens)

### Step 2: Train a model

**For M4 MacBook (recommended):**
```bash
# Return to project root
cd ../..

# Make training script executable
chmod +x train_m4.sh

# Start training with M4-optimized settings!
./train_m4.sh
```

This trains a **medium GPT model** (~40M parameters) that should:
- Take ~20-30 seconds per 500 iterations on M4 Mac
- Reach good results in ~30-40 minutes
- Use ~4-6GB RAM

**For slower/basic training:**
```bash
chmod +x train_mac.sh
./train_mac.sh
```

This trains a **small GPT model** (~10M parameters) in just ~10 minutes on M4!

### Step 3: Monitor training

You'll see output like:
```
iter 0: loss 11.0234, time 1234.56ms
iter 10: loss 8.2341, time 456.78ms
iter 20: loss 6.5432, time 432.10ms
...
iter 500: train loss 2.3456, val loss 2.4567
```

**What to watch:**
- Loss should decrease from ~11 → ~2 (lower is better)
- Training time stabilizes after first few iterations
- Validation loss should be close to training loss (not too much higher)

## Configuration Options

### 🎯 Recommended for M4 MacBook

**Medium Model (sweet spot for M4):**
```bash
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=6 \
    --block_size=384 \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --max_iters=8000
```
- **Parameters:** ~40M
- **Time:** ~30-40 minutes on M4
- **Quality:** Very good Shakespeare-like text
- **This is what `train_m4.sh` uses!**

---

### Other Configurations

### Tiny Model (fastest, for testing)
```bash
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=8 \
    --block_size=128 \
    --n_layer=2 \
    --n_head=2 \
    --n_embd=128 \
    --max_iters=2000
```
- **Parameters:** ~1M
- **Time:** ~5-10 minutes
- **Quality:** Poor but fast

### Small Model (recommended)
```bash
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=4 \
    --block_size=256 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=256 \
    --max_iters=5000
```
- **Parameters:** ~10M
- **Time:** ~20-30 minutes
- **Quality:** Decent Shakespeare-like text

### Medium Model (patient training)
```bash
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=2 \
    --block_size=512 \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --max_iters=10000
```
- **Parameters:** ~40M
- **Time:** ~2-3 hours
- **Quality:** Good Shakespeare-like text

### Large Model (overnight training)
```bash
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=1 \
    --block_size=1024 \
    --n_layer=12 \
    --n_head=12 \
    --n_embd=768 \
    --max_iters=20000
```
- **Parameters:** ~124M (full GPT-2 Small size)
- **Time:** ~8-12 hours
- **Quality:** High-quality Shakespeare text
- **Warning:** May run out of memory on 8GB Macs

## Parameter Explanations

| Parameter | What it does | Impact |
|-----------|-------------|--------|
| `--device=mps` | Use Apple Silicon GPU | 5-10x faster than CPU |
| `--compile=False` | Disable PyTorch 2.0 compile | Avoid MPS compatibility issues |
| `--batch_size=4` | Process 4 sequences at once | Higher = faster but more memory |
| `--block_size=256` | Sequence length (context) | Higher = better but slower |
| `--n_layer=4` | Number of transformer layers | More = smarter but slower |
| `--n_head=4` | Number of attention heads | Usually = n_layer |
| `--n_embd=256` | Embedding dimension | Higher = more expressive |
| `--max_iters=5000` | Training iterations | More = better (diminishing returns) |
| `--learning_rate=1e-3` | How fast model learns | Too high = unstable, too low = slow |
| `--dropout=0.1` | Regularization (prevent overfitting) | 0.0 for small datasets, 0.1+ for fine-tuning |

## Troubleshooting

### Problem: "RuntimeError: MPS backend out of memory"

**Solution:** Reduce memory usage
```bash
# Try smaller batch size
--batch_size=2  # or even 1

# OR smaller model
--n_layer=2 --n_head=2 --n_embd=128

# OR shorter sequences
--block_size=128
```

### Problem: Training is very slow

**Solution:** Check device
```bash
# Make sure you're using MPS, not CPU
python -c "import torch; print(torch.backends.mps.is_available())"

# Should print: True
```

If False, your PyTorch doesn't have MPS. Reinstall:
```bash
conda install pytorch torchvision torchaudio -c pytorch
```

### Problem: "CUDA not available" error

**Solution:** Change device from cuda to mps
```bash
# In train.py line 79, change:
device='mps'  # instead of 'cuda'

# Or use command line:
--device=mps
```

### Problem: Loss is NaN or increasing

**Solution:** Reduce learning rate
```bash
--learning_rate=5e-4  # or even 1e-4
```

### Problem: "No module named tiktoken"

**Solution:** Install missing dependency
```bash
conda activate PyTorch
pip install tiktoken requests
```

## Generate Text After Training

Once training is complete, generate Shakespeare-like text:

```bash
python sample.py \
    --out_dir=out-shakespeare-small \
    --start="To be or not to be" \
    --num_samples=3 \
    --max_new_tokens=200
```

You should see output like:
```
To be or not to be: that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
...
```

## Performance Benchmarks

### M4 MacBook (Your System!) 🚀

The M4 chip is significantly faster than M1, with improved Neural Engine and GPU cores:

| Model Size | Parameters | Memory | Time/iter | Total Time (5k iters) |
|------------|-----------|--------|-----------|----------------------|
| Tiny       | ~1M       | ~1GB   | ~30ms     | ~3 minutes           |
| Small      | ~10M      | ~2GB   | ~120ms    | ~10 minutes          |
| Medium     | ~40M      | ~4GB   | ~400ms    | ~35 minutes          |
| Large      | ~124M     | ~8GB   | ~1200ms   | ~2 hours             |

**M4 Advantages:**
- **~1.5-2x faster** than M1 for transformer training
- **Better memory bandwidth** - can handle larger batch sizes
- **Improved MPS performance** - better GPU utilization 

### M1 MacBook Pro (Reference)

| Model Size | Parameters | Memory | Time/iter | Total Time (5k iters) |
|------------|-----------|--------|-----------|----------------------|
| Tiny       | ~1M       | ~1GB   | ~50ms     | ~5 minutes           |
| Small      | ~10M      | ~2GB   | ~200ms    | ~20 minutes          |
| Medium     | ~40M      | ~4GB   | ~600ms    | ~1 hour              |
| Large      | ~124M     | ~8GB   | ~2000ms   | ~3 hours             |

## Advanced: Training on Your Own Data

### Step 1: Prepare your text file

```bash
# Create a new dataset directory
mkdir -p data/mydata

# Put your text in input.txt
# (Any plain text file, minimum ~100KB recommended)
cp /path/to/your/text.txt data/mydata/input.txt
```

### Step 2: Create prepare.py

```bash
# Copy the Shakespeare prepare script
cp data/shakesphere/prepare.py data/mydata/prepare.py
```

### Step 3: Prepare the data

```bash
cd data/mydata
python prepare.py
cd ../..
```

### Step 4: Train on your data

```bash
python train.py \
    --dataset=mydata \
    --device=mps \
    --compile=False \
    --batch_size=4 \
    --n_layer=4 \
    --n_head=4 \
    --n_embd=256
```

## Tips for Best Results

1. **Start Small**: Always test with tiny model first to ensure everything works
2. **Monitor Validation Loss**: Should be close to training loss (within ~0.5)
3. **Patience**: Good results take time (thousands of iterations)
4. **Learning Rate**: Most important hyperparameter - tune if loss is unstable
5. **Overfitting**: If val_loss >> train_loss, increase dropout or get more data
6. **Early Stopping**: Stop training when validation loss stops improving

## Checkpoint Management

Models are saved in `out_dir/` (default: `out/`):

```
out-shakespeare-small/
├── ckpt.pt           # Latest checkpoint
└── config.json       # Model configuration
```

To resume training:
```bash
python train.py --init_from=resume --out_dir=out-shakespeare-small
```

## Next Steps

1. ✅ Train a tiny model (~5 min) to verify everything works
2. ✅ Train a small model (~30 min) for decent results
3. ✅ Generate samples and see what your model learned
4. ✅ Experiment with different hyperparameters
5. ✅ Try training on your own dataset

## Resources

- **Train.py documentation**: See comments in `train.py`
- **Model architecture**: See `model.py` and `README.md`
- **Original nanoGPT**: https://github.com/karpathy/nanoGPT
- **Andrej's lecture**: https://www.youtube.com/watch?v=kCc8FmEb1nY

---

**Happy Training! 🚀**

Remember: Start small, verify it works, then scale up!
