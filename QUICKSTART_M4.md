# 🚀 Quick Start for M4 MacBook

**You have an M4 Mac - the fastest Apple Silicon for deep learning!**

## 3-Step Training (30 minutes to trained model)

### Step 1: Install dependencies (1 minute)
```bash
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate PyTorch
pip install tiktoken requests
```

### Step 2: Prepare data (2 minutes)
```bash
cd data/shakesphere
python prepare.py
cd ../..
```

### Step 3: Train! (30-40 minutes)
```bash
chmod +x train_m4.sh
./train_m4.sh
```

That's it! ✨

## What's happening?

The M4-optimized script trains a **40M parameter GPT model** on Shakespeare text:
- **6 transformer layers** (deeper than basic tutorials)
- **384 embedding dimensions** (more expressive)
- **Batch size 6** (M4 can handle it!)
- **8000 iterations** (thorough training)

## Expected output:

```
iter 0: loss 11.0234, time 1234ms
iter 10: loss 8.2341, time 400ms
iter 100: loss 4.5678, time 380ms
iter 500: train loss 2.3456, val loss 2.4567
iter 1000: train loss 1.8234, val loss 1.9123
...
iter 8000: train loss 1.2345, val loss 1.3456
```

**Loss ~1.3 = Good Shakespeare quality!**

## Generate text:

```bash
python sample.py \
    --out_dir=out-shakespeare-m4 \
    --start="To be or not to be" \
    --num_samples=5 \
    --max_new_tokens=300
```

## Why M4 is awesome for this:

| Feature | M1 | M4 | Benefit |
|---------|----|----|---------|
| Training Speed | 1x | ~1.8x | Train in half the time! |
| Max Batch Size | 4 | 6-8 | Better gradient estimates |
| Model Size | ~10M params | ~40M params | Better text quality |
| Memory Bandwidth | 68 GB/s | 120 GB/s | Faster data loading |

## Too slow? Quick test version (5 minutes):

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
    --max_iters=1000
```

## Too fast? Go bigger! (2 hours):

```bash
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=4 \
    --block_size=768 \
    --n_layer=8 \
    --n_head=8 \
    --n_embd=512 \
    --max_iters=15000 \
    --out_dir=out-shakespeare-large
```

This trains a **~80M parameter model** with even better quality!

## Troubleshooting:

**"Out of memory"?**
```bash
# Reduce batch size
--batch_size=4  # or even 2
```

**Too slow?**
```bash
# Check you're using MPS
python -c "import torch; print(f'MPS: {torch.backends.mps.is_available()}')"
# Should print: MPS: True
```

**Loss not decreasing?**
```bash
# Reduce learning rate
--learning_rate=5e-4
```

## Next Steps:

1. ✅ Train your first model (30 min)
2. ✅ Generate Shakespeare text
3. ✅ Read `TRAINING_GUIDE.md` for advanced options
4. ✅ Train on your own data!
5. ✅ Experiment with hyperparameters

---

**Your M4 Mac is perfect for learning deep learning! Enjoy! 🎉**
