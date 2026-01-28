#!/bin/bash
# Training script optimized for M4 MacBook with MPS
# Takes advantage of M4's superior performance for larger models

# Activate conda environment
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate PyTorch

echo "🚀 Starting NanoGPT training on M4 MacBook..."
echo "Device: MPS (Apple Silicon M4 GPU)"
echo "Dataset: Shakespeare"
echo ""
echo "M4 is ~1.5-2x faster than M1! 🔥"
echo ""

# M4-optimized settings: Can handle larger models!
python train.py \
    --device=mps \
    --compile=False \
    --dataset=shakesphere \
    --batch_size=6 \
    --block_size=384 \
    --n_layer=6 \
    --n_head=6 \
    --n_embd=384 \
    --dropout=0.1 \
    --learning_rate=1e-3 \
    --max_iters=8000 \
    --eval_interval=500 \
    --eval_iters=100 \
    --log_interval=10 \
    --gradient_acculation_steps=1 \
    --warmup_iters=200 \
    --lr_decay_iters=8000 \
    --out_dir=out-shakespeare-m4\
    --attention_type=gqa\ 
    --n_kv_heads=4\

echo ""
echo "✅ Training complete! Model saved in: out-shakespeare-m4/"
echo "🎭 Generate Shakespeare text with:"
echo "   python sample.py --out_dir=out-shakespeare-m4 --start='To be or not to be'"
