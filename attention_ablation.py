# MHA vs GQA Comparison 
# Multi-Head Attention implementation from scratch 

import torch
import torch.nn as nn
import math
from torch.nn import functional as F


def create_sliding_window_mask(T, window_size=None, sink_size=0, device='cpu'):
    """
    Creates a sliding window + attention sink causal mask.

    Args:
        T: sequence length
        window_size: number of positions to attend to (None = full causal)
        sink_size: number of initial tokens always accessible
        device: torch device

    Returns:
        mask: (T, T) boolean mask where True = can attend
        
    """ 
    # Start with causal mask
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))

    # Apply sliding window if specified
    if window_size is not None and window_size > 0:
        band_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=device),
                              diagonal=-(window_size - 1))
        mask = mask & band_mask

    # Add attention sinks
    if sink_size > 0:
        mask[:, :sink_size] = True
        # Respect causality for sink tokens themselves
        for i in range(sink_size):
            mask[i, i+1:] = False

    return mask


class MHA(nn.Module):
    """
    Multi-Head Attention with separate Q, K, V projections.

    Shape journey:
    Input:  (B, T, C)           e.g., (4, 128, 768)
    Q,K,V:  (B, T, C)           e.g., (4, 128, 768)
    Split:  (B, T, nh, hs)      e.g., (4, 128, 12, 64)
    Trans:  (B, nh, T, hs)      e.g., (4, 12, 128, 64)  <- heads as batch dim
    Att:    (B, nh, T, T)       e.g., (4, 12, 128, 128) <- attention matrix
    Out:    (B, T, C)           e.g., (4, 128, 768)
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        # Store config values we need
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head  # 768/12 = 64
        self.dropout = config.dropout

        # Sliding window parameters
        self.window_size = getattr(config, 'window_size', None)
        self.sink_size = getattr(config, 'sink_size', 0)
        self.use_sliding_window = self.window_size is not None

        # Separate projections for Q, K, V (each: n_embd -> n_embd)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix (fallback for non-sliding window)
        # This ensures position i can only attend to positions <= i
        if not self.use_sliding_window:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
        else:
            self.register_buffer("bias", None)

    def forward(self, x, mask=None, return_attention=False):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Step 1: Project to Q, K, V (each is B, T, C)
        q = self.q_proj(x)  # (B, T, 768)
        k = self.k_proj(x)  # (B, T, 768)
        v = self.v_proj(x)  # (B, T, 768)

        # Step 2: Reshape to (B, T, n_head, head_size) then transpose to (B, n_head, T, head_size)
        # This makes each head operate independently (like a batch dimension)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, 12, T, 64)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, 12, T, 64)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, 12, T, 64)

        # Step 3: Compute attention scores
        # Q @ K^T: (B, 12, T, 64) @ (B, 12, 64, T) -> (B, 12, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))

        # Step 4: Apply causal mask (can't attend to future tokens)
        if self.use_sliding_window:
            # Create sliding window mask dynamically
            mask = create_sliding_window_mask(T, self.window_size, self.sink_size, x.device)
            att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # Use standard causal mask
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Step 5: Softmax to get attention probabilities (each row sums to 1)
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Step 6: Weighted sum of values
        # (B, 12, T, T) @ (B, 12, T, 64) -> (B, 12, T, 64)
        y = att @ v

        # Step 7: Reassemble heads: transpose back and concatenate
        # (B, 12, T, 64) -> (B, T, 12, 64) -> (B, T, 768)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Final output projection
        y = self.resid_dropout(self.out_proj(y))

        if return_attention:
            return y, att
        return y

# =============================================================================
# Test the implementation
# =============================================================================

if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class Config:
        n_embd: int = 768
        n_head: int = 12
        block_size: int = 128
        dropout: float = 0.0
        bias: bool = True

    config = Config()
    mha = MHA(config)

    # Test input
    B, T, C = 2, 64, 768  # batch=2, seq_len=64, emb_dim=768
    x = torch.randn(B, T, C)

    # Forward pass
    y, att = mha(x, return_attention=True)

    # Count parameters
    total_params_mha = sum(p.numel() for p in mha.parameters())
    print(f"\nTotal parameters: {total_params_mha:,}") 
    print(f"  Q proj: {768 * 768 + 768:,}")  # weight + bias
    print(f"  K proj: {768 * 768 + 768:,}")
    print(f"  V proj: {768 * 768 + 768:,}")
    print(f"  Out proj: {768 * 768 + 768:,}")


#------------GQA------------------------------- 
class GQA(nn.Module):
    """
    Group-Query attention
    Shape Journey:
    Input: (B,T,C)
    Q,K,V: (B,T,C)
    N_KV_HEADS= n_kv
    N_q_Heads= n_q = n_head
    assert check the n_head % n_kv_heads == 0
    groups = n_head // n_kv_heads
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_head % config.n_kv_heads == 0, "n_head should be divisible by n_kv_heads for perfect grouping"

        # store config values
        self.n_head = config.n_head
        self.n_kv_heads = config.n_kv_heads
        self.group_size = config.n_head // config.n_kv_heads  # 12/4 = 3
        self.dropout = config.dropout
        self.head_size = config.n_embd // config.n_head

        # Sliding window parameters
        self.window_size = getattr(config, 'window_size', None)
        self.sink_size = getattr(config, 'sink_size', 0)
        self.use_sliding_window = self.window_size is not None

        # projections of q,k,v
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size, bias=config.bias)

        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask: Lower Triangular Matrix (fallback for non-sliding window)
        if not self.use_sliding_window:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )
        else:
            self.register_buffer("bias", None)

    def forward(self, x, mask=None, return_attention=False):
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Step 1: Project to Q, K, V
        q = self.q_proj(x)  # (B, T, 768)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_size)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_size)

        # Step 2: Reshape to (B, T, n_head, head_size) and transpose to (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_size).transpose(1, 2)

        # Expand K, V to match Q heads via reshape + expand
        k = k.view(B, self.n_kv_heads, 1, T, self.head_size).expand(B, self.n_kv_heads, self.group_size, T, self.head_size)
        k = k.reshape(B, self.n_head, T, self.head_size)
        v = v.view(B, self.n_kv_heads, 1, T, self.head_size).expand(B, self.n_kv_heads, self.group_size, T, self.head_size)
        v = v.reshape(B, self.n_head, T, self.head_size)

        # Step 3: Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))

        # Step 4: Apply causal mask
        if self.use_sliding_window:
            # Create sliding window mask dynamically
            mask = create_sliding_window_mask(T, self.window_size, self.sink_size, x.device)
            att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            # Use standard causal mask
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Step 5: Softmax to get attention probabilities
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Step 6: Weighted sum of values
        y = att @ v

        # Step 7: Reassemble heads
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Step 8: Final output projection
        y = self.resid_dropout(self.out_proj(y))

        if return_attention:
            return y, att
        return y 


# Test GQA 
if __name__ =="__main__": 
    from dataclasses import dataclass 

    @dataclass
    class GQAConfig:
        n_embd: int = 768
        n_head: int = 12
        n_kv_heads: int = 4
        block_size: int = 128
        dropout: float = 0.0
        bias: bool = True 
    config=GQAConfig()
    gqa=GQA(config)
    B, T, C = 2, 64, 768
    x = torch.randn(B, T, C)
    y, att = gqa(x, return_attention=True) 
    print("=" * 60) 
    print("GQA Test Results") 
    print("=" * 60) 
    print(f"Input:  {x.shape}")      # (2, 64, 768) 
    print(f"Output: {y.shape}")      # (2, 64, 768) 
    print(f"Attention: {att.shape}") # (2, 12, 64, 64) 
    print()
    print(f"Q heads: {config.n_head}")
    print(f"KV heads: {config.n_kv_heads}")
    print(f"Group size: {config.n_head // config.n_kv_heads}")
    print()

    # Parameter comparison
    # Count parameters

    mha_kv_params = 2 * (config.n_embd * config.n_embd)  # K + V for MHA
    gqa_kv_params = 2 * (config.n_embd * config.n_kv_heads * (config.n_embd // config.n_head))  # K + V for GQA
    print(f"MHA K,V params: {mha_kv_params/1e9:.2f}B")
    print(f"GQA K,V params: {gqa_kv_params/1e9:.2f}B") 
    total_params_mha = sum(p.numel() for p in mha.parameters())
    total_params_gqa = sum(p.numel() for p in gqa.parameters())
    print(f"MHA Total params: {total_params_mha/1e9:.2f}B")
    print(f"GQA Total params: {total_params_gqa/1e9:.2f}B") 
    print(f"Savings: {(1 - total_params_mha/total_params_gqa)*100:.0f}%")  # 66% smaller!

#------------MLA--------------------------- 
class MLA(nn.Module):
    """
    Multi-head Latent Attention (MLA)

    Shape Journey:
      x -> kv_down_proj(C -> L) -> latent_kv
      latent_kv -> k_up_proj / v_up_proj (L -> C)
      q,k,v -> n_head heads

    KV cache stores latent only (useful for inference).
    """

    def __init__(self, config):
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.latent_dim = config.latent_dim
        self.block_size = config.block_size

        self.head_size = self.n_embd // self.n_head
        assert self.head_size * self.n_head == self.n_embd

        # Sliding window parameters
        self.window_size = getattr(config, 'window_size', None)
        self.sink_size = getattr(config, 'sink_size', 0)
        self.use_sliding_window = self.window_size is not None

        # projections
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.kv_down_proj = nn.Linear(self.n_embd, self.latent_dim, bias=config.bias)
        self.k_up_proj = nn.Linear(self.latent_dim, self.n_embd, bias=config.bias)
        self.v_up_proj = nn.Linear(self.latent_dim, self.n_embd, bias=config.bias)

        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask (fallback for non-sliding window)
        if not self.use_sliding_window:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(self.block_size, self.block_size))
                .view(1, 1, self.block_size, self.block_size),
            )
        else:
            self.register_buffer("bias", None)

    def forward(self, x, mask=None, return_attention=False):
        B, T, C = x.size()

        # -------- Q --------
        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        # -------- latent KV --------
        latent_kv = self.kv_down_proj(x)

        # -------- up-project --------
        k = self.k_up_proj(latent_kv)
        v = self.v_up_proj(latent_kv)

        S = k.size(1)

        k = k.view(B, S, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, S, self.n_head, self.head_size).transpose(1, 2)

        # -------- attention --------
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)

        # Apply mask
        if self.use_sliding_window:
            # Create sliding window mask dynamically
            mask = create_sliding_window_mask(T, self.window_size, self.sink_size, x.device)
            att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        else:
            # Use standard causal mask
            att = att.masked_fill(self.bias[:, :, :T, :S] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))

        if return_attention:
            return y, att
        return y

# ------------------ Test MLA ------------------

if __name__ == "__main__":
    import torch
    from dataclasses import dataclass

    @dataclass
    class MLAConfig:
        n_embd: int = 768
        n_head: int = 12
        latent_dim: int = 128
        block_size: int = 128
        dropout: float = 0.0
        bias: bool = True

    config = MLAConfig()
    mla = MLA(config)

    B, T, C = 2, 64, 768
    x = torch.randn(B, T, C)

    y, att = mla(x, return_attention=True)

    print("=" * 60)
    print("MLA Test Results")
    print("=" * 60)

    print(f"Input:      {x.shape}")       # (2, 64, 768)
    print(f"Output:     {y.shape}")       # (2, 64, 768)
    print(f"Attention:  {att.shape}")     # (2, 12, 64, 64)

    print()
    print(f"Heads: {config.n_head}")
    print(f"Latent dim: {config.latent_dim}")
    print()

    # ---------------- Parameter Comparison ----------------

    # Standard MHA K,V params:
    mha_kv_params = 2 * (config.n_embd * config.n_embd)

    # MLA K,V params:
    mla_kv_params = (
        config.n_embd * config.latent_dim +  # down proj
        2 * (config.latent_dim * config.n_embd)  # up proj K,V
    )

    print(f"MHA K,V params: {mha_kv_params:,}")
    print(f"MLA K,V params: {mla_kv_params:,}")


    savings = (1 - mla_kv_params / mha_kv_params) * 100
    print(f"MLA param overhead vs MHA: {-savings:.1f}% (negative means more params)")


# =============================================================================
# Test Sliding Window Attention
# =============================================================================

def test_sliding_window():
    """Test sliding window attention with various configurations"""
    from dataclasses import dataclass

    print("\n" + "=" * 70)
    print("SLIDING WINDOW ATTENTION TEST")
    print("=" * 70)

    @dataclass
    class SlidingWindowConfig:
        n_embd: int = 768
        n_head: int = 12
        block_size: int = 128
        dropout: float = 0.0
        bias: bool = True
        window_size: int = 16
        sink_size: int = 4

    config = SlidingWindowConfig()

    # Test with larger sequence to see sliding window effect
    B, T, C = 2, 32, 768
    x = torch.randn(B, T, C)

    # Test MHA with sliding window
    print("\n1. MHA with Sliding Window")
    print("-" * 70)
    mha_sliding = MHA(config)
    y, att = mha_sliding(x, return_attention=True)
    print(f"✓ Window size: {config.window_size}, Sink size: {config.sink_size}")
    print(f"✓ Input:  {x.shape}")
    print(f"✓ Output: {y.shape}")
    print(f"✓ Attention: {att.shape}")

    # Show mask pattern
    mask = create_sliding_window_mask(T, config.window_size, config.sink_size)
    print(f"\nMask pattern (first 12 positions):")
    print("Pos | Can attend to")
    print("-" * 50)
    for i in range(min(12, T)):
        positions = torch.where(mask[i])[0].tolist()
        window_start = max(0, i - config.window_size + 1)
        print(f" {i:2d} | {positions[:20]}... (sink=[0:{config.sink_size}], window=[{window_start}:{i+1}])")

    # Test without sliding window (standard causal)
    print("\n2. MHA without Sliding Window (Standard Causal)")
    print("-" * 70)
    @dataclass
    class StandardConfig:
        n_embd: int = 768
        n_head: int = 12
        block_size: int = 128
        dropout: float = 0.0
        bias: bool = True
        # No window_size = standard causal

    standard_config = StandardConfig()
    mha_standard = MHA(standard_config)
    y_standard, att_standard = mha_standard(x, return_attention=True)
    print(f"✓ Standard causal attention (no window limit)")
    print(f"✓ Input:  {x.shape}")
    print(f"✓ Output: {y_standard.shape}")

    # Memory comparison
    print("\n3. Memory Complexity Comparison")
    print("-" * 70)
    seq_len = 4096
    standard_memory = seq_len * seq_len
    sliding_memory = seq_len * (config.window_size + config.sink_size)

    print(f"For sequence length = {seq_len}:")
    print(f"  Standard attention: {standard_memory:,} elements ({standard_memory / 1e6:.2f}M)")
    print(f"  Sliding window:     {sliding_memory:,} elements ({sliding_memory / 1e6:.2f}M)")
    print(f"  Memory reduction:   {100 * (1 - sliding_memory / standard_memory):.1f}%")

    # Test GQA with sliding window
    print("\n4. GQA with Sliding Window")
    print("-" * 70)
    @dataclass
    class GQASlidingConfig:
        n_embd: int = 768
        n_head: int = 12
        n_kv_heads: int = 4
        block_size: int = 128
        dropout: float = 0.0
        bias: bool = True
        window_size: int = 16
        sink_size: int = 4

    gqa_config = GQASlidingConfig()
    gqa_sliding = GQA(gqa_config)
    y_gqa, att_gqa = gqa_sliding(x, return_attention=True)
    print(f"✓ GQA with window_size={gqa_config.window_size}, sink_size={gqa_config.sink_size}")
    print(f"✓ Q heads: {gqa_config.n_head}, KV heads: {gqa_config.n_kv_heads}")
    print(f"✓ Output: {y_gqa.shape}")

    # Test MLA with sliding window
    print("\n5. MLA with Sliding Window")
    print("-" * 70)
    @dataclass
    class MLASlidingConfig:
        n_embd: int = 768
        n_head: int = 12
        latent_dim: int = 128
        block_size: int = 128
        dropout: float = 0.0
        bias: bool = True
        window_size: int = 16
        sink_size: int = 4

    mla_config = MLASlidingConfig()
    mla_sliding = MLA(mla_config)
    y_mla, att_mla = mla_sliding(x, return_attention=True)
    print(f" MLA with window_size={mla_config.window_size}, sink_size={mla_config.sink_size}")
    print(f" Latent dim: {mla_config.latent_dim}")
    print(f" Output: {y_mla.shape}")

if __name__ == "__main__":
    test_sliding_window()
