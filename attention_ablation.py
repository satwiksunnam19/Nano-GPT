# MHA vs GQA Comparison 
# Multi-Head Attention implementation from scratch 

import torch
import torch.nn as nn
import math
from torch.nn import functional as F


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

        # Separate projections for Q, K, V (each: n_embd -> n_embd)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # Causal mask: lower triangular matrix
        # This ensures position i can only attend to positions <= i
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, return_attention=False):
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

        # projections of q,k,v
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, self.n_kv_heads * self.head_size, bias=config.bias)

        # output projection
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask: Lower Triangular Matrix
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )

    def forward(self, x, return_attention=False):
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

        # projections
        self.q_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.kv_down_proj = nn.Linear(self.n_embd, self.latent_dim, bias=config.bias)
        self.k_up_proj = nn.Linear(self.latent_dim, self.n_embd, bias=config.bias)
        self.v_up_proj = nn.Linear(self.latent_dim, self.n_embd, bias=config.bias)

        self.out_proj = nn.Linear(self.n_embd, self.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(self.block_size, self.block_size))
            .view(1, 1, self.block_size, self.block_size),
        )

    def forward(self, x, return_attention=False):
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

