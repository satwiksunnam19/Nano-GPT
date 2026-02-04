# positional encoding

# RoPE (position rep as rotation)

import torch
import torch.nn as nn


class RoPE(nn.Module):
    """
    Rotary Position Embedding

    Step 1: What's 2D rotation
    To rotate a point (x,y) by an angle theta
    x_dash = x*cos(theta) - y*sin(theta)
    y_dash = x*sin(theta) + y*cos(theta)

    Step 2: RoPE splits head_dim into pairs
    head_dim = 64
    Split into 32 pairs:
    [x0,x1], [x2,x3], [x4,x5] ----- [x62,x63]
    pair 0,   pair 1,  pair 2  ----  pair 31
    Each pair gets rotated by diff angle

    Step 3: What angle for each pair ?
    theta_i = 1 / (10000^(2i/head_dim)), i = pair_number

    Step 4: Position multiplies the angle
    At position m, pair i rotates by m * theta_i

    Shape Journey:
        inv_freq: (head_dim/2,)
        positions: (seq_len,)
        angles: (seq_len, head_dim/2)
        cos_cached, sin_cached: (max_seq_len, head_dim)
        q, k input: (batch, n_heads, seq_len, head_dim)
        q, k output: (batch, n_heads, seq_len, head_dim)
    """

    def __init__(self, head_dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Step 3: Precompute inverse frequencies θ_i = 1 / (base^(2i/head_dim))
        # Shape: (head_dim/2,)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cos/sin cache
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        # positions: [0, 1, 2, ..., seq_len-1]
        # Shape: (seq_len,)
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)

        # Step 4: angles[pos, i] = pos * inv_freq[i]
        # Shape: (seq_len, head_dim/2)
        angles = positions[:, None] * self.inv_freq[None, :]

        # Compute cos and sin
        # Shape: (seq_len, head_dim/2)
        cos = angles.cos()
        sin = angles.sin()

        # Repeat to match full head_dim: (seq_len, head_dim/2) → (seq_len, head_dim)
        cos_cached = torch.cat([cos, cos], dim=-1)
        sin_cached = torch.cat([sin, sin], dim=-1)

        self.register_buffer("cos_cached", cos_cached)
        self.register_buffer("sin_cached", sin_cached)

    def rotate_half(self, x):
        """
        Step 1 trick: [x0, x1, x2, x3, ...] → [-x1, x0, -x3, x2, ...]

        This lets us apply rotation to all pairs at once:
        x_rotated = x * cos + rotate_half(x) * sin
        """
        x1 = x[..., : x.shape[-1] // 2]  # first half
        x2 = x[..., x.shape[-1] // 2 :]  # second half
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k, seq_len):
        """
        Apply rotary position embedding to q and k.

        Args:
            q: (batch, n_heads, seq_len, head_dim)
            k: (batch, n_heads, seq_len, head_dim)
            seq_len: current sequence length

        Returns:
            q_rotated, k_rotated: same shapes as input
        """
        # Get cos/sin for current sequence length
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]

        # Reshape for broadcasting: (seq_len, head_dim) → (1, 1, seq_len, head_dim)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        # Apply rotation: x_rotated = x * cos + rotate_half(x) * sin
        q_rotated = q * cos + self.rotate_half(q) * sin
        k_rotated = k * cos + self.rotate_half(k) * sin

        return q_rotated, k_rotated
