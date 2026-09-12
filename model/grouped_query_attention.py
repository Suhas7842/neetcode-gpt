import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        # 1. Project x into Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Reshape into heads
        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_kv_heads, self.head_dim)
        V = V.view(B, T, self.num_kv_heads, self.head_dim)

        # (B, T, heads, head_dim) -> (B, heads, T, head_dim)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 3. Repeat each KV head
        repeat_factor = self.num_heads // self.num_kv_heads

        K = K.repeat_interleave(repeat_factor, dim=1)
        V = V.repeat_interleave(repeat_factor, dim=1)

        # 4. Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.head_dim ** 0.5)

        # Causal mask
        mask = torch.tril(
            torch.ones(T, T, device=x.device)
        ).bool()

        scores = scores.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        output = attention @ V

        # 5. Concatenate heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, self.num_heads * self.head_dim)

        output = self.output_proj(output)

        # 6. Round
        return torch.round(output, decimals=4)
