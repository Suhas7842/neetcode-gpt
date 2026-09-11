import torch
import torch.nn as nn
from typing import Tuple, Optional

class KVCache:
    def __init__(self):
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Append new_k and new_v to the cache along the sequence dimension (dim=1).
        # On the first call, initialize the cache with the given tensors.
        # Return the full (cached) K and V tensors.
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)

        return self.cache_k, self.cache_v

    def clear(self):
        self.cache_k = None
        self.cache_v = None

class CachedAttention(nn.Module):
    def __init__(self, model_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Apply a causal mask offset by the number of previously cached tokens
        # 6. Return (rounded output, kv_cache)
        # 1. Project x into Q, K, V
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 2. Create cache if necessary
        if kv_cache is None:
            kv_cache = KVCache()

        # Number of tokens already in cache
        previous_tokens = (
            0 if kv_cache.cache_k is None
            else kv_cache.cache_k.size(1)
        )

        # 3. Update cache
        K, V = kv_cache.update(K, V)

        # 4. Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (x.shape[-1] ** 0.5)

        # 5. Causal mask
        q_len = Q.size(1)
        k_len = K.size(1)

        mask = torch.tril(
            torch.ones(q_len, k_len, device=x.device),
            diagonal=previous_tokens
        ).bool()

        scores = scores.masked_fill(~mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        output = attention @ V

        # 6. Return rounded output + cache
        return torch.round(output, decimals=4), kv_cache
