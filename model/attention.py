import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.query = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

        self.value = nn.Linear(
            embedding_dim,
            attention_dim,
            bias=False
        )

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        # K, Q, V
        K = self.key(embedded)
        Q = self.query(embedded)
        V = self.value(embedded)

        # QK^T / sqrt(d_k)
        scores = Q @ K.transpose(-2, -1)
        scores = scores / (self.query.out_features ** 0.5)

        # Causal mask
        T = embedded.shape[1]
        mask = torch.tril(torch.ones(T, T))

        scores = scores.masked_fill(
            mask == 0,
            float("-inf")
        )

        # Softmax
        scores = torch.softmax(scores, dim=2)

        # Weighted values
        output = scores @ V

        return torch.round(output, decimals=4)
