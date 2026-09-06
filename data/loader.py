import torch
from torchtyping import TensorType
from typing import Tuple

class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        # Choose random valid starting positions
        torch.manual_seed(0)
        starts = torch.randint(
            0,
            len(data) - context_length,
            (batch_size,)
        )

        X = []
        Y = []

        for start in starts:
            start = start.item()

            # Input sequence
            x = data[start : start + context_length]

            # Same sequence, shifted by one position
            y = data[start + 1 : start + context_length + 1]

            X.append(x)
            Y.append(y)

        return torch.stack(X), torch.stack(Y)