import torch
from typing import List, Tuple

class Solution:
    def batch_loader(self, raw_dataset: str, context_length: int, batch_size: int) -> Tuple[List[List[str]], List[List[str]]]:
        # 1. Tokenize by splitting on whitespace: raw_dataset.split()
        # 2. Generate batch_size random start indices using torch.randint()
        #    Range: [0, len(tokens) - context_length)
        # 3. For each index i, X = tokens[i:i+context_length], Y = tokens[i+1:i+1+context_length]
        torch.manual_seed(0)
        # 1. Tokenize
        tokens = raw_dataset.split()

        # 2. Generate random starting positions
        starts = torch.randint(
            0,
            len(tokens) - context_length,
            (batch_size,)
        )

        X = []
        Y = []

        # 3. Create input and target sequences
        for i in starts:
            i = i.item()

            x = tokens[i:i + context_length]
            y = tokens[i + 1:i + context_length + 1]

            X.append(x)
            Y.append(y)

        return X, Y