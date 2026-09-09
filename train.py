import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        model.train()

        for epoch in range(epochs):

            # Required seed for reproducible batch selection
            torch.manual_seed(epoch)

            # Generate random starting positions
            starts = torch.randint(
                0,
                len(data) - context_length,
                (batch_size,)
            )

            X = []
            Y = []

            for start in starts:
                start = start.item()

                # Input tokens
                X.append(
                    data[start:start + context_length]
                )

                # Target tokens shifted by one position
                Y.append(
                    data[start + 1:start + context_length + 1]
                )

            X = torch.stack(X)
            Y = torch.stack(Y)

            # Forward pass
            logits = model(X)

            # logits: (B, T, vocab_size)
            # cross_entropy expects:
            # input:  (N, C)
            # target: (N)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                Y.reshape(-1)
            )

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return round(loss.item(), 4)
