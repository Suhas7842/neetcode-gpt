import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution:
    def generate(self, model, new_chars: int, context: TensorType[int], context_length: int, int_to_char: dict) -> str:
        # 1. Crop context to context_length if it exceeds it: context[:, -context_length:]
        # 2. Run model(context) -> take last position's logits -> apply softmax(dim=-1)
        # 3. Sample next token with torch.multinomial(probs, 1, generator=generator)
        # 4. Append sampled token to context with torch.cat
        # 5. Map token to character using int_to_char and accumulate result
        # Do not alter the fixed code below — it ensures reproducible test output.

        generator = torch.manual_seed(0)
        initial_state = generator.get_state()
        result = ""
        for i in range(new_chars):

           # 1. Keep only the last context_length tokens
            context = context[:, -context_length:]

            # 2. Get model predictions
            logits = model(context)

            # Take logits from the last position
            logits = logits[:, -1, :]

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=-1)

            # 3. Sample the next token
            generator.set_state(initial_state)

            next_token = torch.multinomial(
                probs,
                1,
                generator=generator
            )

            # 4. Append token to context
            context = torch.cat(
                [context, next_token],
                dim=1
            )

            # 5. Convert token ID to character
            result += int_to_char[next_token.item()]

        return result