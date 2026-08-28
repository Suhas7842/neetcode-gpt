import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        sentences = positive + negative

        # Build vocabulary
        vocab = sorted(
            set(
                word
                for sentence in sentences
                for word in sentence.split()
            )
        )

        # IDs start at 1; 0 is padding
        word_to_id = {
            word: i + 1
            for i, word in enumerate(vocab)
        }

        # Encode sentences
        tensors = []

        for sentence in sentences:
            ids = [
                word_to_id[word]
                for word in sentence.split()
            ]

            tensors.append(
                torch.tensor(ids, dtype=torch.long)
            )

        # Pad sequences
        return nn.utils.rnn.pad_sequence(
            tensors,
            batch_first=True
        )