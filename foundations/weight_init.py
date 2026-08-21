import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2 / (fan_in + fan_out))

        W = torch.randn(fan_out, fan_in) * std

        return torch.round(W, decimals=4).tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)

        std = math.sqrt(2 / fan_in)

        W = torch.randn(fan_out, fan_in) * std

        return torch.round(W, decimals=4).tolist()

    def check_activations(
        self,
        num_layers: int,
        input_dim: int,
        hidden_dim: int,
        init_type: str
    ) -> List[float]:

        torch.manual_seed(0)

        # Build ALL weights first
        weights = []

        for i in range(num_layers):

            if init_type == "xavier":
                std = math.sqrt(2 / (input_dim + hidden_dim))
                W = torch.randn(hidden_dim, input_dim) * std

            elif init_type == "kaiming":
                std = math.sqrt(2 / input_dim)
                W = torch.randn(hidden_dim, input_dim) * std

            else:  # random
                W = torch.randn(hidden_dim, input_dim)

            weights.append(W)

            input_dim = hidden_dim

        # Generate input AFTER all weights
        x = torch.randn(input_dim)

        # Forward pass
        result = []

        for W in weights:
            x = W @ x
            x = torch.relu(x)

            result.append(round(x.std().item(), 2))

        return result