import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        stats = []

        with torch.no_grad():
            for layer in model:
                x = layer(x)

                if isinstance(layer, nn.Linear):
                    # Each column = one neuron
                    # Check if neuron never fires for ANY sample
                    dead = (x <= 0).all(dim=0)

                    stats.append({
                        "mean": round(x.mean().item(), 4),
                        "std": round(x.std().item(), 4),
                        "dead_fraction": round(dead.float().mean().item(), 4)
                    })

        return stats

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        model.zero_grad()

        prediction = model(x)

        loss = nn.MSELoss()(prediction, y)

        loss.backward()

        stats = []

        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                grad = layer.weight.grad

                stats.append({
                    "mean": round(grad.mean().item(), 4),
                    "std": round(grad.std().item(), 4),
                    "norm": round(torch.norm(grad).item(), 4)
                })

        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        
        # 1. Dead neurons
        if any(stat["dead_fraction"] > 0.5 for stat in activation_stats):
            return "dead_neurons"

        # 2. Exploding gradients
        if any(stat["norm"] > 1000 for stat in gradient_stats):
            return "exploding_gradients"

        # 3. Vanishing gradients
        if gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # 4. Activation std
        if any(stat["std"] < 0.1 for stat in activation_stats):
            return "vanishing_gradients"

        if any(stat["std"] > 10.0 for stat in activation_stats):
            return "exploding_gradients"

        # 5. Everything looks healthy
        return "healthy"
