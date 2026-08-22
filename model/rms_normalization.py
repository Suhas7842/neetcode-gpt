import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list
        x = np.array(x, dtype=float)
        gamma = np.array(gamma, dtype=float)

        rms = np.sqrt(np.mean(x ** 2) + eps)

        output = gamma * (x / rms)

        return np.round(output, 4).tolist()
