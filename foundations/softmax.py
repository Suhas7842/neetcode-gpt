import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        maxNum=np.max(z)
        exp = np.exp(z - maxNum)
        softmax = exp / np.sum(exp)
        return np.round(softmax,4)