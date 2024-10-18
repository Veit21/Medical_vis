# test file
import numpy as np
from scipy import special

print(special.binom(3, 1) * special.binom(3, 1))


def binomialKernel(k):
    mask = np.zeros((k, k))
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            mask[row, col] = special.binom(k - 1, row) * special.binom(k - 1, col)
    return 1/(np.power(4, k - 1)) * mask


print(binomialKernel(5))
