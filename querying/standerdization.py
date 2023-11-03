import numpy as np

def standardize(matrix):
    _, cols = np.shape(matrix)
    for col in range(cols):
        data = matrix[:, col]
        std = np.std(data)
        mean = np.mean(data)
        matrix[:, col] = (data - std) / mean
    return matrix
