import numpy as np

def standardize(matrix):
    _, cols = np.shape(matrix)
    for col in range(cols):
        data = matrix[:, col]
        std = np.std(data)
        mean = np.mean(data)
        matrix[:, col] = (data - std) / mean
    return matrix

def standardize_cols(matrix, cols):
    rows = len(matrix)
    for col in cols:
        col_total = np.zeros((rows))
        for row in range(rows):
            try:
                col_total[row] = matrix[row][col]
            except:
                print(row)
        std = np.std(col_total)
        mean = np.mean(col_total)
        for row in range(rows):
            matrix[row][col] = (matrix[row][col] - mean) / std
    return matrix
