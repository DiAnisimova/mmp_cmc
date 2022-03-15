import numpy as np


def euclidean_distance(X, Y):
    norm_x = (X ** 2).sum(axis=1)[:, np.newaxis]
    norm_y = (Y ** 2).sum(axis=1)
    minus = 2 * X @ Y.T
    return np.sqrt(norm_x + norm_y - minus)


def cosine_distance(X, Y):
    P = X @ Y.T
    P_norm = np.sqrt((X ** 2).sum(axis=1))[:, np.newaxis] @ np.sqrt((Y ** 2).sum(axis=1))[np.newaxis, :]
    return 1 - P / P_norm
