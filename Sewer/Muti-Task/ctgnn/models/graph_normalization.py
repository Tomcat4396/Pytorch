import numpy as np

def symmetric_matrix_normalization(adj):
    D = adj.sum(axis=1) # Row sum
    D_power = D ** (-0.5)
    D_power = np.diag(D_power)

    adj_norm = np.matmul(np.matmul(D_power, adj), D_power)
    return adj_norm


def in_degree_matrix_normalization(adj):
    D = adj.sum(axis=1) # Row sum
    D_power = D ** (-1.0)
    D_power = np.diag(D_power)

    adj_norm = np.matmul(D_power, adj)
    return adj_norm