from math import sqrt

from pandas import np


def distance(v1, v2) :
    result = 1 - tanimoto_coefficient(v1, v2)
    return result

def _square_euclidean_distance(p_vec, q_vec):
    diff = p_vec - q_vec
    return max(np.sum(diff**2),0)


def euclidean_distance(p_vec, q_vec):
    return max(sqrt(_square_euclidean_distance(p_vec, q_vec)),0)


def tanimoto_coefficient(p_vec, q_vec):
    pq = p_vec * q_vec
    p_square = p_vec**2
    q_square = q_vec**2
    return max(pq / (p_square + q_square - pq))