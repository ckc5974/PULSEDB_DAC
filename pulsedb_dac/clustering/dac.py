import numpy as np

def _farthest_pair(D, idxs):
    m = len(idxs)
    if m < 2:
        return (idxs[0], idxs[0])
    best = (idxs[0], idxs[1])
    best_d = -1.0
    for i in range(m):
        for j in range(i+1, m):
            di = idxs[i]; dj = idxs[j]
            d = D[di, dj]
            if d > best_d:
                best_d = d
                best = (di, dj)
    return best

def _partition(D, idxs, a, b):
    A, B = [], []
    for k in idxs:
        da = D[k, a]; db = D[k, b]
        (A if da <= db else B).append(k)
    return A, B

def dac_cluster(D: np.ndarray, min_cluster: int = 15):
    """
    Divide-and-conquer clustering with correlation distance D:
    pick farthest pair as pivots
    assign by smaller distance to pivot A or B
    recurse until cluster size  min_cluster
    Returns: list[list[int]] clusters of indices
    """
    N = D.shape[0]
    all_idxs = list(range(N))
    clusters = []

    def recurse(idxs):
        if len(idxs) <= min_cluster:
            clusters.append(idxs)
            return
        a, b = _farthest_pair(D, idxs)
        A, B = _partition(D, idxs, a, b)
        if len(A) == 0 or len(B) == 0:
            clusters.append(idxs)
            return
        recurse(A)
        recurse(B)

    recurse(all_idxs)
    return clusters
