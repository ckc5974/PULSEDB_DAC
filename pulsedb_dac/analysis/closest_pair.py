import numpy as np
from similarity.dtw import dtw_distance

def closest_pair_dtw(cluster_signals, original_indices):
    """Return (global_idx_a, global_idx_b, dtw_distance)."""
    m = len(cluster_signals)
    best = (None, None, np.inf)
    for i in range(m):
        for j in range(i+1, m):
            d = dtw_distance(cluster_signals[i], cluster_signals[j])
            if d < best[2]:
                best = (original_indices[i], original_indices[j], d)
    return best
