

import numpy
import tensorly


# Naive online CP decomposition for multi-order tensors

def compute_accum(factors, skip_mode):
    accum = 1.
    for i, A in enumerate(factors):
        if i == skip_mode: continue
        accum *= A.T @ A
    return accum


def online_update_cp_tensor(X_new, A, P, Q, alpha=1, ignore_temporal_mode=True):

    for mode in range(X_new.ndim):
        P[mode] += tensorly.unfolding_dot_khatri_rao(X_new, (None, A), mode)
        Q[mode] += compute_accum(A, skip_mode=mode)
        A[mode] = P[mode] @ numpy.linalg.pinv(Q[mode])

    return P, Q