""" Definition of Minimum Discription Length """

import numpy as np
from sklearn import preprocessing
from scipy.stats import norm


def encoding_score(X, Y):
    """
        X: input data (matrix/tensor)
        Y: reconstruction of X
    """
    diff = (X - Y).ravel()
    prob = norm.pdf(diff, loc=diff.mean(), scale=diff.std())

    return -1 * np.log2(prob).sum()


def model_score(X, normalize=False, float_cost=32, tol=1e-3):
    """
        X: input data (matrix/tensor)
        tol: threshold for zero values
    """
    score = 0

    if X.ndim == 1:
        k = X.shape[0]
        X_nonzero = np.count_nonzero(np.logical_or(X < tol, tol < X))
        score += X_nonzero * (np.log(k) + float_cost)
        score += np.log1p(X_nonzero)


    elif X.ndim == 2:
        k, l = X.shape  # k: # of dimensions of observations
        X_nonzero = np.count_nonzero(X > tol)
        print('Nonzero=', X_nonzero)

        if normalize == True:
            score += X_nonzero * (np.log(k) + np.log(l) + float_cost) / k
        else:
            score += X_nonzero * (np.log(k) + np.log(l) + float_cost)

        score += np.log1p(X_nonzero)

    return score
