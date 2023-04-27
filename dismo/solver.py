""" Parameter estimation methods for DISMO """
import warnings

import time
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tensorly import backend as T
from tensorly.decomposition import parafac, non_negative_parafac
from tensorly import kruskal_to_tensor, unfolding_dot_khatri_rao
from scipy.ndimage import gaussian_filter1d
from sklearn import preprocessing
from termcolor import colored, cprint

from . import glv
from . import utils


SIGMA = 20


def get_timestamps(start_point, n_seasons, n_samples):
    return np.mod(np.arange(n_samples) + start_point, n_seasons)


def cyclic_fold(tensor, n_seasons, mode=0):

    if mode > 0:
        tensor = np.moveaxis(tensor, mode, 0)

    n_period = len(tensor) // n_seasons
    shape = (n_period, n_seasons, *tensor.shape[1:])
    folded = tl.base.fold(tensor[-n_period*n_seasons:], 0, shape)

    return folded


def init_seasonal_components(tensor, n_seasons, ns_comp):

    # Reshape input tensor
    c_folded = cyclic_fold(tensor, n_seasons)

    # Remove biases in each period
    c_folded = np.moveaxis(c_folded, 0, 1)
    means = c_folded.mean(axis=0)
    resid = c_folded - means
    resid = np.moveaxis(resid, 0, 1)

    # Remove biases between periods
    resid = resid.mean(axis=0)

    # Decompose
    return parafac(resid, ns_comp)


def init_lotka_volterra(timeseries, max_iter=10, return_sequence=True, 
                        only_linear=False):

    n_samples, n_dims = timeseries.shape

    # Z = np.zeros(n_dims)
    Z = timeseries[0]  # t = 0
    A = np.zeros(n_dims)
    B = np.zeros((n_dims, n_dims))

    # This doesn't converge well
    # all parameters will be near to zero
    # A, B, Z = glv.fit(timeseries, target='full')

    # for _ in range(max_iter):
    Z, A = glv.fit(timeseries, target='linear',
                init_params=(Z, A), fixed_params=(B))

    if only_linear == False:
        B = glv.fit(timeseries, target='nl_diag',
                    init_params=(B), fixed_params=(Z, A, B))

        B = glv.fit(timeseries, target='nl_nondiag',
                    init_params=(B), fixed_params=(Z, A, B))

    if not return_sequence:
        return A, B, Z

    else:
        seq, _ = glv._generate(n_samples, A, B, Z)
        return A, B, Z, seq


def fit_latent_lotka_volterra(tensor, A, B, Z, W, max_iter=5,
                              return_factor=True):

    # A, B, Z = glv.fit_tensor(tensor, W, target='full')

    # for _ in range(max_iter):
    Z, A = glv.fit_tensor(tensor, W,
                            target='linear',
                            init_params=(Z, A),
                            fixed_params=(B))

    B = glv.fit_tensor(tensor, W,
                        target='nl_diag',
                        init_params=B,
                        fixed_params=(Z, A, B))

    # NOTE: 非線形項は最後に1回最適化すれば十分
    B = glv.fit_tensor(tensor, W,
                        target='nl_nondiag',
                        init_params=B,
                        fixed_params=(Z, A))

    if return_factor == True:
        W[0], _ = glv._generate(len(tensor), A, B, Z)
        return A, B, Z, W

    else:
        return [A, B, Z]


# Equal to the gradient step for non-negative parafac
# TODO: Rename this function
def fit_observation_matrix(tensor, factors, mode, epsilon=10e-12, non_negative=True, normalize_factors=True):
    """ Single iteration for multi-linear operator estimation """

    nc_comp = factors[-1].shape[-1]
    target = tensor

    # For the given mode
    pseudo_inverse = T.tensor(
        np.ones((nc_comp, nc_comp)),
        **T.context(target))

    for i, factor in enumerate(factors):
        if not i == mode:
            pseudo_inverse *= tl.conj(
                T.dot(T.transpose(factor), factor))

    if non_negative:
        accum = 1
        # khatri_rao(factors).tl.dot(khatri_rao(factors))
        # simplifies to multiplications
        sub_indices = [i for i in range(len(factors)) if i != mode]
        for i, e in enumerate(sub_indices):
            if i:
                accum *= tl.dot(tl.transpose(factors[e]), factors[e])
            else:
                accum = tl.dot(tl.transpose(factors[e]), factors[e])

    pseudo_inverse += np.diag(np.full(nc_comp, 1e-10))
    mttkrp = unfolding_dot_khatri_rao(target, (None, factors), mode)

    if non_negative:
        numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
        denominator = tl.dot(factors[mode], accum)
        denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
        factor = factors[mode] * numerator / denominator

    else:    
        factor = T.transpose(T.solve(T.transpose(pseudo_inverse), T.transpose(mttkrp)))
    
    if normalize_factors:
        weights = tl.norm(factor, order=2, axis=0)
        weights = tl.where(tl.abs(weights) <= tl.eps(tensor.dtype), 
                            tl.ones(tl.shape(weights), **tl.context(factors[0])),
                            weights)

        factor = factor / (tl.reshape(weights, (1, -1)))

    return factor


def fit_seasonal_factors(tensor, seasonal_factors, normalize_factors=True):
    n_seasons, nc_comp = seasonal_factors[0].shape

    # Reshape input tensor
    c_folded = cyclic_fold(tensor, n_seasons)
    c_folded = c_folded.mean(axis=0)

    for mode in range(c_folded.ndim):
        pseudo_inverse = T.tensor(
            np.ones((nc_comp, nc_comp)),
            **T.context(c_folded))

        for i, factor in enumerate(seasonal_factors):
            if not i == mode:
                pseudo_inverse *= tl.conj(
                    T.dot(T.transpose(factor), factor))
        
        pseudo_inverse += np.diag(np.full(nc_comp, 1e-10))

        mttkrp = unfolding_dot_khatri_rao(
            c_folded, (None, seasonal_factors), mode)

        seasonal_factors[mode] = T.transpose(T.solve(
            T.transpose(pseudo_inverse),
            T.transpose(mttkrp)))

    if normalize_factors:
        weights = tl.norm(seasonal_factors[mode], order=2, axis=0)
        weights = tl.where(tl.abs(weights) <= tl.eps(tensor.dtype), 
                            tl.ones(tl.shape(weights), **tl.context(seasonal_factors[0])),
                            weights)
        seasonal_factors[mode] = seasonal_factors[mode] / (tl.reshape(weights, (1, -1)))

    return seasonal_factors


def update_seasonal_factors(timestamp, tensor, seasonal_factors, lr=0.1):
    n_samples = tensor.shape[0]
    n_seasons, nc_comp = seasonal_factors[0].shape

    # Input   
    recon_ = gen_seasonality(timestamp, n_samples, seasonal_factors)
    resid_ = tensor - recon_
    resid_ = cyclic_fold(resid_, n_seasons)
    resid_ = resid_.mean(axis=0)

    # Output
    seasonal_lag = np.mod(timestamp, n_seasons)
    smoothed_factors = [np.copy(factor) for factor in seasonal_factors]
    smoothed_factors[0] = np.roll(seasonal_factors[0], -seasonal_lag, axis=0)  # adjust seasons

    for _ in range(1):
        for mode in range(resid_.ndim):
            pseudo_inverse = T.tensor(
                np.ones((nc_comp, nc_comp)),
                **T.context(resid_))

            for i, factor in enumerate(smoothed_factors):
                if not i == mode:
                    pseudo_inverse *= tl.conj(
                        T.dot(T.transpose(factor), factor))

            pseudo_inverse += np.diag(np.full(nc_comp, 1e-10))

            mttkrp = unfolding_dot_khatri_rao(
                resid_, (None, smoothed_factors), mode)

            smoothed_factors[mode] += lr * T.transpose(
                T.solve(T.transpose(pseudo_inverse),
                        T.transpose(mttkrp)))

    # Roll time-axis component
    smoothed_factors[0] = np.roll(smoothed_factors[0], seasonal_lag, axis=0)  # adjust seasons
    return smoothed_factors


def fit_init_state(timestamp, tensor, lve, factors, seasonal_factors, sigma=SIGMA):
    n_samples = tensor.shape[0]
    A, B, Z = lve
    season = gen_seasonality(timestamp, n_samples, seasonal_factors)
    target = tensor - season

    # target = tl.base.fold(gaussian_filter1d(tl.base.unfold(tensor - season, 0), sigma), 0, tensor.shape)
    # target = tl.base.fold(gaussian_filter1d(tl.base.unfold(tensor, 0), sigma), 0, tensor.shape) - ES
    # Z = glv.fit_tensor(target, factors, target='init_state', init_params=Z, fixed_params=(A, B))

    factors[0] = np.ones(
        (tensor.shape[0], factors[-1].shape[-1]))  # time x k

    mttkrp = tl.unfolding_dot_khatri_rao(
        target, (None, factors), 0)

    # numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
    # denominator = tl.dot(ml_factors[0], accum)
    # denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
    # mttkrp = ml_factors[0] * numerator / denominator
    mttkrp = tl.clip(mttkrp, a_min=10e-12, a_max=None)
    mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
    mttkrp = mttkrp.T

    Z = glv.fit(mttkrp, target='init_state', fixed_params=(A, B))
    return Z


def gen_competition(n_samples, A, B, Z, W, generate=True):
    if generate == True:
        W[0], _ = glv._generate(n_samples, A, B, Z)
    return kruskal_to_tensor((None, W))


def gen_seasonality(timestamp, n_samples, seasonal_factors):
    n_seasons = seasonal_factors[0].shape[0]
    factorS = np.roll(
        seasonal_factors[0], -np.mod(timestamp, n_seasons), axis=0)
    factorS = np.tile(factorS, (n_samples // n_seasons + 1, 1))[:n_samples]
    factors = [factorS] + seasonal_factors[1:]
    return kruskal_to_tensor((None, factors))


def fit_dismo(timestamp, tensor, nc_comp, ns_comp, n_seasons,
              tol=1e-4, epslion=10e-12, max_iter=20, n_trial=1,
              non_negative=True, save_train_log=False):

    # Input
    n_modes = tensor.ndim
    n_samples = tensor.shape[0]
    norm_tensor = tl.norm(tensor, 2)

    rec_errors = []  # Optimization log

    # Initialization
    print('init seasonal factors')
    _, S = init_seasonal_components(tensor, n_seasons, ns_comp)
    print('init observation matrices')
    _, W = non_negative_parafac(tensor, nc_comp)
    print('init lotka-volterra equation')
    A, B, Z, W[0] = init_lotka_volterra(W[0])

    competition_tensor = gen_competition(n_samples, A, B, Z, W)
    seasonality_tensor = gen_seasonality(timestamp, n_samples, S)
    reconstruct_tensor = competition_tensor + seasonality_tensor

    # Optimization
    print('Optimization...')
    for iteration in range(max_iter):
        _diff_competition = tensor - competition_tensor
        _diff_seasonality = tensor - seasonality_tensor

        tic = time.process_time()
        # Update observation matrices
        for mode in range(1, n_modes):
            W[mode] = fit_observation_matrix(
                _diff_seasonality, W, mode, non_negative=non_negative)
        toc = time.process_time() - tic
        cprint("e-time [obs]: {:.3f} sec.".format(toc), 'green')

        tic = time.process_time()
        # Update non-linear dynamical system
        A, B, Z, W = fit_latent_lotka_volterra(_diff_seasonality, A, B, Z, W)
        toc = time.process_time() - tic
        cprint("e-time [lve]: {:.3f} sec.".format(toc), 'green')

        tic = time.process_time()
        # Update seasonal components
        S = fit_seasonal_factors(_diff_competition, S)
        toc = time.process_time() - tic
        cprint("e-time [ssn]: {:.3f} sec.".format(toc), 'green')

        # Compute reconstruction error
        competition_tensor = gen_competition(n_samples, A, B, Z, W)
        seasonality_tensor = gen_seasonality(timestamp, n_samples, S)
        reconstruct_tensor = competition_tensor + seasonality_tensor

        error = T.norm(tensor - reconstruct_tensor, 2) / norm_tensor
        rec_errors.append(error)
        print('iter=', iteration + 1, 'error=', error)

        # Check convergence
        if iteration > 2:
            if np.abs(rec_errors[-1] - rec_errors[-2]) < tol:
                break
            if rec_errors[-1] > rec_errors[-2] + tol:
                break

    else:
        warnings.warn("DISMO did not converge")

    return A, B, Z, W, S

def grad_update_nn_parafac(tensor, factors, mode,
                           learning_rate=1, min_iter=1,
                           epsilon=10e-12, non_negative=False,
                           normalize_factors=False):

    for _ in range(10):
        mttkrp = unfolding_dot_khatri_rao(tensor, (None, factors), mode)
        accum = 1
        sub_indices = [i for i in range(len(factors)) if i != mode]
        for i, e in enumerate(sub_indices):
            if i:
                accum *= tl.dot(tl.transpose(factors[e]), factors[e])
            else:
                accum = tl.dot(tl.transpose(factors[e]), factors[e])

        numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
        denominator = tl.dot(factors[mode], accum)
        denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
        factor = factors[mode] * learning_rate * numerator / denominator

        # if normalize_factors:
        weights = tl.norm(factor, order=2, axis=0)
        weights = tl.where(
            tl.abs(weights) <= tl.eps(tensor.dtype),
            tl.ones(tl.shape(weights),
            **tl.context(factors[0])),
            weights)

        factor = factor / (tl.reshape(weights, (1, -1)))
        # factors[mode] = factor

    return factor

def smooth_update_seasonal_factors(
        timestamp, tensor, seasonal_factors,
        lr=0.1, epsilon=10e-12, normalize_factors=True):

    n_sample = tensor.shape[0]
    period, ns = seasonal_factors[0].shape

    # recon_ = gen_seasonality(timestamp, n_sample, seasonal_factors)
    # resid_ = tensor - recon_
    # resid_ = cyclic_fold(resid_, period)
    # resid_ = resid_.mean(axis=0)
    resid_ = cyclic_fold(tensor, period)
    resid_ = resid_.mean(axis=0)
    # print(resid_.shape)

    season_lag = np.mod(timestamp, period)
    seasonal_factors[0] = np.roll(seasonal_factors[0], -season_lag, axis=0)

    # _, seasonal_factors = non_negative_parafac(resid_, ns, n_iter_max=1)
    # seasonal_factors[0] = np.roll(seasonal_factors[0], season_lag, axis=0)
    # return seasonal_factors

    for _ in range(10):
        for mode in range(tensor.ndim):
            mttkrp = unfolding_dot_khatri_rao(resid_, (None, seasonal_factors), mode)

            accum = 1
            sub_indices = [i for i in range(len(seasonal_factors)) if i != mode]
            for i, e in enumerate(sub_indices):
                if i:
                    accum *= tl.dot(tl.transpose(
                        seasonal_factors[e]), seasonal_factors[e])
                else:
                    accum = tl.dot(tl.transpose(
                        seasonal_factors[e]), seasonal_factors[e])

            numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
            denominator = tl.dot(seasonal_factors[mode], accum)
            denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
            factor = seasonal_factors[mode] * (1 - lr) + seasonal_factors[mode] * lr * numerator / denominator

            if normalize_factors:
                weights = tl.norm(factor, order=2, axis=0)
                weights = tl.where(
                    tl.abs(weights) <= tl.eps(tensor.dtype),
                    tl.ones(tl.shape(weights),
                            **tl.context(seasonal_factors[0])),
                    weights)
                factor = factor / (tl.reshape(weights, (1, -1)))

            seasonal_factors[mode] = factor

    # Reset season
    seasonal_factors[0] = np.roll(seasonal_factors[0], season_lag, axis=0)
    return seasonal_factors


def init_seasonal_factors(timestamp, tensor, period, rank,
                          n_iter_max=100, tol=10e-7, random_state=123):

    s_folded_tensor = np.zeros((period, *tensor.shape[1:]))

    n_season = tensor.shape[0] // period

    if n_season == 0:
        exit("Period is too short")

    # Mean based approach
    for i in range(n_season):
        one_season = tensor[i * period: (i + 1) * period]
        s_folded_tensor += one_season - one_season.mean(axis=0)
        # s_folded_tensor += tensor[i * period: (i + 1) * period]
    s_folded_tensor /= n_season

    # s_folded_tensor = np.array([
    #     tensor[i*period:(i+1)*period] for i in range(n_season)])

    _, seasonal_factors = non_negative_parafac(
        s_folded_tensor, rank,
        n_iter_max=n_iter_max, tol=tol, random_state=random_state)

    # seasonal_factors[0] += 1e-5
    # s_folded_tensor -= s_folded_tensor.mean(axis=0)
    # _, seasonal_factors = parafac(
    #     s_folded_tensor, rank,
    #     n_iter_max=n_iter_max, tol=tol, random_state=random_state)

    return seasonal_factors
    # return seasonal_factors[1:]


def init_ml_factors(tensor, rank, n_iter_max=100, tol=10e-7, random_state=123):

    _, ml_factors = non_negative_parafac(
        tensor, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state)

    return ml_factors


# Main fitting algorithm
def dismo_solver(timestamp, tensor, period, nc, ns,
                 tol=1e-3, epsilon=10e-12, n_iter_max=10
                 ):

    n_dim = tensor.ndim
    n_sample = tensor.shape[0]
    duration = n_sample // period
    norm_tensor = tl.norm(tensor, 2)
    zero = np.zeros((n_sample, nc))
    
    rec_errors = []  # train history

    # Initialization of all components

    S = init_seasonal_factors(timestamp, tensor, period, ns)
    S_recon_tensor = kruskal_to_tensor((None, S))
    S_recon_tensor = np.tile(S_recon_tensor, (duration, 1, 1))
    # S_recon_tensor = gen_seasonality(timestamp, n_sample, S)
    print('Initialized seasonal factors')

    C = init_ml_factors(tensor - S_recon_tensor, nc)  # Non-negative parafac
    print('Initialized ml factors')

    # plt.plot(C[0])
    # plt.title("Initial C0 factor")
    # plt.show()

    # ここはいらないのでは？
    #---------------
    C0_matrix = preprocessing.normalize(C[0])

    # plt.plot(C0_matrix)
    # plt.title("Normalized C0 factor")
    # plt.show()

    A, B, Z, C0 = init_lotka_volterra(C0_matrix, max_iter=2, return_sequence=True)
    C[0]        = C0  # generated factor with the LVE
    print('Initialized nlds')

    for _ in range(100):
        for mode in range(1, n_dim):
            C[mode] = grad_update_nn_parafac(
                tensor - S_recon_tensor, C, mode, non_negative=True)
    print('Initialized ml factors')

    C_recon_tensor = kruskal_to_tensor((None, C))
    #---------------

    # Compute initial residual tensors
    C_resid_tensor = tensor - S_recon_tensor  # data C should fit
    S_resid_tensor = tensor - C_recon_tensor  # data S should fit
    
    # Optimization
    for iteration in range(n_iter_max):

        # Fit Competing Activity
        for mode in range(n_dim):
            if mode == 0:

                mttkrp = tl.unfolding_dot_khatri_rao(
                    C_resid_tensor, (None, C), mode)

                # Compute weights of ml factors
                # accum = tl.dot(tl.transpose(C[mode]), C[mode])
                # for i in range(1, len(C)):
                #     accum *= tl.dot(tl.transpose(C[i]), C[i])

                # numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
                # denominator = tl.dot(C0[mode], accum)
                # denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
                # factor = seasonal_factors[mode] * (1 - lr) + seasonal_factors[mode] * lr * numerator / denominator
                # plt.plot(numerator * denominator)
                # plt.title("mttkrp for updating NLDS")
                # plt.show()
                # plt.plot(numerator)
                # plt.title("mttkrp (only numerator) for updating NLDS")
                # plt.show()

                # mttkrp = numerator / denominator
                mttkrp = tl.clip(mttkrp, a_min=epsilon, a_max=None)
                mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
                mttkrp = mttkrp.T

                # plt.plot(mttkrp)
                # plt.title("mttkrp for updating NLDS")
                # plt.show()

                # if (mttkrp < 0).any():
                #     warnings.warn("mttkrp includes negative values")
                # mttkrp = np.maximum(zero, mttkrp)
                # mttkrp = preprocessing.normalize(mttkrp)
                # mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
                # mttkrp = mttkrp.T
                # mttkrp = preprocessing.minmax_scale(mttkrp)
                # print(mttkrp)
                
                # Plotting MTTKRP logs
                # plt.plot(mttkrp)
                # plt.savefig(f'log/mttkrp_{iteration}.png')
                # plt.close()

                A, B, Z, C0 = init_lotka_volterra(
                    mttkrp, max_iter=2, return_sequence=True)
                C[0] = C0

                # plt.plot(C0)
                # plt.title("fitting result of mttkrp")
                # plt.show()

                # plt.plot(numerator)
                # plt.title("mttkrp for updating NLDS")
                # plt.show()

                # A, B, Z, C = fit_latent_lotka_volterra(
                #     C_resid_tensor, A, B, Z, C)

            else:
                C[mode] = grad_update_nn_parafac(
                    C_resid_tensor, C, mode, non_negative=True)
                # C[mode] = fit_observation_matrix(C_resid_tensor, C, mode,
                #     normalize_factors=True)

        # Fit Seasonal Activity
        folded_S_resid_tensor = np.zeros((period, *tensor.shape[1:]))

        for i in range(duration):
            # one_season = tensor[i * period: (i + 1) * period]
            # folded_S_resid_tensor += one_season - one_season.mean(axis=0)
            folded_S_resid_tensor += S_resid_tensor[i*period:(i+1)*period]

        folded_S_resid_tensor /= duration

        # S = fit_seasonal_factors(folded_S_resid_tensor, S)
        _, S = non_negative_parafac(folded_S_resid_tensor, ns, n_iter_max=100)
        # for ncp_iter in range(1000):
        # for mode in range(n_dim):
        #     if mode == 0:
        #         S[mode] = grad_update_nn_parafac(
        #             folded_S_resid_tensor, S, mode, non_negative=False)
        #         # S[mode] += 1e-10
        #         print((S[mode] == 0).sum())
        #     else:
        #         S[mode] = grad_update_nn_parafac(
        #             folded_S_resid_tensor, S, mode, non_negative=True)

            # S[mode] = fit_observation_matrix(folded_S_resid_tensor, S, mode,
            #     non_negative=True, normalize_factors=True)

        # S = init_seasonal_factors(timestamp, folded_S_resid_tensor, period, ns)

        # Reconstruction
        C_recon_tensor = kruskal_to_tensor((None, C))
        # S_recon_tensor = gen_seasonality(timestamp, n_sample, S)
        S_recon_tensor = kruskal_to_tensor((None, S))
        S_recon_tensor = np.tile(S_recon_tensor, (duration, 1, 1))

        # Residual tensor
        C_resid_tensor = tensor - S_recon_tensor
        S_resid_tensor = tensor - C_recon_tensor
        rec_error = tl.norm(
            tensor - C_recon_tensor - S_recon_tensor, 2) / norm_tensor
        rec_errors.append(rec_error)

        print('Iter=', iteration, 'rec_error=', rec_error)

        if iteration > 2:
            if np.abs(rec_errors[-1] - rec_errors[-2]) < tol:
                break
    
    else:
        warnings.warn("DISMO did not converge")

    return A, B, Z, C, S


def create_nlds(timestamp, tensor, ml_factors, s_factors,
                n_iter_max=20, tol=1e-3, learning_rate=0.1):

    # Given:
    n_sample = tensor.shape[0]
    n_dim = tensor.ndim
    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []

    # Generate previous seasonal pattenrs
    S_recon_tensor = gen_seasonality(timestamp, n_sample, s_factors)
    C_resid_tensor = tensor - S_recon_tensor

    # Initialize NLDS
    A, B, Z, ml_factors = update_nlds(C_resid_tensor, ml_factors)
    C_recon_tensor = kruskal_to_tensor((None, ml_factors))
    S_resid_tensor = tensor - C_recon_tensor

    # Optimization
    for iteration in range(n_iter_max):

        # Estimate latent competing activities
        A, B, Z, ml_factors = update_nlds(C_resid_tensor, ml_factors)

        # Smoothly update seasonal factors
        s_factors = smooth_update_seasonal_factors(
            timestamp, S_resid_tensor, s_factors, lr=learning_rate, normalize_factors=False)

        C_recon_tensor = kruskal_to_tensor((None, ml_factors))
        S_recon_tensor = gen_seasonality(timestamp, n_sample, s_factors)
        C_resid_tensor = tensor - S_recon_tensor
        S_resid_tensor = tensor - C_recon_tensor
        rec_error = tl.norm(tensor - C_recon_tensor - S_recon_tensor, 2)
        rec_error /= norm_tensor
        rec_errors.append(rec_error)

        print('Iter=', iteration, 'rec_error=', rec_error)

        if iteration > 2:
            if np.abs(rec_errors[-1] - rec_errors[-2]) < tol:
                break  # convergence criterion 
        
    else:
        warnings.warn("NLDS Creation did not converge")

    # Create new NLDS object
    new_nlds = [A, B, Z]
    return new_nlds, s_factors


def create_ml_factor(timestamp, mode, tensor, nlds, ml_factors, s_factors,
    n_iter_max=500, tol=1e-6, epsilon=10e-12, learning_rate=0.1):

    # Given:
    n_sample = tensor.shape[0]
    n_dim = tensor.ndim
    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []
    A, B, Z = nlds
    nc = len(Z)

    # Generate previous seasonal pattenrs
    S_recon_tensor = gen_seasonality(timestamp, n_sample, s_factors)
    C_resid_tensor = tensor - S_recon_tensor

    ml_factors[0] = np.zeros(
        (tensor.shape[0], ml_factors[-1].shape[-1]))  # time x k
    mttkrp = tl.unfolding_dot_khatri_rao(
        C_resid_tensor, (None, ml_factors), 0)  # time mode
    mttkrp = tl.clip(mttkrp, a_min=epsilon, a_max=None)
    mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
    mttkrp = mttkrp.T

    Z = glv.fit(mttkrp, target='init_state', fixed_params=(A, B))
    ml_factors[0], _ = glv._generate(n_sample, A, B, Z)

    # ml_factors[mode] = np.eye(*ml_factors[mode].shape) + epsilon
    # ml_factors[mode] = np.ones(
    #     (tensor.shape[mode], ml_factors[-1].shape[-1]))  # mode shape x k
    # ml_factors[mode] = np.random.rand(*ml_factors[mode].shape) + 1e-2
    # Update initial values
    # ml_factors[0] = np.ones(
    #     (tensor.shape[0], ml_factors[-1].shape[-1]))  # time x k

    C_recon_tensor = kruskal_to_tensor((None, ml_factors))
    S_resid_tensor = tensor - C_recon_tensor

    # Optimization
    for iteration in range(n_iter_max):

        # Update m-th ml factor
        ml_factors[mode] = grad_update_nn_parafac(
            C_resid_tensor, ml_factors, mode, non_negative=True)

        # print(np.abs(ml_factors[mode] - tmp).sum())
        # Update initial values
        mttkrp = tl.unfolding_dot_khatri_rao(
            C_resid_tensor, (None, ml_factors), 0)  # time mode

        mttkrp = tl.clip(mttkrp, a_min=epsilon, a_max=None)
        mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
        mttkrp = mttkrp.T

        Z = glv.fit(mttkrp, target='init_state', fixed_params=(A, B))
        ml_factors[0], _ = glv._generate(n_sample, A, B, Z)

        # Smoothly update seasonal factors
        s_factors = smooth_update_seasonal_factors(
            timestamp, S_resid_tensor, s_factors,
            lr=learning_rate, normalize_factors=False)

        C_recon_tensor = kruskal_to_tensor((None, ml_factors))
        S_recon_tensor = gen_seasonality(timestamp, n_sample, s_factors)
        C_resid_tensor = tensor - S_recon_tensor
        S_resid_tensor = tensor - C_recon_tensor
        rec_error = tl.norm(tensor - C_recon_tensor - S_recon_tensor, 2)
        rec_error /= norm_tensor
        rec_errors.append(rec_error)

        print('Iter=', iteration, 'rec_error=', rec_error)

        if iteration > 2:
            if np.abs(rec_errors[-1] - rec_errors[-2]) < tol:
                break  # convergence criterion

    else:
        warnings.warn("ML-factor Creation did not converge")

    # Create new NLDS object
    nlds = [A, B, Z]
    return nlds, ml_factors, s_factors


def update_nlds(tensor, ml_factors, epsilon=10e-12, only_linear=False):

    ml_factors[0] = np.ones(
        (tensor.shape[0], ml_factors[-1].shape[-1]))  # time x k

    mttkrp = tl.unfolding_dot_khatri_rao(
        tensor, (None, ml_factors), 0)  # time mode

    # mttkrp = preprocessing.normalize(mttkrp)
    mttkrp = tl.clip(mttkrp, a_min=epsilon, a_max=None)
    mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
    mttkrp = mttkrp.T

    A, B, Z, C0 = init_lotka_volterra(mttkrp,
        return_sequence=True,
        only_linear=only_linear)

    ml_factors[0] = C0

    return A, B, Z, ml_factors


def update_init_state(timestamp, tensor, nlds, ml_factors, s_factors,
                      n_iter_max=20, tol=1e-6, epsilon=10e-12, learning_rate=0.1, only_linear=False):

    # Given:
    n_sample = tensor.shape[0]
    norm_tensor = tl.norm(tensor, 2)
    rec_errors = []
    A, B, Z = nlds

    # Generate previous seasonal pattenrs
    S_recon_tensor = gen_seasonality(timestamp, n_sample, s_factors)
    C_resid_tensor = tensor - S_recon_tensor

    # Initialize NLDS
    # mttkrp = tl.unfolding_dot_khatri_rao(
    #     C_resid_tensor, (None, ml_factors), 0)
    # mttkrp = preprocessing.normalize(mttkrp)
    # accum = 1
    # sub_indices = [i for i in range(len(ml_factors)) if i != 0]
    # for i, e in enumerate(sub_indices):
    #     if i:
    #         accum *= tl.dot(tl.transpose(
    #             ml_factors[e]), ml_factors[e])
    #     else:
    #         accum = tl.dot(tl.transpose(
    #             ml_factors[e]), ml_factors[e])

    ml_factors[0] = np.ones(
        (tensor.shape[0], ml_factors[-1].shape[-1]))  # time x k

    mttkrp = tl.unfolding_dot_khatri_rao(
        C_resid_tensor, (None, ml_factors), 0)

    # numerator = tl.clip(mttkrp, a_min=epsilon, a_max=None)
    # denominator = tl.dot(ml_factors[0], accum)
    # denominator = tl.clip(denominator, a_min=epsilon, a_max=None)
    # mttkrp = ml_factors[0] * numerator / denominator
    mttkrp = tl.clip(mttkrp, a_min=epsilon, a_max=None)
    mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
    mttkrp = mttkrp.T

    Z = glv.fit(mttkrp, target='init_state', fixed_params=(A, B))
    ml_factors[0], _ = glv._generate(n_sample, A, B, Z)

    C_recon_tensor = kruskal_to_tensor((None, ml_factors))
    S_resid_tensor = tensor - C_recon_tensor

    # Optimization
    for iteration in range(n_iter_max):

        # Estimate initial state of latent competing activities
        mttkrp = tl.unfolding_dot_khatri_rao(
            C_resid_tensor, (None, ml_factors), 0)

        # mttkrp = preprocessing.normalize(mttkrp)
        mttkrp = tl.clip(mttkrp, a_min=epsilon, a_max=None)
        mttkrp = mttkrp.T / np.linalg.norm(mttkrp, ord=2, axis=1)
        mttkrp = mttkrp.T

        Z = glv.fit(mttkrp, target='init_state', fixed_params=(A, B))
        ml_factors[0], _ = glv._generate(n_sample, A, B, Z)

        # Smoothly update seasonal factors
        s_factors = smooth_update_seasonal_factors(
            timestamp, S_resid_tensor, s_factors, lr=learning_rate, normalize_factors=False)

        C_recon_tensor = kruskal_to_tensor((None, ml_factors))
        S_recon_tensor = gen_seasonality(timestamp, n_sample, s_factors)
        C_resid_tensor = tensor - S_recon_tensor
        S_resid_tensor = tensor - C_recon_tensor
        rec_error = tl.norm(tensor - C_recon_tensor - S_recon_tensor, 2)
        rec_error /= norm_tensor
        rec_errors.append(rec_error)

        # print('Iter=', iteration, 'rec_error=', rec_error)

        if iteration > 2:
            if np.abs(rec_errors[-1] - rec_errors[-2]) < tol:
                break  # convergence criterion

    else:
        warnings.warn("NLDS Creation did not converge")

    # Create new NLDS object
    new_nlds = [A, B, Z]
    return new_nlds, s_factors

"""
[2] Amnon Shashua and Tamir Hazan,
       "Non-negative tensor factorization with applications to statistics and computer vision",
       In Proceedings of the International Conference on Machine Learning (ICML),
       pp 792-799, ICML, 2005
"""
