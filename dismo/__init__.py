
from copy import deepcopy
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import norm
import tensorly as tl
from tqdm import tqdm
from tensorly.decomposition import non_negative_parafac, parafac

from . import glv
from . import utils


def unfolding_dot_khatri_rao(tensor, factors, mode):
    return tl.unfolding_dot_khatri_rao(tensor, (None, factors), mode)


def kruskal_to_tensor(factors):
    return tl.kruskal_to_tensor((None, factors))


def predict_seasonal_tensor(factors, n_sample, t=0):
    n_season = factors[0].shape[0]
    n_fold = n_sample // n_season + 1
    pred = tl.kruskal_to_tensor((None, factors))
    pred = np.tile(pred, (n_fold, *[1] * (len(factors) - 1)))
    return np.roll(pred, -np.mod(t, n_season), axis=0)[:n_sample]


def compute_accum(factors, skip_matrix):
    accum = 1.
    for i, factor in enumerate(factors):
        if i == skip_matrix: continue
        accum *= tl.dot(tl.transpose(factor), factor)

    return accum


def compute_seasonal_mean_tensor(tensor, period, t=0, remove_temporal_mean=True):

    n_sample    = tensor.shape[0]
    n_dims      = tensor.shape[1:]
    mean_tensor = np.zeros((period, *n_dims))  # output

    n_section   = n_sample // period
    season_ids  = np.arange(t, t + n_sample, 1) % period
    diff_ids    = n_sample - period * n_section
    start_point = np.where(season_ids==0)[0][0]

    rolled_tensor = np.roll(tensor, -start_point, axis=0)
    if diff_ids > 1:
        rolled_tensor = rolled_tensor[:-diff_ids]

    for w in range(n_section):
        one_period = rolled_tensor[w * period: (w+1) * period]
        mean_tensor += one_period
        if remove_temporal_mean:
            mean_tensor -= one_period.mean(axis=0)

    mean_tensor /= n_section
        
    return mean_tensor


def init_seasonal_factors(tensor, rank, period, t=0, n_iter_max=100, tol=1e-8,
                          non_negative=True, random_state=None):

    if period == 0: return None
    mean_tensor = compute_seasonal_mean_tensor(tensor, period, t)

    if non_negative:
        _, factors = non_negative_parafac(
            mean_tensor, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state)
    else:
        _, factors = parafac(
            mean_tensor, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state)

    return factors


def init_interaction_factors(tensor, rank, n_iter_max=100, tol=1e-8, random_state=None):
    _, factors = non_negative_parafac(
        tensor, rank, n_iter_max=n_iter_max, tol=tol, random_state=random_state)

    return factors


# Initialization in the static dismo solver
def initialize(tensor, n_interaction, n_seasonality, n_season, t=0,
               non_negative_seasonality=True,
               n_iter_max=100, tol=1e-8, random_state=None):

    if n_season > 0:
        seas_factors = init_seasonal_factors(
            tensor, n_seasonality, n_season, t,
            n_iter_max=n_iter_max, tol=tol,
            non_negative=non_negative_seasonality,
            random_state=random_state)
        seas_tensor = predict_seasonal_tensor(seas_factors, len(tensor), t=t)

    else:
        # Assume no seasonality
        seas_factors = None
        seas_tensor = np.zeros(tensor.shape)

    mlds_factors = init_interaction_factors(
        tensor - seas_tensor, n_interaction,
        n_iter_max=n_iter_max, tol=tol, random_state=random_state)

    mlds_tensor = tl.kruskal_to_tensor((None, mlds_factors))

    return seas_factors, mlds_factors, seas_tensor, mlds_tensor


def update_linear_projection(
    tensor, factors, mode, learning_rate=1, epsilon=10e-12, normalization=True):

    # Take a gradient step of NCP
    mttkrp = tl.unfolding_dot_khatri_rao(tensor, (None, factors), mode)
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
    # Update
    factor = factors[mode] * learning_rate * numerator / denominator

    # Normalization
    if normalization:
        weights = tl.norm(factor, order=2, axis=0)
        weights = tl.where(
            tl.abs(weights) <= tl.eps(tensor.dtype),
            tl.ones(tl.shape(weights),
            **tl.context(factors[0])),
            weights)

        return factor / (tl.reshape(weights, (1, -1)))
    else:
        return factor


def update_seasonal_projection(
    tensor, factors, mode, non_negative=True,
    learning_rate=1, epsilon=10e-12, normalize_factor=True):

    # Take a gradient step of NCP
    mttkrp = tl.unfolding_dot_khatri_rao(tensor, (None, factors), mode)

    if non_negative:
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

        # Update
        factor = factors[mode] * learning_rate * numerator / denominator

    else:
        n_components = factors[-1].shape[-1]
        pseudo_inverse = tl.tensor(np.ones(
            (n_components, n_components)), **tl.context(tensor))
        
        for i, factor in enumerate(factors):
            if not i == mode:
                pseudo_inverse *= tl.conj(tl.dot(tl.transpose(factor), factor))

        pseudo_inverse += np.diag(np.full(n_components, 1e-10))
        factor = tl.transpose(tl.solve(tl.transpose(pseudo_inverse), tl.transpose(mttkrp)))

    # Normalization
    if normalize_factor:
        weights = tl.norm(factor, order=2, axis=0)
        weights = tl.where(
            tl.abs(weights) <= tl.eps(tensor.dtype),
            tl.ones(tl.shape(weights),
            **tl.context(factors[0])),
            weights)

        factor = factor / (tl.reshape(weights, (1, -1)))

    return factor
    

def solve_dismo(X, n_dim_c, n_dim_s=0, n_season=0, t=0,
    max_iter=10, tol=1e-4, n_trial=3, max_time=3,
    interaction_type='full', fit_self_interaction=False,
    use_carrying_capacity=True, normalize_ml_projection=False,
    non_negative_seasonality=True, random_state=None, verbose=1):

    n_sample = X.shape[0]
    n_modes  = X.ndim
    X_norm   = tl.norm(X, 2)
    t_eval   = np.linspace(0, 1, n_sample)

    rec_errors = []

    S, C, Y, Z = initialize(X, n_dim_c, n_dim_s, n_season, t,
        non_negative_seasonality=non_negative_seasonality,
        random_state=random_state)

    for iteration in range(max_iter):
        XZ = X - Z  # residual tensor to update S
        XY = X - Y  # residual tensor to update C

        # Update Linear Projections
        for mode in range(1, n_modes):
            for _ in range(1):
                C[mode] = update_linear_projection(XY, C, mode,
                    normalization=normalize_ml_projection)
        
        # Update Latent Interactions
        mttkrp = update_linear_projection(XY, C, 0,
            normalization=normalize_ml_projection)

        theta = glv.GLVC(
            interaction_type=interaction_type,
            fit_self_interaction=fit_self_interaction,
            use_carrying_capacity=use_carrying_capacity)

        converged = theta.fit(mttkrp, t_eval=t_eval,
            n_trial=n_trial, max_time=max_time, try_catch=True)
        if not converged: break

        if verbose == 2:
            print("z=", theta.z0)
            print("r=", theta.r)
            print("K=", theta.K)
            print("A=\n", theta.A)

        C[0] = glv.generate_glvc_data(theta.z0, t_eval, theta.r, theta.A, theta.K)

        if S is not None:
            mean_XZ = compute_seasonal_mean_tensor(XZ, n_season, t=t)
            for mode in range(n_modes):
                S[mode] = update_seasonal_projection(
                    mean_XZ, S, mode,
                    non_negative=non_negative_seasonality,
                    normalize_factor=False)

            Y = predict_seasonal_tensor(S, n_sample, t=t)

        Z = kruskal_to_tensor(C)

        # Check convergence

        rec_error = tl.norm(X - Y - Z, 2) / X_norm
        rec_errors.append(rec_error)

        if verbose > 0:
            print(f"Iter= {iteration+1} \trec_error= {rec_error:.4f}")

        if iteration > 1:
            if np.abs(rec_errors[-1] - rec_errors[-2]) < tol:
                print('converged!')
                break
    
    return theta, C, S


# Temporal fitting functions

def fit_init_states(tensor, theta, factors, max_iter=5, tol=1e-4, n_trial=5, max_time=5):

    norm_tensor = tl.norm(tensor, 2)
    factors[0] = np.full(factors[0].shape, 1.)
    prev_error = np.inf

    for _ in range(max_iter):
        init_time_factor = update_linear_projection(tensor, factors, 0, normalization=False)
        _, r, A, K = theta.get_params()
        converged = theta.fit(init_time_factor,
            r=r, A=A, K=K, z0=init_time_factor[0],
            est_vars='z0', n_trial=n_trial, max_time=max_time)

        factors[0] = glv.generate_glvc_data(
            theta.z0, np.linspace(0, 1, len(tensor)), r, A, K) 

        pred = kruskal_to_tensor(factors)
        error = tl.norm(tensor - pred, 2) / norm_tensor
        if np.abs(error - prev_error) < tol:
            print("Early stopping @ fit init states")
            break
        else:
            prev_error = error
  
    return theta, factors


def fit_nlds(tensor, factors, max_iter=10, tol=1e-4, n_trial=5, max_time=5,
             interaction_type='full', fit_self_interaction=False,
             use_carrying_capacity=True):

    norm_tensor = tl.norm(tensor, 2)
    factors[0] = np.full(factors[0].shape, 1.)
    prev_error = np.inf
    
    for _ in range(max_iter):
        init_time_factor = update_linear_projection(tensor, factors, 0, normalization=False)
        theta = glv.GLVC(interaction_type=interaction_type,
                            fit_self_interaction=fit_self_interaction,
                            use_carrying_capacity=use_carrying_capacity)

        converged = theta.fit(init_time_factor, n_trial=n_trial, max_time=max_time)
        factors[0] = glv.generate_glvc_data(
            theta.z0, np.linspace(0, 1, len(tensor)), theta.r, theta.A, theta.K)

        pred = kruskal_to_tensor(factors)
        error = tl.norm(tensor - pred, 2) / norm_tensor
        if np.abs(error - prev_error) < tol:
            print("Early stopping @ fit nlds")
            break
        else:
            prev_error = error
        
    return theta, factors


def fit_obsmat(tensor, mode, theta, factors, tol=1e-4,
               n_trial=5, max_time=5, max_iter=5, verbose=0):

    len_tensor = len(tensor)
    norm_tensor = tl.norm(tensor, order=2)
    factors[0] = np.full((len_tensor, len(theta.z0)), 1.)
    init_time_factor = update_linear_projection(tensor, factors, 0, normalization=False)
    _, r, A, K = theta.get_params()  # fixed parameters
    prev_error = np.inf

    for iteration in range(max_iter):
        # Update the initial states
        converged = theta.fit(init_time_factor,
                            r=r, A=A, K=K, est_vars='z0',
                            n_trial=n_trial, max_time=max_time)
        # Generate latent activities
        factors[0] = glv.generate_glvc_data(
            theta.z0, np.linspace(0, 1, len_tensor), r, A, K)
        # Update m-th mode projection
        factors[mode] = update_linear_projection(tensor, factors, mode, normalization=False)

        pred = kruskal_to_tensor(factors)
        error = tl.norm(tensor - pred, order=2) / norm_tensor
        if np.abs(error - prev_error) < tol:
            print("Early stopping @ fit obsmat")
            break
        else:
            prev_error = error
        if verbose:
            print('Iter=', iteration, 'Err=', error)        

    return factors[mode]


class DISMO:
    def __init__(self, n_dims, n_season=0,
                 minc=2, maxc=8, mins=2, maxs=8,
                 interaction_type='full',
                 use_self_interaction=False,
                 use_carrying_capacity=True,
                 n_trial=5, max_time=5, 
                 online_max_iter=5,
                 non_negative_seasonality=False,
                 normalize_ml_projection=False,
                 regime_shift=True,
                 init_complemenatry_matrices=False,
                 save_train_log=True,
                 verbose=0):

        self.n_dims = n_dims
        self.n_modes = len(n_dims)
        self.n_season = n_season
        self.minc, self.maxc = minc, maxc
        self.mins, self.maxs = mins, maxs

        self.nlds = []
        self.obsmat = [[] for _ in range(self.n_modes + 1)]
        self.seasonality = None
        self.regime_history = []
        self.decision_hisotry = []
        self.elapsed_times = []
        self.P = None  # unfolding_dot_khatri_rao
        self.Q = None  # accum

        # Options

        self.n_trial = n_trial  # maximum tiral in NLDS fitting
        self.max_time = max_time
        self.online_max_iter = online_max_iter
        self.interaction_type = interaction_type
        self.use_self_interaction = use_self_interaction
        self.use_carrying_capacity = use_carrying_capacity
        self.non_negative_seasonality = non_negative_seasonality
        self.normalize_ml_projection = normalize_ml_projection
        self.regime_shift = regime_shift
        self.init_complementary_matrices = init_complemenatry_matrices
        self.save_train_log = save_train_log
        self.verbose = verbose

    def num_nlds(self):
        return len(self.nlds)

    def num_obsmat(self, mode):
        return len(self.obsmat[mode]) # m_k
    
    def get_nlds(self, index):
        return deepcopy(self.nlds[index])  # theta

    def get_obsmat(self, index, mode):
        return deepcopy(self.obsmat[mode][index]) # U,V,...

    def get_factors(self, ids, n_sample=0):
        factors = [None] if n_sample == 0 else [np.zeros((n_sample, self.c))]
        factors += [self.get_obsmat(ids[mode], mode) for mode in range(1, self.n_modes + 1)]
        return factors # M matrices

    def get_seasonality(self):
        return deepcopy(self.seasonality) # M + 1 matrices

    def get_last_regime_ids(self):
        return self.regime_history[-1]

    def get_regime_history(self):
        return np.array(self.regime_history, dtype=int)

    def grid_search(self, tensor, t=0, max_iter=10):

        n_samples = tensor.shape[0]
        c_set = range(self.minc, self.maxc + 1)
        s_set = range(self.mins, self.maxs + 1)
        candidates = list(product(c_set, s_set))
        scores = []

        for c, s in tqdm(candidates, desc='GridSearch'):
            theta, C, S = self.fit(tensor, c, s, t=0, max_iter=max_iter)
            recon_ = tl.kruskal_to_tensor((None, C))
            if S is not None:
                recon_ += predict_seasonal_tensor(S, n_samples, t=t)
            
            score = self.coding_cost(tensor, recon_)
            score += self.model_cost(self.n_modes + 1, theta, C, S, normalize=True)
            print('c=', c, 's=', s, 'cost=', score)
            scores.append(score)
        
        # Set best numbers of components
        opt_c, opt_s = candidates[np.argmin(scores)]
        self.c = opt_c
        self.s = opt_s
        print("best # interaction=", opt_c)
        print("best_# seasonality=", opt_s)

        return scores

    def initialize(self, tensor, c=None, s=None, t=0, max_iter=10, tol=1e-3, verbose=1):
        self.c = c = self.c if c is None else c
        self.s = s = self.s if s is None else s

        # Estimate full parameters
        theta, C, S = self.fit(tensor, c, s, t, max_iter, tol, verbose)

        # Initialize parameter sets
        self.nlds.append(theta)
        for mode in range(1, self.n_modes + 1):
            self.obsmat[mode].append(C[mode])

        self.seasonality = S
        self.regime_history.append([0, 0, 0])

    def fit(self, tensor, c, s=0, t=0, max_iter=10, tol=1e-3, verbose=1):

        # Validate parameters

        if c < self.minc: c = self.minc
        if c > self.maxc: c = self.maxc
        if s > self.maxs: s = self.maxs

        # Factorization
        theta, C, S = solve_dismo(tensor, c, s,
            n_season=self.n_season,
            t=t,
            max_iter=max_iter,
            tol=tol,
            interaction_type=self.interaction_type,
            fit_self_interaction=self.use_self_interaction,
            use_carrying_capacity=self.use_carrying_capacity,
            normalize_ml_projection=self.normalize_ml_projection,
            non_negative_seasonality=self.non_negative_seasonality,
            verbose=verbose)

        if self.init_complementary_matrices:
            print("Update seasonality")
            resid_tensor = tensor - kruskal_to_tensor(C)
            mean_tensor = compute_seasonal_mean_tensor(resid_tensor, self.n_season, t=t)
            print("mean_tensor", mean_tensor.shape)
            self.P = [unfolding_dot_khatri_rao(mean_tensor, S, mode) for mode in range(len(S))]
            self.Q = [compute_accum(S, mode) for mode in range(len(S))]

        return theta, C, S

    @staticmethod
    def coding_cost(X, Y):
        diff = (X - Y).ravel()
        # print('mean=', diff.mean(), 'std=', diff.std())
        prob = norm.pdf(diff, loc=diff.mean(), scale=diff.std())
        return -1 * np.log2(prob).sum()

    @staticmethod
    def count_nonzero(X, epsilon):
        return np.count_nonzero(np.logical_or(X < -epsilon, epsilon < X))

    @staticmethod
    def compute_score(arr, count, float_cost):
        return count * (np.log(arr.shape).sum() + float_cost)

    def model_cost_arr(self, arr, epsilon, float_cost, normalize=True):
        if not normalize:
            return self.compute_score(arr, self.count_nonzero(arr, epsilon), float_cost)
        else:
            return arr.shape[1] * self.compute_score(arr, self.count_nonzero(arr, epsilon), float_cost) / arr.shape[0]

    def model_cost(self, n_mode, theta=None, C=None, S=None, float_cost=32, epsilon=1e-4, normalize=True):

        score = np.log2(2 + n_mode) # c, s, M

        # GLVC
        if theta is not None:
            c = len(theta.r)
            score += self.count_nonzero(theta.r, epsilon) * (np.log(c) + float_cost)
            score += self.count_nonzero(theta.K, epsilon) * (np.log(c) + float_cost)
            score += self.count_nonzero(theta.A, epsilon) * (np.log(c) * 2 + float_cost)

        # Projection
        if C is not None:
            for mode, M in enumerate(C):
                if mode == 0: continue
                score += self.model_cost_arr(M, epsilon, float_cost, normalize=normalize)

        # Seasonality
        if S is not None:
            for M in S:
                score += self.model_cost_arr(M, epsilon, float_cost, normalize=normalize)

        # return score / n_mode
        return score

    def total_model_cost(self):
        return

    def _find_nlds(self, method, tensor, t):

        # initialization

        index = next_theta = None
        coding_cost = model_cost = np.inf
        n_sample = tensor.shape[0]
        last_ids = self.get_last_regime_ids()
        C = self.get_factors(last_ids, n_sample=n_sample)

        if method == 'stay':
            # Try to keep using the last regime
            index = last_ids[0]
            theta = self.get_nlds(index)
            theta, C = fit_init_states(tensor, theta, C,
                n_trial=self.n_trial,
                max_time=self.max_time,
                max_iter=self.online_max_iter)

            recon_ = self.predict(t, C)
            coding_cost = self.coding_cost(tensor, recon_)  # TODO
            model_cost = 0
            next_theta = theta

        elif method == 'select':
            # Try to switch to either of existing regimes
            tmp_score = []

            for rindex, theta in enumerate(self.nlds):
                if rindex == last_ids[0]: continue
                theta, C = fit_init_states(tensor, theta, C,
                    n_trial=self.n_trial,
                    max_time=self.max_time,
                    max_iter=self.online_max_iter)
                recon_ = self.predict(t, C)
                score = self.coding_cost(tensor, recon_)
                tmp_score.append(score)

            if tmp_score: # choose the best
                index = np.argmin(tmp_score)
                coding_cost = tmp_score[index]
                model_cost = 0
                next_theta = self.get_nlds(index)

        elif method == 'generate':
            # Try to generate a new regime to switch
            next_theta, C = fit_nlds(tensor, C,
                n_trial=self.n_trial,
                max_time=self.max_time,
                max_iter=self.online_max_iter,
                interaction_type=self.interaction_type,
                fit_self_interaction=self.use_self_interaction,
                use_carrying_capacity=self.use_carrying_capacity)

            index = self.num_nlds()
            recon_ = self.predict(t, C)
            coding_cost = self.coding_cost(tensor, recon_)
            model_cost = self.model_cost(self.n_modes + 1, theta=next_theta)

        return next_theta, index, coding_cost, model_cost

    def _find_obsmat(self, method, mode, tensor, t, epsilon=1e-4, float_cost=32):
        
        n_samples = tensor.shape[0]
        last_ids = self.get_last_regime_ids()
        theta = self.get_nlds(last_ids[0])

        next_obsmat = index = None
        coding_cost = model_cost = np.inf

        if method == 'select':
            # Search for the bset projection in existing regimes
            tmp_score = []

            for rindex, obsmat in enumerate(self.obsmat[mode]):
                if rindex == last_ids[mode]: continue
                C = self.get_factors(last_ids, n_sample=n_samples)
                C[mode] = obsmat
                _, C = fit_init_states(tensor, theta, C,
                    n_trial=self.n_trial, max_time=self.max_time,
                    max_iter=self.online_max_iter)

                recon_ = self.predict(t, C)
                score = self.coding_cost(tensor, recon_)
                tmp_score.append(score)
                del C
            
            if tmp_score: # is not empty
                index = np.argmin(tmp_score)
                coding_cost = tmp_score[index]
                model_cost = 0
                next_obsmat = self.get_obsmat(index, mode)

        elif method == 'generate':
            # Iterate updateting the initial state and m-th mode projection
            C = self.get_factors(last_ids, n_sample=n_samples)
            C[mode] = fit_obsmat(tensor, mode, theta, C,
                n_trial=self.n_trial, max_time=self.max_time,
                max_iter=self.online_max_iter)

            index = self.num_obsmat(mode)
            next_obsmat = C[mode]
            recon_ = self.predict(t, C)
            coding_cost = self.coding_cost(tensor, recon_)
            # model_cost = self.model_cost(self.n_modes + 1, C=[next_obsmat])
            model_cost = self.model_cost_arr(next_obsmat, epsilon, float_cost)

        return next_obsmat, index, coding_cost, model_cost

    def _find_candidates(self, tensor, t):
        print("Searching for candidate regimes...")
        result = []  # mode, score, index, status
        params = []  # estimated params for each mode

        # 1. Fit initial states and generate latent dynamics

        param, index, coding_cost, model_cost = self._find_nlds('stay', tensor, t)
        result.append([None, index, coding_cost, model_cost, 'stay'])

        # 2. Try switching either of modes

        mode = 0  # time mode
        for method in ['select', 'generate']:
            print('mode= {}; method= {}'.format(mode, method))
            param, index, coding_cost, model_cost = self._find_nlds(method, tensor, t)
            result.append([mode, index, coding_cost, model_cost, method])
            params.append(param)

        for mode in range(1, self.n_modes + 1):
            for method in ['select', 'generate']:
                print('mode= {}; method= {}'.format(mode, method))
                param, index, coding_cost, model_cost = self._find_obsmat(method, mode, tensor, t)
                result.append([mode, index, coding_cost, model_cost, method])
                params.append(param)

        result = pd.DataFrame(result,
            columns=['mode', 'rindex', 'costC', 'costM', 'status'])

        return result, params

    def update(self, tensor, t):
        
        last_ids = self.get_last_regime_ids()
        curr_ids = deepcopy(last_ids)

        resid_tensor = tensor.copy()
        seas_factors = self.get_seasonality()

        if seas_factors is not None:
            seas_tensor = predict_seasonal_tensor(seas_factors, len(tensor), t=t)
            resid_tensor -= seas_tensor

        if self.regime_shift:
            result, params = self._find_candidates(resid_tensor, t)
            result['costT'] = result['costC'] + result['costM']
            best_decision = result[result.costT == result.costT.min()]
            self.decision_hisotry.append(best_decision['status'])

            print(result, '\n')
            print(best_decision)

            if best_decision['status'].iloc[0] in ['generate', 'select']:
                # Update current factor indices #
                # ----------------------------- #
                best_decision_mode = int(best_decision['mode'].iloc[0])
                curr_ids[best_decision_mode] = int(best_decision['rindex'].iloc[0])

            if best_decision['status'].iloc[0] == 'generate':
                # Add a new factor #
                # ---------------- #
                estimated_param = params[best_decision.index.values[0] - 1]
                if best_decision_mode == 0:
                    self.nlds.append(estimated_param)
                else:
                    self.obsmat[best_decision_mode].append(estimated_param)

        self.regime_history.append(curr_ids)
        print("Current regime set=", curr_ids)

        # Update seasonal factor with the initial state
        # Set init_complementary_matrices = True for this online update
        if self.init_complementary_matrices:
            # Compute residual tensor to update seaonal factors
            C = self.get_factors(curr_ids, n_sample=len(tensor))
            theta = self.get_nlds(curr_ids[0])
            theta, C = fit_init_states(resid_tensor, theta, C,
                n_trial=self.n_trial,
                max_time=self.max_time,
                max_iter=self.online_max_iter)

            # Online update of CP decomposition
            seas_resid_tensor = tensor - kruskal_to_tensor(C)
            seas_mean_tensor = compute_seasonal_mean_tensor(
                seas_resid_tensor, self.n_season, t, remove_temporal_mean=False)

            self.online_update_seasonality(seas_mean_tensor, seas_factors)
            
    def online_update_seasonality(self, tensor, seas_factors, forgetting_rate=0.1):
        for _ in range(1):
            for mode in reversed(range(len(seas_factors))):
                # Update complementary matrices
                self.P[mode] += unfolding_dot_khatri_rao(tensor, seas_factors, mode)
                self.Q[mode] += (seas_factors[mode].T @ seas_factors[mode]) @ compute_accum(seas_factors, skip_matrix=mode)
                # self.P[mode] = forgetting_rate * self.P[mode] + (1 - forgetting_rate) * unfolding_dot_khatri_rao(tensor, seas_factors, mode)
                # self.Q[mode] = forgetting_rate * self.Q[mode] + (1 - forgetting_rate) * compute_accum(seas_factors, skip_matrix=mode)

                # Online update
                seas_factors[mode] = tl.transpose(
                    tl.solve(tl.transpose(self.Q[mode]),
                            tl.transpose(self.P[mode])))
        
        return seas_factors

    def predict(self, t, C, S=None, forecast_step=0):
        # Interaction
        pred = kruskal_to_tensor(C)
        # Seasonality (optional)
        if S is not None: pred += predict_seasonal_tensor(S, len(pred), t=t)
        return pred

    def fit_predict(self, forecast_step, tensor, t, regime_ids=None,
                    return_model=False, return_latent_dynamics=False,
                    return_dynamics=False, return_full_sequence=False):
        """
            forecast_step: length for forecasting ahead
            tensor: current window
            t: start timepoint of the tensor
        """
        if regime_ids is None:
            regime_ids = self.get_last_regime_ids()

        n_samples = tensor.shape[0]

        # Prepare model parameters
        theta = self.get_nlds(regime_ids[0])
        factors = self.get_factors(regime_ids, n_sample=n_samples)
        seas_factors = self.get_seasonality()

        # Filtering out seasonal effects (optional)
        if seas_factors is not None:
            pred_seas = predict_seasonal_tensor(seas_factors, n_samples + forecast_step, t=t)
            resid_tensor = tensor - pred_seas[:n_samples]
        else:
            resid_tensor = tensor

        # Fitting the initial states for a given tensor
        theta, factors = fit_init_states(resid_tensor, theta, factors,
            n_trial=self.n_trial, max_time=self.max_time)

        # forecasting
        t_eval = np.linspace(0, 1, n_samples)
        t_pred = t_eval[1] * np.arange(0, n_samples + forecast_step)
        print(t_eval[-1], t_pred[n_samples])
        factors[0] = glv.generate_glvc_data(
            theta.z0, t_pred, theta.r, theta.A, theta.K)

        pred_intr = kruskal_to_tensor(factors)
        pred = pred_intr.copy()
        if seas_factors is not None:
            pred += pred_seas

        if not return_full_sequence:
            pred = pred[-forecast_step:]
            pred_intr = pred_intr[-forecast_step:]

        outputs = [pred]
        if return_dynamics:
            outputs.append(pred_intr)
            print(pred_intr.shape)
        if return_latent_dynamics:
            outputs.append(factors[0])
            print(factors[0].shape)
        if return_model:
            outputs.append(theta)

        return outputs

    def save(self, outdir):
        """ Save all parameters in dismo """

        np.savetxt(outdir + '/regime_history.txt.gz', self.regime_history)
        
        for i, theta in enumerate(self.nlds):
            np.savetxt(outdir + f'/r_{i}.txt.gz', theta.r)
            np.savetxt(outdir + f'/K_{i}.txt.gz', theta.K)
            np.savetxt(outdir + f'/A_{i}.txt.gz', theta.A)

        for i, obsmat_set in enumerate(self.obsmat):
            for j, obsmat in enumerate(obsmat_set):
                np.savetxt(outdir + f'/W_{i}_{j}.txt.gz', obsmat)

        if self.seasonality is not None:
            for i, obsmat in enumerate(self.seasonality):
                np.savetxt(outdir + f'/S_{i}.txt.gz', obsmat)
        
