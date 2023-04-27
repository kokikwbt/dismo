""" Generalized Lotka-Volterra Equation
    ===================================

    References
    ==========
    * https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html
    * https://pundit.pratt.duke.edu/wiki/Python:Ordinary_Differential_Equations
    * https://python.atelierkobato.com/integrate/
    About the equations
    * https://en.wikipedia.org/wiki/Competitive_Lotka%E2%80%93Volterra_equations
    * https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation
    * https://www.hindawi.com/journals/ddns/2021/9935127/
    * https://en.wikipedia.org/wiki/Generalized_Lotka%E2%80%93Volterra_equation
"""

from copy import deepcopy
import contextlib
from itertools import product
import os
import sys
import time
from numpy.core.numeric import full
from tqdm import trange
from termcolor import colored
import warnings

import numpy as np
import lmfit
from scipy.integrate import solve_ivp, RK45, odeint
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import mean_squared_error
import tensorly as tl
from tensorly import kruskal_to_tensor


BOUND = 1e+2


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd

@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
       stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied: 
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied


def random_initialize(n_dims):
    A = np.random.randn(n_dims)
    B = np.random.randn(n_dims, n_dims)
    return A, B


def de_glvc(t, s, r, A):
    ndim = len(s)
    diff = np.zeros(ndim)
    for i in range(ndim):
        diff[i] = r[i] * s[i] * (1 - np.dot(A[i], s))
    return diff


def de_glvc_K(t, s, r, A, K):
    ndim = len(s)
    diff = np.zeros(ndim)
    for i in range(ndim):
        diff[i] = r[i] * s[i] * (1 - np.dot(A[i], s) / K[i])
    return diff


# Setting for scipy.integrate`.odeint
def my_odeint(func, y0, t, args):
    with stdout_redirected():
        return odeint(func,
                      y0,
                      t,
                      args=args,
                      rtol=1e-4,
                      atol=1e-4,
                      full_output=0,
                      printmessg=False,
                      tfirst=True)


def generate_glvc_data(y0, t, r, A, K=None, scale=0):
    # Generate GLVC sequence
    if K is None:
        data = my_odeint(de_glvc, y0, t, args=(r, A))
    else:
        data = my_odeint(de_glvc_K, y0, t, args=(r, A, K))
    # Add Gaussian noise
    if scale > 0: data += np.random.normal(0, scale, data.shape)
    return data.clip(min=0.)


def generate_glvc_data_sub(X, y0, t, r, A, K=None, scale=0, n_trial=0, metric='rmse'):
    val_met = np.inf
    optY = None

    for _ in range(n_trial):
        Y = generate_glvc_data(y0, t, r, A, K, scale)
        
        if metric == 'rmse':
            met = np.sqrt(mean_squared_error(X.ravel(), Y.ravel()))
        
        if met < val_met:
            val_met == met
            optY = Y
        
    return optY

class GLVC:
    def __init__(self,
        interaction_type='full',
        fit_self_interaction=False,
        use_carrying_capacity=True,
        epsilon=1e-8):

        assert interaction_type in ['full', 'competing', 'self']
        self.interaction_type = interaction_type
        self.fit_self_interaction = fit_self_interaction
        self.use_carrying_capacity = use_carrying_capacity
        self.epsilon = epsilon
    
    # Generalized Lotka-Volterra Competition model
    # without the carrying capacity
    def glvc(s, t, r, A):
        return de_glvc(s, t, r, A)

    def generate(n_samples, r, A, z0):
        n_dims = z0.shape[0]
        args = (r, A)
        t = np.linspace(0, 1, n_samples)
        data = my_odeint(de_glvc, z0, t, args=args)
        return data

    def get_params(self):
        if self.use_carrying_capacity:
            return self.z0, self.r, self.A, self.K
        else:
            return self.z0, self.r, self.A

    def init_params(self, r=None, A=None, K=None, z0=None, maxA=1, est_vars=None):
        params = lmfit.Parameters()

        # Growth rates

        if r is None: r = np.random.rand(self.n_dims)
        vary = True if 'r' in est_vars else False
        for i in range(self.n_dims):
            params.add(f'r{i}', value=r[i], vary=vary, min=0)

        # Interaction coefficients

        if A is None:
            # A = np.random.rand(self.n_dims, self.n_dims)
            A = np.diag(np.ones(self.n_dims))
        # if not self.fit_self_interaction:
        #     A[np.diag_indices(self.n_dims)] = 1

        if 'A' in est_vars:
            if self.fit_self_interaction:
                vary = np.full((self.n_dims, self.n_dims), True)
            else:
                vary = np.full((self.n_dims, self.n_dims), True)
                vary[np.diag_indices(self.n_dims)] = False
        else:
            vary = np.full((self.n_dims, self.n_dims), False)

        if self.interaction_type == 'full':
            minA = -maxA
        elif self.interaction_type == 'competing':
            minA = 0
        elif self.interaction_type == 'self':
            A = np.diag(np.diag(A))  # off-diagonal elements are zero
            vary = np.diag(np.diag(vary))
            minA = -maxA
        else:
            raise ValueError

        for i, j in product(range(self.n_dims), repeat=2):
            params.add(f'A{i}{j}', value=A[i, j], vary=vary[i, j], min=minA, max=maxA)

        # Carrying capacity

        if self.use_carrying_capacity:
            if K is None: K = np.random.rand(self.n_dims).clip(min=self.epsilon)
            vary = True if 'K' in est_vars else False
            for i in range(self.n_dims):
                params.add(f'K{i}', value=K[i], vary=vary, min=self.epsilon)

        # Initial state

        if z0 is None:
            # z0 = np.random.rand(self.n_dims)
            # z0 = z0.clip(min=self.epsilon)
            z0 = np.full(self.n_dims, 0.1)

        vary = True if 'z0' in est_vars else False
        for i in range(self.n_dims):
            # max value depends on normalization
            params.add(f'z0{i}', value=z0[i], vary=vary, min=self.epsilon)

        self.r, self.A, self.K, self.z0 = r, A, K, z0

        return params

    @staticmethod
    def params2numpy(params, r, A, z0):
        """
            params: lmfit.Parameters object
        """
        for i in range(len(r)):
            r[i] = params[f'r{i}']
            z0[i] = params[f'z0{i}']
            for j in range(len(r)):
                A[i, j] = params[f'A{i}{j}']

        return r, A, z0

    @staticmethod
    def params2numpy_withK(params, r, A, K, z0):
        for i in range(len(r)):
            r[i] = params[f'r{i}']
            K[i] = params[f'K{i}']
            z0[i] = params[f'z0{i}']
            for j in range(len(r)):
                A[i, j] = params[f'A{i}{j}']

        return r, A, K, z0

    @staticmethod
    def residual(params, data, self, t0):
        # Update GLVC parameters
        r, A, z0 = self.params2numpy(params, self.r, self.A, self.z0)
        pred = generate_glvc_data(z0, self.t, r, A)
        return (data - pred).ravel()

    @staticmethod
    def residual_withK(params, data, self, t0):
        # Update GLVC parameters
        r, A, K, z0 = self.params2numpy_withK(params, self.r, self.A, self.K, self.z0)
        pred = generate_glvc_data(z0, self.t, r, A, K)
        return (data - pred).ravel()

    @staticmethod
    def residual_withK_obsmat(params, data, self, t0):
        r, A, K, z0 = self.params2numpy_withK(params, self.r, self.A, self.K, self.z0)
        pred = generate_glvc_data(z0, self.t, r, A, K) @ self.obsmat.T
        return (data - pred).ravel()

    @staticmethod
    def residual_withK_tensor(params, data, self, t0):
        r, A, K, z0 = self.params2numpy_withK(params, self.r, self.A, self.K, self.z0)
        self.C[0] = generate_glvc_data(z0, self.t, r, A, K)
        pred = tl.kruskal_to_tensor((None, self.C))
        return (data - pred).ravel()

    def fit(self, X, r=None, A=None, K=None, z0=None, C=None, obsmat=None, t_eval=None,
            est_vars=None, maxA=2, n_trial=1, max_time=1, try_catch=True, verbose=0):

        if C is None and obsmat is None:
            self.n_dims = X.shape[1]
            # X += 1e-8  # clip near to zero
            if self.use_carrying_capacity:
                resid_func = self.residual_withK
            else:
                resid_func = self.resdiual

        if C is not None:
            self.n_dims = C[-1].shape[-1]
            self.C = C
            resid_func = self.residual_withK_tensor
        
        if obsmat is not None:
            self.n_dims = obsmat.shape[-1]
            self.obsmat = obsmat
            resid_func = self.residual_withK_obsmat

        if t_eval is None:
            self.t = np.linspace(0, 1, X.shape[0])
        else:
            self.t = t_eval

        if est_vars is None:
            est_vars = 'rAKz0'
            # est_vars = 'rKz0'

        # Set time limit to fit
        def iter_cb(params, iter, resid, data, self, t0):
            if time.process_time() - t0 > max_time:
                return True  # terminate the LM algorithm

        min_loss = np.inf

        for _ in range(n_trial):
            # Train
            if try_catch:
                try:
                    result = lmfit.minimize(
                        resid_func,
                        self.init_params(r=r, A=A, K=K, z0=z0, maxA=maxA, est_vars=est_vars),
                        method='leastsq',  # Levenberg-Marquardt
                        args=(X, self, time.process_time()),
                        xtol=1e-7,
                        ftol=1e-7,
                        # maxfev=500,
                        nan_policy='raise',
                        iter_cb=iter_cb,
                        calc_covar=False) 
                except:
                    continue
            else:
                result = lmfit.minimize(
                    resid_func,
                    self.init_params(r=r, A=A, K=K, z0=z0, maxA=maxA, est_vars=est_vars),
                    method='leastsq',
                    args=(X, self, time.process_time()),
                    iter_cb=iter_cb,
                    calc_covar=False) 

            # Evaluate
            if self.use_carrying_capacity:
                r, A, K, z0 = self.params2numpy_withK(
                    result.params, self.r, self.A, self.K, self.z0)
            else:
                r, A, z0 = self.params2numpy(
                    result.params, self.r, self.A, self.z0)

            loss = np.sqrt(np.mean(result.residual ** 2))

            if loss < min_loss:
                min_loss = loss
                best_r  = deepcopy(r)
                best_A  = deepcopy(A)
                best_z0 = deepcopy(z0)
                if self.use_carrying_capacity:
                    best_K = deepcopy(K)

        try:
            self.r = best_r
            self.A = best_A
            self.z0 = best_z0
            if self.use_carrying_capacity:
                self.K = best_K
            return True # convergence

        except UnboundLocalError:
            return False


def glv(t, state, A, B, bound=10):
    # diff = A * state - np.dot(B, np.outer(state, state)).sum(axis=1)
    diff = A * state - np.diag(np.dot(B, np.outer(state, state)))
    # diff = A * state * (1 - B @ state)
    # diff = A * state - B @ (state ** 2)
    # diff = A * state - A * state * B @ state
    diff[diff > bound] = bound
    return diff



def generate(n_samples, A, B, Z, rk4=True):
    n_dims = Z.shape[0]
    t_eval = np.arange(n_samples)
    args   = (A, B)
    span   = [0, n_samples]

    #  Default values are 1e-3 for rtol and 1e-6 for atol.
    result = solve_ivp(glv, span, Z,
                       method='RK45',  # Non-stiff
                    #    method='Radau',  # Stiff
                       t_eval=t_eval,
                       args=args,
                       rtol=1e-2,
                       atol=1e-1,
                       )

    # result = RK45(glv, 0, Z, t_bound=n_samples, )

    if not result.status == 0:
        # print('[generate] RK status=', result.status)
        pred = np.zeros((n_samples, n_dims))
        pred[:len(result.y.T)] = result.y.T
        pred[len(result.y.T):] = result.y.T[-1]
        return pred

    else:
        return result.y.T  # pred: multi-dim sequence


def _generate(n_samples, A, B, Z, rk4=True, verbose=False):
    n_dims = Z.shape[0]
    t_eval = np.arange(n_samples)
    args   = (A, B)
    span   = [0, n_samples]

    #  Default values are 1e-3 for rtol and 1e-6 for atol.
    if verbose:
        tic = time.process_time()
        print('RK....')
    
    result = solve_ivp(glv, span, Z,
                       method='RK45',  # Non-stiff
                        # method='RK23',
                    #    method='Radau',  # Stiff
                       t_eval=t_eval,
                    #    vectorized=True,
                       args=args,
                       rtol=1e-2,
                       atol=1e-1,
                       dense_output=True,
                       )
    if verbose:
        toc = time.process_time() - tic
        print('\tRK:', toc, '[sec]')

    if not result.status == 0:
        # print('[_generate] RK status=', result.status)
        warnings.warn("RK45 did not converge")
        pred = np.zeros((n_samples, n_dims))
        pred[:len(result.y.T)] = result.y.T
        pred[len(result.y.T):] = result.y.T[-1]
        return pred, result.status

    else:
        return result.y.T, result.status  # pred: multi-dim sequence


def get_lmfit_parameters(n_dims, target='linear', init_params=None, bound=20):
    """ Define lmfit.Parameters object """

    params = lmfit.Parameters()

    if target == 'init_state':
        Z = np.zeros(n_dims) if init_params is None else init_params
        for i in range(n_dims):
            # params.add('Z{}'.format(i), value=0)
            # params.add('Z{}'.format(i), value=Z[i])
            params.add('Z{}'.format(i), value=Z[i], min=0)

    elif target == 'linear':
        if not init_params is None:
            Z = init_params[0]
            A = init_params[1]
        else:
            Z = np.zeros(n_dims)
            A = np.zeros(n_dims)

        for i in range(n_dims):
            # params.add('Z{}'.format(i), value=Z[i])
            params.add('Z{}'.format(i), value=Z[i], min=0)
            # params.add('Z{}'.format(i), value=Z[i], min=0, max=5)
            # params.add('Z{}'.format(i), value=Z[i], min=0, max=bound)
            params.add('A{}'.format(i), value=A[i], min=-10, max=10)
            # params.add('A{}'.format(i), value=A[i], min=-10, max=10)

    elif target == 'nl_diag':
        B = np.zeros(n_dims) if init_params is None else np.diag(init_params)
        # B = np.zeros(n_dims) if init_params is None else np.diag(init_params)
        for i in range(n_dims):
            # params.add('B{}'.format(i), value=B[i], min=-bound, max=bound)
            # params.add('B{}'.format(i), value=B[i])
            params.add('B{}'.format(i), value=B[i], min=-10, max=10)

    elif target == 'nl_nondiag':
        # init_params = B
        B = np.zeros(n_dims) if init_params is None else init_params
        for i, j in product(range(n_dims), repeat=2):
            vary = False if i == j else True # diagonal elements are fixed
            # vary = False if i == j or i > j else True # diagonal elements are fixed
            params.add('B{}{}'.format(i, j),
                value=B[i, j], vary=vary, min=-1, max=1)
            # params.add('B{}{}'.format(i, j), value=B[i, j], vary=vary)

    elif target == 'nl_full':
        # init_params = B
        B = np.zeros((n_dims, n_dims)) if init_params is None else init_params
        for i, j in product(range(n_dims), repeat=2):
            params.add('B{}{}'.format(i, j), value=B[i, j], vary=True, min=-bound, max=bound)

    elif target == 'full':
        # init_params = (Z, A, B)
        if init_params is None:
            Z = np.zeros(n_dims)
            A = np.zeros(n_dims)
            B = np.zeros((n_dims, n_dims))
        else:
            Z, A, B = init_params

        print(Z.shape, A.shape, B.shape)
        for i in range(n_dims):
            params.add('Z{}'.format(i), value=Z[i], min=0, max=bound)
            params.add('A{}'.format(i), value=A[i], min=-bound, max=bound)

        for i, j in product(range(n_dims), repeat=2):
            params.add('B{}{}'.format(i, j), value=B[i, j], vary=True, min=-bound, max=bound)

    else:
        raise ValueError("Invalid target has been specified")

    return params


# 関数を動的に定義して繰り替えした方が早い？
def read_lmfit_parameters(params, n_dims, target):

    if target == 'init_state':
        Z = np.array([params['Z{}'.format(i)] for i in range(n_dims)])
        return Z

    elif target == 'linear':
        Z = np.array([params['Z{}'.format(i)] for i in range(n_dims)])
        A = np.array([params['A{}'.format(i)] for i in range(n_dims)])
        return Z, A

    elif target == 'nl_diag':
        B = np.diag(np.array([params['B{}'.format(i)] for i in range(n_dims)]))
        return B

    elif target in ['nl_nondiag', 'nl_full']:
        B = np.array([params['B{}{}'.format(i, j)]
            for i, j in product(range(n_dims), repeat=2)
            ]).reshape((n_dims, n_dims)).T
        # diagB = np.diag(B)
        # B[B > 0] = 0
        # B[np.diag_indices(n_dims)] = diagB
        return B

    elif target == 'full':
        Z = np.array([params['Z{}'.format(i)] for i in range(n_dims)])
        A = np.array([params['A{}'.format(i)] for i in range(n_dims)])
        B = np.array([params['B{}{}'.format(i, j)]
            for i, j in product(range(n_dims), repeat=2)
            ]).reshape((n_dims, n_dims)).T
        return A, B, Z

    else:
        raise ValueError("Invalid fitting target has been specified")


def residual_linear(params, data, B, Z, integration_status):
    n, d = data.shape
    A = np.array([params['A{}'.format(i)] for i in range(d)])
    pred, integration_status = _generate(n, A, B, Z)
    resid = (data - pred).flatten()
    return resid


def residual_linear_with_init_state(params, data, fixed_params, integration_status):
    B = fixed_params
    n, d = data.shape
    Z = np.array([params['Z{}'.format(i)] for i in range(d)])
    A = np.array([params['A{}'.format(i)] for i in range(d)])
    pred, integration_status = _generate(n, A, B, Z)
    resid = (data - pred).flatten()
    return resid


def residual_nonlinear_diag(params, data, fixed_params, integration_status):
    Z, A, B = fixed_params
    n, d = data.shape
    # Update only diagonal elements
    B[np.diag_indices(d)] = np.array([params['B{}'.format(i)] for i in range(d)])
    pred, integration_status = _generate(n, A, B, Z)
    resid = (data - pred).flatten()
    return resid


def residual_nonlinear_full(params, data, fixed_params, integration_status):
    Z, A, B = fixed_params
    n, d = data.shape
    # Update all elements
    B = np.array([params['B{}{}'.format(i, j)] for i, j in product(range(d), repeat=2)])
    B = B.reshape((d, d)).T
    pred, integration_status = _generate(n, A, B, Z)
    resid = (data - pred).flatten()
    return resid


def residual_full(params, data, fixed_params, integration_status):
    # Read updated parameters
    n, d = data.shape

    Z = np.array([params['Z{}'.format(i)] for i in range(d)])
    A = np.array([params['A{}'.format(i)] for i in range(d )])
    B = np.array([params['B{}{}'.format(i, j)] for i, j in product(range(d), repeat=2)])
    B = B.reshape((d, d)).T

    # Compute predicted values
    pred, integration_status = _generate(n, A, B, Z)
    # print(np.isnan(pred).sum())
    # print(mean_squared_error(data, pred))
    resid = (data - pred).flatten()
    return resid


def residual_init_state(params, data, fixed_params, integration_status):
    A, B = fixed_params
    n, d = data.shape
    # Update initial state
    Z = np.array([params['Z{}'.format(i)] for i in range(d)])
    pred, integration_status = _generate(n, A, B, Z)
    resid = (data - pred).flatten()
    return resid


def iter_cb(params, iter, resid, data, fixed_params, tic):
    """ This function is called after evaluations lmfit.leastsq
        If this function returns True,
        then the optimization will be terminated.

        The arguments for this function consist of:
        - lmfit.Parameter object
        - iteration number
        - residual obtained by resid_function
        - unfoled fcn_args, e.g., training data and fixed parameters.
    """
    # print(time.process_time() - tic)
    if time.process_time() - tic > 1:
        return True
    # _, integration_status = _generate
    # print(iter, 'status=', integration_status)
    # if integration_status == -1:
    #     warnings.warn("RK45 did not converge")
    #     return True # kill the optimization

    """ check the maximum iteration"""
    # print('iter=', iter)
    # if iter > 1000:
    #     return True
    # print(np.mean(np.abs(resid)))

    """ check nan values """
    # if np.isnan(resid).sum() > 0:
    #     exit("[ERROR] residuals include nan")


def get_resid_func(target):
    if target == 'full':
        return residual_full
    elif target == 'nl_diag':
        return residual_nonlinear_diag
    elif target in ['nl_nondiag', 'nl_full']:
        return residual_nonlinear_full
    elif target == 'linear':
        return residual_linear_with_init_state
    elif target == 'init_state':
        return residual_init_state
    else:
        raise ValueError("Invalid target has been specified")


def set_method():
    return 'leastsq'  # common
    # return 'least_squares'  # not fast
    # return 'differential_evolution'  # too heavy
    # return 'bfgs'  # fantastic
    # return 'tcn'  # fine
    # return 'trust-krylov'  # how to get Jacobian?
    # return 'trust-constr'  # soso
    # return 'slsqp'
    # return 'lbfgsb'


def fit(data, target='full', init_params=None, fixed_params=None,
        xtol=5e-2, ftol=5e-2, maxfev=10000): # good

    _, n_dims = data.shape
    tic = time.process_time()
    fcn_args = (data, fixed_params, tic)
    params = get_lmfit_parameters(n_dims, target, init_params)
    residual = get_resid_func(target)

    solver = lmfit.Minimizer(residual, params, fcn_args=fcn_args)
    # solver = lmfit.Minimizer(
    #     residual, params, fcn_args=fcn_args, iter_cb=iter_cb)

    result = solver.leastsq(xtol=xtol, ftol=ftol)
    # result = solver.leastsq(xtol=xtol, ftol=ftol, maxfev=maxfev)
    return read_lmfit_parameters(result.params, n_dims, target)
    # for _ in range(5):
    #     try:
    #         result = solver.leastsq(xtol=xtol, ftol=ftol, maxfev=maxfev)
    #         return read_lmfit_parameters(result.params, n_dims, target)
    #     except lmfit.minimizer.AbortFitException:
    #         continue
    # else:
    #     return read_lmfit_parameters(params, n_dims, target)

    # result = solver.leastsq()  # default

    # print('nlds fit...')
    # tic = time.process_time()
    # result = solver.leastsq(xtol=xtol, ftol=ftol, maxfev=maxfev)
    # toc = time.process_time() - tic
    # print('elpased time [nlds]=', toc)

    # result = solver.leastsq(xtol=xtol, ftol=ftol, maxfev=maxfev)
    # method = set_method()
    # result = lmfit.minimize(residual, params, method=method, args=fcn_args)


""" Optimization functions for tensor analysis
"""


def get_tensor_resid_func(target):
    if target == 'full':
        return tensor_residual_full
    elif target == 'nl_full':
        return tensor_residual_nonlinear_full
    elif target == 'nl_nondiag':
        return tensor_residual_nonlinear_nondiag
    elif target == 'nl_diag':
        return tensor_residual_nonlinear_diag
    elif target == 'linear':
        return tensor_residual_linear
    elif target == 'init_state':
        return tensor_residual_init_state
    else:
        raise ValueError("Invalid target has been specified")


def tensor_residual_full(params, data, factors, fixed_params, d, rk4):
    n = data.shape[0]
    # Update parameters
    Z = np.array([params['Z{}'.format(i)] for i in range(d)])
    A = np.array([params['A{}'.format(i)] for i in range(d )])
    B = np.array([params['B{}{}'.format(i, j)] for i, j in product(range(d), repeat=2)])
    B = B.reshape((d, d)).T
    # Generate latent states
    lv_factor = generate(n, A, B, Z, rk4=rk4)
    # Predict original data
    factors[0] = lv_factor
    pred = kruskal_to_tensor((None, factors), mask=None)
    return (data - pred).flatten()


def tensor_residual_nonlinear_full(params, data, factors, fixed_params, d, rk4):
    n = data.shape[0]
    Z, A = fixed_params
    # Update parameters
    B = np.array([params['B{}{}'.format(i, j)] for i, j in product(range(d), repeat=2)])
    B = B.reshape((d, d)).T
    # Generate latent states
    lv_factor = generate(n, A, B, Z, rk4=rk4)
    # Predict original data
    factors[0] = lv_factor
    pred = kruskal_to_tensor((None, factors), mask=None)
    return (data - pred).flatten()


def tensor_residual_nonlinear_diag(params, data, factors, fixed_params, d, rk4):
    n = data.shape[0]
    Z, A, B = fixed_params
    # Update parameters
    B[np.diag_indices(d)] = np.array([params['B{}'.format(i)] for i in range(d)])
    # Generate latent states
    lv_factor = generate(n, A, B, Z, rk4=rk4)
    # Predict original data
    factors[0] = lv_factor
    pred = kruskal_to_tensor((None, factors), mask=None)
    return (data - pred).flatten()


def tensor_residual_nonlinear_nondiag(params, data, factors, fixed_params, d, rk4):
    n = data.shape[0]
    Z, A = fixed_params
    B = np.array([params['B{}{}'.format(i, j)] for i, j in product(range(d), repeat=2)])
    B = B.reshape((d, d)).T
    # Generate latent states
    lv_factor = generate(n, A, B, Z, rk4=rk4)
    # Predict original data
    factors[0] = lv_factor
    pred = kruskal_to_tensor((None, factors), mask=None)
    return (data - pred).flatten()


def tensor_residual_linear(params, data, factors, fixed_params, d, rk4):
    n = data.shape[0]
    B = fixed_params
    Z = np.array([params['Z{}'.format(i)] for i in range(d)])
    A = np.array([params['A{}'.format(i)] for i in range(d)])
    # Generate latent states
    lv_factor = generate(n, A, B, Z, rk4=rk4)
    # Predict original data
    factors[0] = lv_factor
    pred = kruskal_to_tensor((None, factors), mask=None)
    return (data - pred).flatten()


def tensor_residual_init_state(params, data, factors, fixed_params, d, rk4):
    n = data.shape[0]
    A, B = fixed_params
    Z = np.array([params['Z{}'.format(i)] for i in range(d)])
    factors[0] = generate(n, A, B, Z, rk4=rk4)
    pred = kruskal_to_tensor((None, factors), mask=None)
    return (data - pred).flatten()


def fit_tensor(data, factors, target='full',
               init_params=None, fixed_params=None, sigma=10,
               xtol=1e-3, ftol=1e-3, maxfev=10000, rk4=True):
    """ Non-linear optimization for tensor series """

    _, n_dims = factors[-1].shape  # n_components

    # _data = tl.base.fold(gaussian_filter1d(
    #     tl.base.unfold(data, 0), sigma), 0, data.shape)
    _data = data
    fcn_args = (_data, factors, fixed_params, n_dims, rk4)

    params = get_lmfit_parameters(n_dims, target, init_params)
    residual = get_tensor_resid_func(target)
    # method = set_method()

    solver = lmfit.Minimizer(residual, params, fcn_args=fcn_args)
    # solver = lmfit.Minimizer(residual, params, fcn_args=fcn_args, iter_cb=iter_cb)
    # result = solver.leastsq()  # default
    # result = solver.leastsq(xtol=xtol, ftol=ftol)
    result = solver.leastsq(xtol=xtol, ftol=ftol, maxfev=maxfev)
    # result = lmfit.minimize(residual, params, method=method, args=fcn_args)
    # result = lmfit.minimize(residual, params, method=method, args=fcn_args, iter_cb=iter_cb)

    return read_lmfit_parameters(result.params, n_dims, target)

def fit_dot_tensor(data, factors):

    # data: tensor
    length = len(data)
    n_dims = factors[-1].shape[-1]
    params = get_lmfit_parameters(n_dims, target='full')
    fcn_args = (data, factors, length, n_dims)

    def resid(params, data, factors, n, d):
        # Update parameters
        Z = np.array([params['Z{}'.format(i)] for i in range(d)])
        A = np.array([params['A{}'.format(i)] for i in range(d )])
        B = np.array([params['B{}{}'.format(i, j)] for i, j in product(range(d), repeat=2)])
        B = B.reshape((d, d)).T

        # Generate latent states
        lv_factor = generate(n, A, B, Z, True)

        tmp = lv_factor * factors[0]  # trends * seasonality
        updated = [tmp] + factors[1:]
        pred = kruskal_to_tensor((None, updated), mask=None)

        return (data - pred).flatten()

    solver = lmfit.Minimizer(resid, params, fcn_args=fcn_args)
    result = solver.leastsq(xtol=1e-3, ftol=1e-3)

    return read_lmfit_parameters(result.params, n_dims, 'full')
