import numpy as np
from dateutil import parser
import pandas as pd
from datetime import datetime


def equal_interval_creation(
        particles,
        fields,
        time_intervals,
        time_horizon,
        time_passed):
    """Instrumental function for inference evaluation and pllotting in the artificial experiment.
    Create subsets of particles with growing length,
    assigning a time-passed variable based splitting this time on the overall samples set.
    Return a list of different samples subsets."""
    ret = []
    time_ratio = time_horizon / time_passed
    for i in range(1, time_intervals + 1):
        w = {}
        for key in particles.keys():
            if (key in fields) | (fields is None):
                x = particles[key]
                # N0 = int(x.shape[0] * time_ratio * ((i - 1) / time_intervals))
                N1 = int(x.shape[0] * time_ratio * (i / time_intervals))
                N0 = int(N1 * 0.1)
                w[key] = x[N0:N1]
        t_i = time_passed * (N1 / x.shape[0])
        w = {'particles': w}
        w['time'] = t_i
        w['epoch'] = i
        ret.append(w)
    return ret


def asymmetric_mix_dict(d1, d2):
    """Merge the keys of d2 into d1. Common keys are overrided."""
    ret = d1
    for k in d2:
        ret[k] = d2[k]
    return ret


def concat_dict(d1, d2):
    """Apply np.concatenate to every key in common between two dictionaries."""
    if d1 is None:
        return d2
    if d2 is None:
        return d1
    else:
        assert set(d1.keys()) == set(d2.keys())
        return {k: np.concatenate([d1[k], d2[k]], axis=0) for k in d1}


def dict_tf2numpy(self):
    '''
    Transform a dictionary of list of tensor into a dict or list of numpy objects.
    The single objects transformations happens through .numpy method.
    :param self: Any dict or list of objects with a .numpy method
    :type self: dict or list
    :return: dict or list of objects of class numpy
    :rtype: dict or list
    '''
    ret = {}
    for k in self:
        x = self[k]
        if type(x) == dict:
            ret[k] = {v: dict_tf2numpy(x[v]) for v in x}
        elif type(x) == list:
            ret[k] = [dict_tf2numpy(v) for v in x]
        else:
            ret[k] = x.numpy()
    return ret


def inv_softplus(x, _limit_upper=30, _limit_lower=1e-12):
    '''
    Returns y (float32), s.t. softplus(y)=x    
    '''
    if isinstance(x, np.float) or isinstance(x, np.int):
        if x < _limit_upper:
            ret = np.log(np.exp(x) - 1)
        else:
            ret = x
    else:
        ret = np.zeros(x.shape, dtype=np.float32)
        under_limit = x < _limit_upper
        over_limit = np.logical_not(under_limit)
        ret[under_limit] = np.float32(np.log(np.exp(x[under_limit]) - 1 + _limit_lower))
        ret[over_limit] = x[over_limit]
    return ret


def safe_softplus(x, limit=10):
    """Softplus function correction to avoid numeric overflow."""
    ret = x
    _under_limit = x < limit
    ret[_under_limit] = np.log(1.0 + np.exp(x[_under_limit]))
    return ret


def lagify(y, p):
    '''
    Taken time series y (vertical), returns columns with the last p lags of y.
    Returns both y and ylag, aligned so that ylag sees just until yesterday.
    '''
    T, N = y.shape
    ylag = np.ones([T, N * p + 1])
    for pp in range(p):
        ylag[pp + 1:T, N * pp + 1:pp * N + N + 1] = y[:T - pp - 1, :]
    return np.float32(y[p:, :]), np.float32(ylag[p:, :])


def VAR_data_generation(T, N, par_p, cov_wn, const_terms):
    '''
    generates T x N data, with par_p VAR structure, cov_wn noise covariance and a vector of constant terms cont_terms.
    
    '''
    p = int(par_p.shape[0] / N)
    eps = np.random.multivariate_normal(np.zeros(N), cov_wn, size=T)
    y = np.zeros([T, N])
    last_y = np.zeros([p, N])
    ylag = np.zeros([T, N * p + 1])
    for t in range(T):
        ylag[t] = np.concatenate([np.ones([1, 1]), last_y.reshape(1, -1)], axis=1)
        y[t, :] = const_terms + np.matmul(last_y.reshape(1, -1), par_p) + eps[t]
        last_y[:p - 1] = last_y[1:]
        last_y[p - 1] = y[t]
    return y, ylag


def spiral_indexes(N):
    '''
    return the indexes of a line vector that corresponds to the elements of a triangular matrix.
    spiral means that the elements in the matrix are inserted using a spiral sequence (as tensorflow.fill_triangular does).
    '''
    spiral_matrix = np.zeros([N, N], dtype=np.int)
    spiral_line_tril = np.zeros(int(N * (N + 1) / 2), dtype=np.int)
    last_num = 0
    ln = 0
    for n in range(N):
        if (n % 2) == 0:
            # assigns the inverted rows
            val_n = N - int(n / 2)
            spiral_matrix[N - 1 - int(n / 2), :N - int(n / 2)] = np.flip(last_num + np.arange(val_n))

            # print(ln,ln+N-int(n/2))
            qn = N ** 2 - int(n / 2) * N
            inds = (np.arange(qn - N, qn - int(n / 2)))
            spiral_line_tril[ln:ln + N - int(n / 2)] = np.flip(inds)
            last_num += val_n
            ln += N - (int(n / 2))
        else:
            # assign the rows
            val_n = int((n + 1) / 2)
            spiral_matrix[int((n - 1) / 2), :int((n + 1) / 2)] = last_num + np.arange(val_n)
            last_num += val_n

            qn = (val_n - 1) * N  # int(val_n*(val_n-1)/2)
            inds = np.arange(qn, qn + val_n)
            spiral_line_tril[ln:ln + val_n] = inds
            ln += val_n
    return spiral_matrix[np.diag_indices(N)], spiral_matrix[np.tril_indices(N, -1)], spiral_matrix[
        np.tril_indices(N)], spiral_matrix, spiral_line_tril


def fromMat2diag_udiag(mat):
    '''
    Given a matrix returns the diagonal and the strictly lower triangular part of Cholesky(mat).
    The strict lower matrix returned is normalized per the diagonal elements corresponding.
    '''
    N = mat.shape[0]
    cholmat = np.linalg.cholesky(mat)
    choldiag = np.diag(cholmat)
    normmat = np.tile(np.reshape(choldiag, [1, N]), [N, 1])
    choludiag = (cholmat / normmat)[np.tril_indices(N, -1)]
    return choldiag, choludiag


def arctanh(x):
    '''
    returns arctanh(x), doesn't check for nans.
    '''
    ret = 0.5 * np.log((1 + x) / (1 - x))
    if (np.sum(np.isnan(ret)) > 0):
        print(x)
        ret[np.isnan(ret)] = 0.0
    return ret


class indexes_librarian:
    '''
    A single class that collects different set of indexes, useful to gather ndarrays.
    '''

    def __init__(self, N):
        self.spiral_diag, self.spiral_udiag, self.spiral_tril, self.spiral_matrix, self.spiral_line = spiral_indexes(N)
        self.diag = np.diag_indices(N)
        self.udiag = np.tril_indices(N, -1)
        self.tril = np.tril_indices(N)


def from_daily2_monthly(y, log_returns=False):
    '''
    Transform the pandas Dataframe with time-index to a montly series. log_returns parameter controls if log-returns must be computed.
    '''
    ind_dates = np.zeros(y.shape[0], dtype=np.int)
    last_date = None
    jj = 0
    for ii in range(y.index.shape[0]):
        date_ii = parser.parse(y.index[ii])
        if ii == 0 or not date_ii.month == last_date.month:
            ind_dates[jj] = ii
            jj += 1
        last_date = date_ii
    ind_dates = ind_dates[:jj]
    ret = y.iloc[ind_dates, :].values
    if log_returns:
        ret = np.log(ret[1:, :]) - np.log(ret[:-1, :])
    ret = pd.DataFrame(ret, y.index[ind_dates[1:]])
    return ret


def init_BetaPdfLowVariance_fromPoint(x, b=10.0, _min_a=1e-1):
    '''
    Given x, a ndarray of observed vaues from different Beta distributions, returns a pair of parameters a,b that corresponds to Beta distributions with expected value equal to x and variance controlled by b (bigger the b, lower the variance).
    '''
    xb = np.ones(x.shape, dtype=np.float32) * b
    xa = xb * (x) / (1.0 - x)
    if isinstance(x, np.float):
        if xa / xb < _min_a:
            xa = _min_a * xb
    else:
        under_min = xa / xb < _min_a
        xa[under_min] = _min_a * xb[under_min]
    return xa, xb


def init_GammaPdfLowVariance_fromPoint(x, b=10.0):
    '''
    Given x, a ndarray of observed vaues from different Gamma distributions, returns a pair of parameters a,b that corresponds to Gamma distributions with expected value equal to x and variance controlled by b (bigger the b, lower the variance).
    '''
    xb = np.ones(x.shape, dtype=np.float32) * b
    xa = xb * x
    return xa, xb


def view_stats(x, axis=None):
    if axis is None:
        print(f'min: {np.min(x)}\nmean: {np.mean(x)}\nmax: {np.max(x)}\nstd: {np.std(x)}')
    else:
        print(
            f'min: {np.min(x, axis=axis)}\nmean: {np.mean(x, axis=axis)}\nmax: {np.max(x, axis=axis)}\nstd: {np.std(x, axis=axis)}')


def matrix2line_diagFunc(M, inv_func=inv_softplus):
    '''
    Takes a matrix and extract a line with the coefficient of the Cholesky decomposition.
    The order is spyral, so this function is the numpy inverse of tensorflow.fill_triangular.
    :param M: The matrix
    :type M: np.array
    :param inv_func: The function to apply to the coefficients on the diagonal
    :type inv_func: tf.function with domain positive number and codomain the real line
    :return: A line with the coefficients.
    :rtype: np.array
    '''
    N = M.shape[0]
    assert N == M.shape[1]
    ind = indexes_librarian(N)
    _cholvcv = np.linalg.cholesky(M)
    diag_cholvcv = _cholvcv[ind.diag]
    _cholvcv[ind.diag] = inv_func(diag_cholvcv)
    ret = _cholvcv[ind.tril]
    ret[ind.spiral_diag] = _cholvcv[ind.diag]
    ret[ind.spiral_udiag] = _cholvcv[ind.udiag]
    return ret


def matrix2line_diagFunc_timeseries(M, inv_func=inv_softplus):
    '''
    Takes a matrix and extract a line with the coefficient of the Cholesky decomposition.
    The order is spyral, so this function is the numpy inverse of tensorflow.fill_triangular.
    :param M: The matrix
    :type M: np.array
    :param inv_func: The function to apply to the coefficients on the diagonal
    :type inv_func: tf.function with domain positive number and codomain the real line
    :return: A line with the coefficients.
    :rtype: np.array
    '''
    t, n, _ = M.shape
    assert n == _
    ind = indexes_librarian(n)
    _cholvcv = np.linalg.cholesky(M)
    diag_cholvcv = _cholvcv[:, ind.diag[0], ind.diag[1]]
    _cholvcv[:, ind.diag[0], ind.diag[1]] = inv_func(diag_cholvcv)
    ret = _cholvcv[:, ind.tril[0], ind.tril[1]]
    ret[:, ind.spiral_diag] = _cholvcv[:, ind.diag[0], ind.diag[1]]
    ret[:, ind.spiral_udiag] = _cholvcv[:, ind.udiag[0], ind.udiag[1]]
    return ret


def decorator_multilevel(f):
    """Decorator the apply hierarchically the decorated function to every element in:
    -dictionaries
    -list
    -single elements"""
    def f_decorated(x):
        if type(x) == dict:
            fx = {k: f_decorated(x[k]) for k in x}
        elif (type(x) == list) | (type(x) == tuple):
            fx = [f_decorated(k) for k in x]
        else:
            fx = f(x)
        return fx

    return f_decorated


def format_number(x):
    return np.round(x, 2)


def runtime_print(f):
    """Decorate a function to print its runtime"""
    def decorated_fun(*args, **kwargs):
        t0 = datetime.now()
        ret = f(*args, **kwargs)
        t1 = datetime.now()
        print(f'Runtime: {t1 - t0}')
        return ret

    return decorated_fun


def print_formatted_values(**kwargs):
    """Print all the values with ',' as a separator."""
    string = ', '.join([f'{k}: {format_number(kwargs[k])}' for k in kwargs])
    print(string)


if __name__ == '__main__':
    M = np.repeat(np.eye(3)[np.newaxis, :, :], 5, axis=0)
    print(M)
    M_line = matrix2line_diagFunc_timeseries(M)
    print(M_line)
