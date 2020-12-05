import pandas as pd
import numpy as np
from dccR import rmgarch_inference
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from utils import matrix2line_diagFunc_timeseries


def train_test_decorator_naive(f):
    """Return a decorated function that apply the input function to both train and test variables."""
    def f_decorated(train, test):
        return f(train), f(test)

    return f_decorated


def dcc(y_train, y_test):
    """
    Estimate DCC on train,test data (fitting only on train)
    :param y_train: The train data dim=t1,n
    :param y_test: The test data dim=t2,n
    :return: A dictionary with the results
    """
    y = np.concatenate([y_train, y_test], axis=0)
    i = y_test.shape[0]
    DCC = rmgarch_inference(df=y, out_sample=i, file_script=r'./dccR/rmgarch_code_4py_col.R')
    vcv_test = np.float32(DCC['Vcv_forecast'][-i - 1:-1])
    vcv_train = np.float32(DCC['Vcv'])
    mu_train = np.float32(DCC['Ey'])
    mu_test = np.float32(DCC['Ey_forecast'][-i - 1:-1])
    return {'vcv': vcv_train, 'mu': mu_train}, {'vcv': vcv_test, 'mu': mu_test}


def staticLedoitWolf(x_train, x_test):
    """
    Estimate a static mean vector-covariance matrix on train,test data (fitting only on train)
    :param y_train: The train data dim=t1,n
    :param y_test: The test data dim=t2,n
    :return: A dictionary with the results
    """
    [dt, N] = x_test.shape
    [T, _] = x_train.shape
    vcv = np.repeat(LedoitWolf().fit(x_train).covariance_[np.newaxis], T + dt, axis=0)
    mu = np.repeat(x_train.mean(axis=0)[np.newaxis], T + dt, axis=0)
    return {'vcv': vcv[:-dt], 'mu': mu[:-dt]}, {'mu': mu[-dt:], 'vcv': vcv[-dt:]}


def standardize(x_train, x_test):
    """Standardize by subtracting the mean and dividing
    by standard deviation all the series, separately."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    return x_train, scaler.transform(x_test)


def _elements_cross_product(x, axis=-1):
    """
    Compute the cross-product of the elements on a given axis.
    :param x: An input multi-dim array, dim=.,axis=n,.
    :param axis: Which axis to do the cross-product
    :return: An output multi-dim array, dim=.,axis=n,axis+1=n,.
    """
    if type(x) == pd.DataFrame:
        x = x.values
    n = x.shape[axis]
    r = np.stack([x.take(i, axis=axis) * x.take(j, axis=axis) for i in range(n) for j in range(i, n)], axis=axis)
    return r


def elements_cross(x_train, x_test):
    """
    Compute the cross-product of the elements, on both train and test.
    """
    c_train = _elements_cross_product(x_train, axis=1)
    c_test = _elements_cross_product(x_test, axis=1)
    return c_train, c_test


def _lagger(x, lag=10, collapse=False):
    """Return the last lags of x as new enlarged set of features"""
    x_df = pd.DataFrame(x)
    lagged = [x_df.shift(i).values for i in range(1, lag + 1)]
    if collapse:
        lagged = np.concatenate(lagged, axis=1)
    else:
        # Keep an extra dimension with shape = lag
        lagged = np.stack(lagged, axis=1)
    return lagged


def lagify(x_train, x_test, lag=10, collapse=False):
    """Apply the _lagger transformation concatenating train and test,
     then dividing them again"""
    x_all = np.concatenate([x_train, x_test], axis=0)
    x_all = _lagger(x_all, lag=lag, collapse=collapse)
    x_train, x_test = x_all[:x_train.shape[0]], x_all[x_train.shape[0]:]
    return x_train, x_test


def _see_col(d, name=''):
    """Explore a dictionary or a single array, counting the nans"""
    print(name)
    if type(d) == dict:
        [print(k, d[k].shape, np.sum(np.isnan(d[k]))) for k in d]
    else:
        print(d.shape, np.sum(np.sum(np.isnan(d))))


def log_returns(x, fill_t0=0.0):
    """Compute the log-returns of a series"""
    y0 = np.zeros(x[:1].shape)
    y0[:, :] = fill_t0
    y = np.log(x[1:] / x[:-1])
    y = np.stack([y0, y], axis=0)
    return y


def prices_from_returns(x, fill_t0=1.0):
    """Compute the series of prices, with initial value fill_t0, using the log-returns."""
    y = np.exp(np.cumsum(x)) * fill_t0
    return y


if __name__ == '__main__':

    from econometrics_problem import DatasetFactory

    data_config = {
        'name': 'FRED'
    }
    train_test_config = {
        'sequential': True,
        'test_size': 0.3
    }
    dsf = DatasetFactory.get(**data_config)
    train_time = dsf.split_train_test(**train_test_config)

    # Test all functions
    dsf.withColumn('y', *standardize(*dsf['y']))
    # dsf.withColumn('dcc', *dcc(*dsf['y']))
    dsf.withColumn('crosses', *elements_cross(*dsf['y']))
    dsf.withColumn('x_timesteps', *lagify(*dsf['y'], collapse=False))
    dsf.withColumn('x', *lagify(*dsf['y'], collapse=True))
    dsf.withColumn('ledoitWolf_static', *staticLedoitWolf(*dsf['y']))
    dsf.dropna()

    # Print results
    for k in dsf.train:
        x_train, x_test = dsf[k]
        _see_col(x_train, k + ' train')
        _see_col(x_test, k + ' test')

    batch = dsf.get_batch(dsf.train.keys(), 5)
    for k in batch:
        x_train = batch[k]
        _see_col(x_train, k + ' batch')
