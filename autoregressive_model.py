import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf
from utils import inv_softplus
from gaussian_multivariate_model import log_lkl_timeseries
import pandas as pd
from dccR import rmgarch_inference
import sys

_min_number = 1e-10


def extractionParameters_DCCrmgarch(y_train, y_valid, ar, ma, p, q, o, distribution='mvnorm',
                                    vol='gjrGARCH'):
    """
    Takes data and estimate central DCC parameters value.
    :param y_train: Training observations
    :param y_valid: Test observations (only evaluate DCC on them)
    :param ar: Autoregressive order for the mean
    :param ma: Moving average order for the mean
    :param p:  Autoregressive order for GARCH
    :param q:  Moving average order for GARCH (square residuals)
    :param o: Asymmetric order in GARCH
    :param distribution: The residuals distributions
    :param vol: the kind of GARCH to use (eGARCH, sGARCH, gjrGARCH)
    :return: A dictionary with the estimated parameters
    """
    mean = 'Constant'
    ret = {}
    [t, n] = y_train.shape
    [t_valid, _] = y_valid.shape
    assert _ == n
    if vol == 'eGARCH':
        o = p
    if (p != o) and (vol == 'sGARCH'):
        raise Exception('p and o must be equal when using rmgarch initialization')
    print(f' HVI DCCs O is: {o}' )
    y_all = np.concatenate([y_train, y_valid], axis=0)
    DCC = rmgarch_inference(df=pd.DataFrame(y_all), model_GARCH=vol, ar_order=ar, ma_order=ma, pdf_DCC=distribution,
                            p_order=p, q_order=q,
                            file_script=r'./dccR/rmgarch_code_4py_col.R', out_sample=t_valid)
    vcv_valid = DCC['Vcv_forecast'][-t_valid - 1:-1]
    vcv_train = DCC['Vcv']
    # ret['stdresid'] = DCC['stdresid']
    order = np.max([ar, ma, p, o, q])
    ret['Ey0'] = DCC['Ey'][0]
    ret['Ey'] = DCC['Ey']
    diag_ind = np.diag_indices(n)
    sigmas = DCC['Vcv'][:, diag_ind[0], diag_ind[1]]
    ret['vcv'] = DCC['Vcv']
    ret['sigma'] = sigmas
    ret['sigma0'] = sigmas[0]
    ret['corr'] = DCC['Corr']
    ret['corr0'] = DCC['Corr'][0]

    def coefRmgarch2dict(DCC):
        def get_coef_ind(i, i_final=None):
            if i_final is None:
                ret = DCC['Coef'][:, i]
            else:
                ret = DCC['Coef'][:, i:i_final]
                ret = ret.transpose(1, 0)
            return ret

        ret = {}
        if ar > 0:
            ret['ar'] = np.zeros([ar, n], dtype=np.float32)
        if ma > 0:
            ret['ma'] = np.zeros([ma, n], dtype=np.float32)
        ret['mu'] = np.zeros([1, n], dtype=np.float32)
        ret['mu'][0, :] = get_coef_ind(0)
        ret['omega'] = np.zeros([1, n], dtype=np.float32)
        ret['omega'][0, :] = get_coef_ind(1)
        if p > 0:
            ret['alpha'] = np.zeros([p, n], dtype=np.float32)
            ret['alpha'][:, :] = get_coef_ind(2, 2 + p)
        if q > 0:
            ret['beta'] = np.zeros([q, n], dtype=np.float32)
            ret['beta'][:, :] = get_coef_ind(2 + p, 2 + p + q)
        if o > 0:
            ret['gamma'] = np.zeros([o, n], dtype=np.float32)
            ret['gamma'][:, :] = get_coef_ind(2 + p + q, 2 + p + q + o)
        ret['a'] = DCC['a']
        ret['b'] = DCC['b']
        return ret

    ret = {**ret, **coefRmgarch2dict(DCC)}
    return ret


def extracted_dict_pars_2_values(mc, o, order, p, q, ret):
    """Compose a dictionary of parameters setting the vector dimensions for usage in DCC class"""
    sigma_t0 = np.float32(ret['sigma0'][np.newaxis, np.newaxis])
    sigma_t0 = np.repeat(sigma_t0, mc, axis=0)
    sigma_t0 = np.repeat(sigma_t0, order, axis=1)
    mu0 = ret['mu'][0][np.newaxis]
    omega0 = ret['omega'][0][np.newaxis]
    if p > 0:
        alpha0 = ret['alpha'][np.newaxis]
    else:
        alpha0 = None
    if q > 0:
        beta0 = ret['beta'][np.newaxis]
    else:
        beta0 = None
    if o > 0:
        gamma0 = ret['gamma'][np.newaxis]
    else:
        gamma0 = None
    a0 = np.reshape(ret['a'], [1, 1])
    b0 = np.reshape(ret['b'], [1, 1])
    corr0 = np.repeat(np.float32(ret['corr0'][np.newaxis]), mc, axis=0)
    return mu0, omega0, alpha0, beta0, gamma0, sigma_t0, corr0, a0, b0


def shape_garch(particles, mu0, omega0, alpha0, gamma0, beta0):
    """
    Shape a vector of particles into GARCH parameters, centered in the
    center values: mu0,omega0,alpha0,gamma0,beta0.
    :param particles: A vector
    :param mu0: The value of mu for the image of the 0-vector (center of the parametrization)
    :param omega0: The value of omega for the image of the 0-vector (center of the parametrization)
    :param alpha0: The value of alpha for the image of the 0-vector (center of the parametrization)
    :param gamma0: The value of gamma for the image of the 0-vector (center of the parametrization)
    :param beta0: The value of beta for the image of the 0-vector (center of the parametrization)
    :return: Samples of all the parameters and the Jacobian of the shaping transformation itself.
    """
    last_ind = 0
    mc, n = mu0.shape
    particles = tf.reshape(particles, [particles.shape[0], -1, n])
    # mu
    norm_mus = 0.01
    samples_mu = particles[:, last_ind, :] * norm_mus + mu0
    jacobian = np.log(norm_mus)  # Jacobian mus
    # omega
    norm_omegas = 0.1
    _omegas = particles[:, last_ind, :] + inv_softplus(1.0 / norm_omegas)
    samples_omega = tf.nn.softplus(_omegas) * norm_omegas * omega0 + _min_number
    jacobian += tf.reduce_mean(tf.nn.sigmoid(_omegas), axis=1) + np.log(norm_omegas)  # Jacobian omegas
    # alpha
    if alpha0 is None:
        p = 0
        samples_alpha = None
    else:
        p = alpha0.shape[1]
        norm_alphas = 0.01
        _alphas = particles[:, last_ind:last_ind + p, :] + inv_softplus(1.0 / norm_alphas)
        samples_alpha = tf.nn.softplus(_alphas) * norm_alphas * alpha0 + _min_number
        jacobian += tf.reduce_mean(tf.nn.sigmoid(_alphas), axis=(1, 2)) + np.log(norm_alphas)  # Jacobian alphas
    # beta
    if beta0 is None:
        q = 0
        samples_beta = None
    else:
        q = beta0.shape[1]
        norm_betas = 0.1
        _betas = particles[:, last_ind:last_ind + q, :] + inv_softplus(1.0 / norm_betas)
        samples_beta = tf.nn.softplus(_betas) * norm_betas * beta0 + _min_number
        jacobian += tf.reduce_mean(tf.nn.sigmoid(_betas), axis=(1, 2)) + np.log(norm_betas)  # Jacobian betas
    # gamma
    if gamma0 is None:
        o = 0
        samples_gamma = None
    else:
        o = gamma0.shape[1]
        norm_gammas = 0.1
        _gammas = particles[:, last_ind:last_ind + o, :] + inv_softplus(1.0 / norm_gammas)
        samples_gamma = tf.nn.softplus(_gammas) * norm_gammas * gamma0 + _min_number
        jacobian += tf.reduce_mean(tf.nn.sigmoid(_gammas), axis=(1, 2)) + np.log(norm_gammas)  # Jacobian gammas

    return samples_mu, samples_omega, samples_alpha, samples_gamma, samples_beta, jacobian


def power_tf_func(x, power):
    if power == 2.0:
        return tf.math.square(x)
    elif power == 1.0:
        return tf.math.abs(x)
    else:
        return tf.math.exp(tf.math.log(tf.math.abs(x)) * power)


def reconstructionParameters(particles, a0, b0):
    """
    Shape a vector of particles into DCC parameters (the covariance update part), centered in the
    values: a0,b0.
    :param particles: A tensor of particles
    :param a0: The value of a for the image of the 0-vector (center of the parametrization)
    :param b0: The value of b for the image of the 0-vector (center of the parametrization)
    :return: Samples of the a,b parameters and the Jacobian of the shaping transformation itself
    """
    samples = {}
    min_number = 1e-10  # imposed to avoid log(0) in the Jacobian transformation
    jacobian = 0.0
    norm_corrA = np.min([np.abs(a0), np.abs(1 - a0)]) * 0.1
    _corrA = particles[:, 0:1] * norm_corrA + np.arctanh(a0 * 2 - 1)
    samples_corrA = tf.nn.tanh(_corrA) / 2 + 0.5 + min_number
    samples['corrA'] = samples_corrA
    jacobian += (
            tf.reduce_mean(tf.math.log(1 - tf.math.square(samples_corrA * 2 - 1)), axis=1)
            + np.log(norm_corrA / 2)
    )
    norm_corrB = np.min([np.abs(b0), np.abs(1 - b0)]) * 0.1
    _corrB = particles[:, 1:2] * norm_corrB + np.arctanh(b0 * 2 - 1)
    samples_corrB = tf.nn.tanh(_corrB) / 2 + 0.5 + min_number
    samples['corrB'] = samples_corrB
    jacobian += (
            tf.reduce_mean(tf.math.log(1 - tf.math.square(samples_corrB * 2 - 1)), axis=1)
            + np.log(norm_corrB / 2)
    )
    return samples, jacobian


def tf_VectorsCrossProduct(x):
    """
    Computes the cross products of elements in x, returning a matrix C_k,i,j = x_k,i*x_k,j
    :param x: tensor of dimension k,d
    :return: A tensor of dimension k,d,d
    """
    rx = tf.expand_dims(x, -1)
    reorder = np.arange(len(x.shape) + 1)
    reorder[-1], reorder[-2] = reorder[-2], reorder[-1]
    return tf.matmul(rx, tf.transpose(rx, reorder))


def filtering_dcc(standardized_residuals, corrA, corrB, corr0, order):
    """
    Takes standardized residuals and DCC correlation parameters to estimate the correlation matrix for every time.
    :param standardized_residuals: A vector of residuals with dim t,d
    :param corrA: A batch of a parameters, with dim t,1
    :param corrB: A batch of b parameters, with dim t,1
    :param corr0: The initial correlation matrix at t=0
    :param order: The order of the autoregressive (=1 for standard DCC)
    :return: A tensor with the correlation matrix for every time, with dim t,d,d
    """
    timesteps = standardized_residuals.shape[1] - order
    corrA = corrA[:, :, tf.newaxis]
    corrB = corrB[:, :, tf.newaxis]
    conditional_correlations = []
    for i in range(order):
        conditional_correlations.append(corr0)
    epsilon_cross_batch = tf_VectorsCrossProduct(standardized_residuals)
    # epsilon-->q
    for t in range(order, order + timesteps):
        new_correlations = (
                corr0 * (1 - corrA - corrB) +
                epsilon_cross_batch[:, t] * corrA + conditional_correlations[t - 1] * corrB
        )
        conditional_correlations.append(new_correlations)
        correlations = tf.stack(conditional_correlations, axis=1)
        # q-->rho\n",
        diag_correlations = tf.linalg.diag_part(correlations)
        diag_normalization = tf.sqrt(tf_VectorsCrossProduct(diag_correlations))
        correlations_normalized = correlations / diag_normalization
    return correlations_normalized


def prior_dcc(power, mu, omega, alpha=None, gamma=None, beta=None, sigma_t0=None, corr0=None, a=None, b=None):
    """Flat prior for DCC models."""
    return tf.constant(0.0, dtype=np.float32)


def filtering_garch(y, power, mu, omega, alpha, gamma, beta, sigma_t0):
    """
    Takes residuals and compute GARCH volatilities for the single components, for every time.
    :param y: A vector of residuals with dim t,d
    :param power: The power of the GARCH
    :param mu: A batch of a parameters, with dim t,1
    :param omega: A batch of a parameters, with dim t,1
    :param alpha: A batch of a parameters, with dim t,1
    :param gamma: A batch of a parameters, with dim t,1
    :param beta: A batch of a parameters, with dim t,1
    :param sigma_t0: The initial volatility for the components at t=0 (with dim 1,d)
    :return: A tensor with the volatilties for every time, with dim t,d
    """
    if alpha is None:
        p = 0
    else:
        p = alpha.shape[1]
    if beta is None:
        q = 0
    else:
        q = beta.shape[1]
    if gamma is None:
        o = 0
    else:
        o = gamma.shape[1]

    order = np.max([p, o, q])
    timesteps = y.shape[1] - order
    conditional_sigmas = []

    def power_func(x):
        return power_tf_func(x, power)

    # adds the starting points
    for l in range(order):
        conditional_sigmas.append(power_func(tf.sqrt(sigma_t0[:, order - 1 - l])))
    # update with garch
    for t in range(order, timesteps + order):
        new_sigmas = omega
        for l in range(q):
            new_sigmas += conditional_sigmas[t - 1 - l] * beta[:, l]
        for l in range(o):
            new_sigmas += power_func(tf.nn.relu(-y[:, t - l] + mu)) * gamma[:, l]
        for l in range(p):
            new_sigmas += power_func(y[:, t - l] - mu) * alpha[:, l]
        conditional_sigmas.append(new_sigmas)
    sigmas_k = tf.stack(conditional_sigmas, axis=1)
    sigmas = tf.math.exp(tf.math.log(tf.nn.relu(sigmas_k) + _min_number) * 2 / power)
    vcv = tf.linalg.diag(sigmas)
    standardized_residuals = (y - mu[:, tf.newaxis, :]) / tf.math.sqrt(sigmas)
    Ey = tf.tile(mu[:, tf.newaxis, :], [1, order + timesteps, 1])
    # states = {'mu': Ey, 'vcv': vcv}
    return Ey, vcv, standardized_residuals


class DCC:
    """Class that represent a DCC model parametrisation.
     Allows for parameters mapping and filtering of mean-Covariance states, with truncated autoregressive estimate
    (as done with Recurrent Neural Networks).
    """
    def __init__(self, n, p=1, o=1, q=1, power=2.0):
        self.order = np.max([p, o, q])
        # dim = n * (2 + p + q + o)
        self.p = p
        self.o = o
        self.q = q
        self.n = n
        self.dim_garch = n * (p + o + q + 1)
        self.dim_arma = n * (p + o + q + 1)
        self.dim_corr = 2
        self.dim_all = self.dim_garch + self.dim_arma + self.dim_corr
        self.power = power
        self.pars_dict = None

    def infer_parameters0(self, y_train, y_test):
        """
        Infer the center of the reparametrisation
        :param y_train: Obervations of the train set
        :param y_test:  Obervations on the test set (not used to estimate DCC, only to evaluate it)
        :return: None
        """
        # Extract parameters
        self.pars_dict = extractionParameters_DCCrmgarch(
            y_train, y_test, ar=0, ma=0, p=self.p, q=self.q, o=self.o, power=self.power, vol='gjrGARCH')

    def _sample_pars(self, mc):
        """
        Takes mc particles in R^d and maps them to the parameters space using the center parameters already estimated.
        :param mc: The particles in R^d with dim k,d
        :return: The mapped DCC parameters
        """
        mu0, omega0, alpha0, beta0, gamma0, sigma_t0, corr0, a0, b0 = extracted_dict_pars_2_values(
            mc, self.o, self.order, self.p, self.q, self.pars_dict)
        return mu0, omega0, alpha0, beta0, gamma0, sigma_t0, corr0, a0, b0

    def model_evaluation(self, y, particles_t, particles):
        """
        Evaluate the mean-covariance states, the likelihood, the prior and the log-Jacobian of the mapping,
        using input parameters particles, on the batch of data.
        :param y: A batch of data, with dim t,d
        :param particles_t: IGNORED, written for compatibility with models with observation specifics parameters.
        :param particles: A line of particles with dim k,1. With k the number of DCC parameters.
        :return: states=mean-coviarance states vector, theta=DCC parameters,
                llkl=log-likelihood of the model, p0=log-prior value,
                log_jacobian=log of the Jacobian of the parameters shaping function,
                variables=list of tensorflow variables that affects the parameters mapping
        """
        states, theta, log_jacobian, variables = self.shape(particles_t, particles)
        p0 = prior_dcc(**theta)
        llkl = log_lkl_timeseries(y, **states)
        # Only the last vector of the lagged timesteps contains the final forecast
        return states, theta, llkl, p0, log_jacobian, variables

    def shape(self, particles_t, particles_global):
        """
        Public method for shaping parameters
        :param particles_t: IGNORED
        :param particles_global: tensor with dim t,d used to shape the parameters.
        :return: The mapped DCC parameters particles
        """
        mc = particles_global.shape[0]
        mu0, omega0, alpha0, beta0, gamma0, sigma_t0, corr0, a0, b0 = self._sample_pars(mc)
        return self._shape(particles_t, particles_global,
                           mu0, omega0, alpha0, beta0, gamma0, sigma_t0, corr0, a0, b0)

    def _shape(self,
               particles_t, particles_global,
               mu0, omega0, alpha0, beta0, gamma0, sigma_t0, corr0, a0, b0):
        power, p, o, q = self.power, self.p, self.o, self.q
        mu, omega, alpha, gamma, beta, jacobian = shape_garch(
            particles_global[:, :-2], mu0, omega0, alpha0, gamma0, beta0)
        Ey, vcv, standardized_residuals = filtering_garch(
            particles_t, power, mu, omega, alpha, gamma, beta, sigma_t0)
        # corr
        samples, jacobian = reconstructionParameters(particles_global[:, -2:], a0, b0)
        correlations_normalized = filtering_dcc(
            standardized_residuals, samples['corrA'], samples['corrB'], corr0, self.order)
        vcv_chol = tf.sqrt(vcv)
        vcv = tf.matmul(vcv_chol, tf.matmul(correlations_normalized, vcv_chol))

        theta = {
            'power': self.power,
            'mu': mu,
            'omega': omega,
            'a': samples['corrA'],
            'b': samples['corrB'],
            'corr0': corr0,
        }
        if q > 0:
            theta['beta'] = beta
        if p > 0:
            theta['alpha'] = alpha
        if o > 0:
            theta['gamma'] = gamma
        # To forecast use the last timesteps of the autoregressive dimension
        states = {
            'mu': Ey[:, -1],
            'vcv': vcv[:, -1]
        }
        return states, theta, jacobian, []

    def llkl_func(self, y, states, particles_global=None):
        """
        The log-likelihood of the model, given states and observations.
        :param y: The batch of observations
        :param states: The mean-covariance states
        :param particles_global: IGNORED, because the DCC parameters does not directly affect the likelihood
        :return: A tensor with likelihood values, with dim t,1
        """
        llkl = log_lkl_timeseries(y, states['mu'], vcv_tril=states['vcv_tril'])
        return llkl

    def prior(self, states, theta=None, index=None):
        """
        The log-prior of the model.
        :param states: The mean-covariance states
        :param theta: The DCC parameters
        :param index: The time index (to allow for time-specific priors)
        :return: A tensor with the log-prior values
        """
        return prior_dcc(**theta)


if __name__ == '__main__':
    mc = 2
    n = 3
    p = 0
    q = 1
    o = p
    order = np.max([p, o, q])
    dim = n * (2 + p + q + o) + 2
    # y = np.random.randn(mc, timesteps, n)
    y_train, y_valid = np.random.randn(200, 3), np.random.randn(200, 3)
    model = DCC(n=n, p=p, q=q, o=o)
    model.infer_parameters0(y_train, y_valid)
    particles = np.random.rand(mc, dim) * 0

    # sovle the issue of y vs particles_t
    particles_t = y_train[np.newaxis]  # np.random.rand(, 2) * 0
    states, theta, llkl, p0, log_jacobian, variables = model.model_evaluation(
        y_train, particles=particles, particles_t=particles_t)
