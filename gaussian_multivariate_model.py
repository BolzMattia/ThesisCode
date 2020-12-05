import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import scipy.stats as sps
from sklearn.covariance import LedoitWolf
from utils import matrix2line_diagFunc, matrix2line_diagFunc_timeseries, indexes_librarian


def log_lkl(y, mu, vcv=None, vcv_tril=None, eta=None, eta_tril=None):
    """log-likelihood of a Gaussian model"""
    assert (
            (vcv is not None) | (vcv_tril is not None) | (eta is not None) | (eta_tril is not None)
    ), "Must provide at least one covariance structure"
    if vcv_tril is not None:
        pdf = tfp.distributions.MultivariateNormalTriL(mu, vcv_tril)
    elif vcv is not None:
        pdf = tfp.distributions.MultivariateNormalFullCovariance(mu, vcv)
    elif eta_tril is not None:
        tril = tf.linalg.inv(eta_tril)
        pdf = tfp.distributions.MultivariateNormalTriL(mu, tril)
    else:
        vcv = tf.linalg.inv(eta)
        pdf = tfp.distributions.MultivariateNormalFullCovariance(mu, vcv)
    return tf.reduce_sum(pdf.log_prob(y[:, np.newaxis]), axis=0)


def log_lkl_timeseries(y, mu, vcv=None, vcv_tril=None, eta=None, eta_tril=None):
    """log-likelihood of a timeseries Gaussian model (input y has dimensions t,n)"""
    assert (
            (vcv is not None) | (vcv_tril is not None) | (eta is not None) | (eta_tril is not None)
    ), "Must provide at least one covariance structure"
    if vcv_tril is not None:
        pdf = tfp.distributions.MultivariateNormalTriL(mu, vcv_tril)
    elif vcv is not None:
        pdf = tfp.distributions.MultivariateNormalFullCovariance(mu, vcv)
    elif eta_tril is not None:
        tril = tf.linalg.inv(eta_tril)
        pdf = tfp.distributions.MultivariateNormalTriL(mu, tril)
    else:
        vcv = tf.linalg.inv(eta)
        pdf = tfp.distributions.MultivariateNormalFullCovariance(mu, vcv)
    return pdf.log_prob(y)


def prior_conjugate_np(mu, eta, p0_mu, p0_k, p0_nu, p0_eta):
    """Evaluate the conjugate Gaussian-Wishart prior"""
    pdf_eta = sps.wishart.pdf(eta, p0_nu, p0_eta)
    pdf_mu = sps.multivariate_normal.pdf(mu, p0_mu, np.linalg.inv(eta) / p0_k)
    return pdf_mu * pdf_eta


def _prior_muEta(p0_mu, p0_k, p0_nu, p0_eta, mu, eta):
    """Evaluate the conjugate Gaussian-Wishart prior"""
    vcv = tf.linalg.inv(eta)
    p0pdf_eta = tfp.distributions.Wishart(np.float32(p0_nu), np.float32(p0_eta),
                                          input_output_cholesky=False)
    p0pdf_mu = tfp.distributions.MultivariateNormalFullCovariance(np.float32(p0_mu), vcv / p0_k)

    p0_mu = p0pdf_mu.log_prob(mu)
    p0_eta = p0pdf_eta.log_prob(eta)

    return p0_mu + p0_eta


def _prior_muEtaTril(p0_mu, p0_k, p0_nu, p0_eta, mu, eta_tril):
    """Evaluate the conjugate Gaussian-Wishart prior on the Cholesky decomposition of the precision matrix"""
    tril = tf.linalg.inv(eta_tril)
    p0pdf_eta = tfp.distributions.Wishart(np.float32(p0_nu), np.float32(p0_eta),
                                          input_output_cholesky=True)
    p0pdf_mu = tfp.distributions.MultivariateNormalTriL(np.float32(p0_mu), tril / np.sqrt(p0_k))
    p0_mu = p0pdf_mu.log_prob(mu)
    p0_eta = p0pdf_eta.log_prob(eta_tril)

    return p0_mu + p0_eta


def prior_conjugate(p0_mu, p0_k, p0_nu, p0_eta, mu, eta=None, eta_tril=None, **kwargs):
    """Evaluate the conjugate Gaussian-Wishart prior on the Cholesky decomposition of the precision matrix"""
    if eta_tril is None:
        p0 = _prior_muEta(p0_mu, p0_k, p0_nu, p0_eta, mu, eta)
    else:
        p0 = _prior_muEtaTril(p0_mu, p0_k, p0_nu, p0_eta, mu, eta_tril)
    return p0


def prior_vector_variability(x):
    """
    Estimate the covariance matrix of x with the LedoitWolf estimator
    :param x: an array of dim (t,n)
    :return: The estimated covariance matrix
    """
    dx = LedoitWolf().fit(x).covariance_
    return dx


def prior_matrix_variability(vcv):
    """
    Estimate the covariance matrix of the variations of the coefficients of a timeseries of covariance matrixes,
     with the LedoitWolf estimator.
    :param vcv: the series of covariance matrixes, an array of dim (t,n,n)
    :return: The estimated covariance matrix of the coefficients (dim = n**2,n**2)
    """
    t, n, _ = vcv.shape
    assert n == _
    lv = np.reshape(vcv[1:] - vcv[:-1], [t - 1, n ** 2])
    dvcv = LedoitWolf().fit(lv).covariance_
    return dvcv


def prior_centeredMuVcv(p0_mu, p0_dmu, p0_vcv, p0_dvcv, mu, vcv=None, vcv_tril=None, **kwargs):
    """
    DISPOSED
    """
    t, n, _ = p0_vcv.shape
    p0pdf_mu = tfp.distributions.MultivariateNormalFullCovariance(np.float32(np.zeros(n)), p0_dmu)
    p0pdf_vcv = tfp.distributions.MultivariateNormalFullCovariance(np.float32(np.zeros(n ** 2)), p0_dvcv)
    pi_mu = p0pdf_mu.log_prob(mu - p0_mu)
    pi_vcv = p0pdf_vcv.log_prob(np.reshape(vcv - p0_vcv, [t, n ** 2]))
    return pi_mu + pi_vcv


def shaper_MuVcv(particles, mu0, eta0=None, eta0_line=None):
    '''
    Shape the line of particles as a dictionary of parameters.
    Returns also the log_density and the variables of the transformation.
    :param particles: The line of particles
    :type particles: tf.Tensor
    :param mu0: The center of the mean reconstruction (any 0-vector particle is mapped in mu0)
    :type mu0: np.ndarray
    :param eta0: The center of the precision reconstruction (any 0-vector particle is mapped in eta0)
    :type eta0: np.ndarray
    :param eta0_line: The linearized version of eta0 (if not provided, is computed from eta0)
    :type eta0_line: np.ndarray
    :return: The dictionary with the name of parameters, the tensor with the log density and the variables.
    :rtype: (dict[str: tf.Tensor], tf.Tensor, list)
    '''
    N = mu0.shape[0]
    # mu shaping
    mu = particles[:, :N] + mu0[np.newaxis, :]
    # eta shaping
    eta, jacobian_particles, samples_etaCholesky = shape_Eta(particles[:, N:], eta0, eta0_line)
    states = {'mu': mu, 'eta': eta, 'eta_tril': samples_etaCholesky}
    return states, jacobian_particles, []


def shape_Eta(particles, eta0, eta0_line=None):
    """Same as shape_MuVcv but only shapes a precision, positive definite, matrix"""
    if eta0_line is None:
        eta0_line = matrix2line_diagFunc(eta0)
    N = eta0.shape[-1]
    diagSoftPlus = tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
    samples_ieta_line = eta0_line[np.newaxis, :] + particles
    samples_ietaCholesky = tfp.math.fill_triangular(samples_ieta_line)
    samples_ietaDiag = tf.gather(samples_ieta_line, indexes_librarian(N).spiral_diag, axis=1)
    samples_etaDiag = tf.nn.softplus(samples_ietaDiag)  # +1e-7 #Use this correction to avoid numerical underflow
    samples_etaCholesky = diagSoftPlus.forward(samples_ietaCholesky)
    eta = tf.matmul(samples_etaCholesky, samples_etaCholesky, transpose_b=True)
    # The Jacobian of the transformation of the Cholesky samples
    cholesky_factor = (N - np.arange(N))[np.newaxis, :]
    jacobian_chol = tf.reduce_sum(cholesky_factor * tf.math.log(samples_etaDiag), axis=1) + N * np.log(2)
    # The Jacobian of the softplus on diagonal elements
    jacobian_softplus = tf.reduce_sum(tf.math.log(tf.nn.sigmoid(samples_ietaDiag)), axis=1)
    # The Jacobian, overall
    jacobian_particles = jacobian_softplus + jacobian_chol
    return eta, jacobian_particles, samples_etaCholesky


def shaper_MuVcv_timeseries(particles, mu0, v0=None, v0_line=None):
    '''
    Shape the line of particles as a dictionary of parameters.
    Dimension of particles must be (:, n*(n+1)/2 + n), where n is the dimension of mu0.
    Returns also the log_density and the variables of the transformation.
    :param particles: The line of particles
    :type particles: tf.Tensor
    :param mu0: The center of the mean reconstruction (any 0-vector particle is mapped in mu0)
    :type mu0: np.ndarray
    :param v0: The center of the precision reconstruction (any 0-vector particle is mapped in eta0)
    :type v0: np.ndarray
    :param v0_line: The linearized version of eta0 (if not provided, is computed from eta0)
    :type v0_line: np.ndarray
    :return: The dictionary with the name of parameters, the tensor with the log density and the variables.
    :rtype: (dict[str: tf.Tensor], tf.Tensor, list)
    '''
    mu_scale = 1e-4
    if len(mu0.shape) == 1:
        N = mu0.shape[0]
        # mu shaping
        mu = particles[:, :N] * mu_scale + mu0[np.newaxis, :]
        # eta shaping
        if v0_line is None:
            v0_line = matrix2line_diagFunc(v0)
        samples_ieta_line = particles[:, N:] * mu_scale + v0_line[np.newaxis, :]
    else:
        T, N = mu0.shape
        # mu shaping
        mu = particles[:, :N] + mu0
        # eta shaping
        if v0_line is None:
            v0_line = matrix2line_diagFunc_timeseries(v0)
        samples_ieta_line = particles[:, N:] + v0_line

    diagSoftPlus = tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
    samples_ietaCholesky = tfp.math.fill_triangular(samples_ieta_line)
    samples_ietaDiag = tf.gather(samples_ieta_line, indexes_librarian(N).spiral_diag, axis=1)
    samples_etaDiag = tf.nn.softplus(samples_ietaDiag)  # +1e-7 #Avoids numerical underflow
    samples_etaCholesky = diagSoftPlus.forward(samples_ietaCholesky)
    eta = tf.matmul(samples_etaCholesky, samples_etaCholesky, transpose_b=True)

    # The Jacobian of the transformation of the Cholesky samples
    cholesky_factor = (N - np.arange(N))[np.newaxis, :]
    jacobian_chol = tf.reduce_sum(cholesky_factor * tf.math.log(samples_etaDiag), axis=1) + N * np.log(2)
    # The Jacobian of the softplus on diagonal elements
    jacobian_softplus = tf.reduce_sum(tf.math.log(tf.nn.sigmoid(samples_ietaDiag)), axis=1)
    # Jacobian for mean scale
    jacobian_mu = tf.math.log(mu_scale)
    # The Jacobian, overall
    jacobian_particles = jacobian_softplus + jacobian_chol + jacobian_mu
    states = {'mu': mu, 'vcv': eta, 'vcv_tril': samples_etaCholesky}
    return states, jacobian_particles, []


def prior_dMudVcv(dmu, dvcv, dmu0, dvcv0, dmu_df, dvcv_df):
    """
    Compute the prior density for the variations of coefficients
    of two timeseries: one for the mean vectors and one for the covariance vectors
    :param dmu: The variations of the mean coefficients
    :param dvcv: The variations of the covariance coefficients
    :param dmu0: The mean of the variations in the mean coefficients
    :param dvcv0: The mean of the variations in the covariance coefficients
    :param dmu_df: The degree of freedom of the variations mean Wishart's distribution
    :param dvcv_df: The degree of freedom of the variations covariance Wishart's distribution
    :return:
    """
    p0pdf_dvcv = tfp.distributions.Wishart(np.float32(dvcv_df), np.float32(dvcv0),
                                           input_output_cholesky=False)
    p0pdf_dmu = tfp.distributions.Wishart(np.float32(dmu_df), np.float32(dmu0),
                                          input_output_cholesky=False)
    p0_dmu = p0pdf_dmu.log_prob(dmu)
    p0_dvcv = p0pdf_dvcv.log_prob(dvcv)
    return p0_dmu + p0_dvcv


def prior_MuVcv_timeseries(mu, dmu, vcv, dvcv, mu0, vcv0, dmu0, dvcv0, dmu_df, dvcv_df):
    """Compute both the prior, centered and variations based and sum them."""
    t = mu.shape[0]
    p0 = prior_centeredMuVcv(mu0, dmu, vcv0, dvcv, mu, vcv)
    p0_d = prior_dMudVcv(dmu, dvcv, dmu0, dvcv0, dmu_df, dvcv_df)
    return p0 + p0_d / t


class gaussian_timeseries_centered_model:
    """DEPRECATED"""
    def __init__(self, train):
        self.read_train(train)

    def model_evaluation(self, y, particles_t, particles, mu, vcv):
        states, theta, log_jacobian, variables = self.shape(particles_t, particles, mu, vcv)
        p0 = prior_MuVcv_timeseries(
            mu, theta['dmu'],
            vcv, theta['dvcv'],
            states['mu'], states['vcv'],
            self.dmu0, self.dvcv0, self.dmu_df, self.dvcv_df)
        # p0 = prior_centeredMuVcv(mu, self.dmu, vcv, self.dvcv, states['mu'], states['vcv'])
        llkl = log_lkl_timeseries(y, states['mu'], vcv_tril=states['vcv_tril'])
        return states, theta, llkl, p0, log_jacobian, variables

    def shape(self, particles_t, particles_global, mu, vcv):
        vcv_line = matrix2line_diagFunc_timeseries(vcv)
        z_t, log_jacobian, variables = shaper_MuVcv_timeseries(particles_t, mu, v0_line=vcv_line)
        #Forces the mean untouched
        z_t['mu'] = mu
        dvcv, logJ_dvcv, _ = shape_Eta(particles_global[:, self.global_dim_dvcv], self.dvcv0)
        dmu, logJ_dmu, _ = shape_Eta(particles_global[:, self.global_dim_dvcv:], self.dmu0)
        theta = {'dmu': dmu[0], 'dvcv': dvcv[0]}
        return z_t, theta, log_jacobian + logJ_dvcv + logJ_dmu, variables

    def read_train(self, train):
        self.mu_train = train['mu']
        self.vcv_train = train['vcv']
        self.vcv_line_train = matrix2line_diagFunc_timeseries(self.vcv_train)
        # Prior variability
        self.dvcv0 = prior_matrix_variability(self.vcv_train) * 10
        n = self.mu_train.shape[1]
        self.global_dim_dmu = int(n * (n + 1) / 2)
        self.global_dim_dvcv = int((n ** 2) * ((n ** 2) + 1) / 2)
        self.global_dim = self.global_dim_dmu + self.global_dim_dvcv
        min_det = 1e-20
        if np.linalg.det(self.dvcv0) == 0:
            self.dvcv0 += np.eye(n ** 2) * (min_det ** (1 / n ** 2))
        self.dvcv_df = n ** 2
        self.dmu0 = prior_vector_variability(self.mu_train)
        if np.linalg.det(self.dmu0) == 0:
            self.dmu0 += np.eye(n) * (min_det ** (1 / n))
        self.dmu_df = n

    def llkl_func(self, y, states, particles_global=None):
        llkl = log_lkl_timeseries(y, states['mu'], vcv_tril=states['vcv_tril'])
        return llkl

    def prior_train(self, states, theta=None, index=None):
        mu = self.mu_train
        vcv = self.vcv_train
        t = mu.shape[0]
        if index is not None:
            mu, vcv = mu[index], vcv[index]
        p0 = prior_MuVcv_timeseries(
            mu, theta['dmu'],
            vcv, theta['dvcv'],
            states['mu'], states['vcv'],
            self.dmu0, self.dvcv0, self.dmu_df, self.dvcv_df)
        return p0

    def prior_theta(self, dmu, dvcv, dmu0, dvcv0, dmu_df, dvcv_df):
        p0pdf_dvcv = tfp.distributions.Wishart(np.float32(dvcv_df), np.float32(dvcv0),
                                               input_output_cholesky=False)
        p0pdf_dmu = tfp.distributions.Wishart(np.float32(dmu_df), np.float32(dmu0),
                                              input_output_cholesky=False)
        p0_dmu = p0pdf_dmu.log_prob(dmu)
        p0_dvcv = p0pdf_dvcv.log_prob(dvcv)
        return p0_dmu, p0_dvcv

    def shape_train(self, particles_t, particles_global=None, index=None):
        mu = self.mu_train
        vcv_line = self.vcv_line_train
        if index is not None:
            mu, vcv_line = mu[index], vcv_line[index]
        z_t, log_jacobian, variables = shaper_MuVcv_timeseries(particles_t, mu, v0_line=vcv_line)
        #Forces the mean untouched
        z_t['mu'] = mu
        if particles_global is None:
            particles_global = np.zeros(self.global_dim, dtype=np.float32)
        dvcv = shape_Eta(particles_global[:self.global_dim_dvcv], self.dvcv0)
        dmu = shape_Eta(particles_global[:self.global_dim_dmu], self.dmu0)
        theta = {'dmu': dmu, 'dvcv': dvcv}
        return z_t, theta, log_jacobian, variables

    def model_evaluation_train(self,
                               y,
                               particles_t,
                               particles_global=None,
                               index=None):
        states, theta, log_jacobian, variables = self.shape_train(particles_t, particles_global=particles_global,
                                                                  index=index)
        p0 = self.prior_train(states, theta, index=index)
        llkl = self.llkl_func(y, states, theta)
        return llkl, p0, log_jacobian, variables


if __name__ == '__main__':
    data_config = {
        'name': 'FRED'
    }
    train_test_config = {
        'sequential': True,
        'test_size': 0.3
    }
    from econometrics_problem import DatasetFactory
    import timeseries_transformations as T

    dsf = DatasetFactory.get(**data_config)
    train_time = dsf.split_train_test(**train_test_config)

    dsf.withColumn('y', *T.standardize(*dsf['y']))
    dsf.withColumn('dcc', *T.dcc(*dsf['y']))
    dsf.withColumn('ledoitWolf_static', *T.staticLedoitWolf(*dsf['y']))

    gt = gaussian_timeseries_centered_model(dsf['dcc'][0])

    t_train, n = dsf.get_length()[0]
    t_test, n = dsf.get_length()[1]
    d = int(n * (n + 1) / 2) + n
    particles_t = np.float32(np.random.randn(t_train, d) * 1e-7)
    particles_global = np.float32(np.random.randn(gt.global_dim) * 1e-7)
    states, theta, llkl, p0, log_jacobian, variables_model = gt.model_evaluation(
        dsf['y'][0], particles_t, particles_global, **dsf['dcc'][0]
    )
    print(llkl, p0)
    # z_t_train, theta_train, log_p_train, variables = gt.shape_train(particles_t, particles)
    # print(gt.model_evaluation_train(dsf['y'][0], z_t_train, theta_train))
    # print(np.linalg.det(gt.dvcv), np.linalg.det(gt.dmu))
    # print(gt.prior_func(*dsf['ledoitWolf_static']))
    # print(gt.prior_func(*dsf['dcc']))
