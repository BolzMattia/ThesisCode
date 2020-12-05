import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from utils import indexes_librarian, matrix2line_diagFunc_timeseries
from sklearn.covariance import LedoitWolf


def num_parameters_mean_cholesky(n):
    """Compute the number of parameters required to
    reparametrize a mean vector + covariance matrix."""
    return int(n + n * (n + 1) / 2)


def shape_mu_vcv(x, N):
    """
    Takes a line vector x with the right dimension and maps it into a mean vector+ covariance matrix.
    :param x: line vector with dim batch_size,num_parameters_mean_cholesky(N)
    :param N: The dimension of the mean vector
    :return: mean vector, covariance matrix (both arrays with batch_size as first dim)
    """
    diag_soft_plus = tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
    cholesky_line = x[:, N:]
    mu = x[:, :N]
    cholesky = tfp.math.fill_triangular(cholesky_line)
    diag = tf.gather(cholesky_line, indexes_librarian(N).spiral_diag, axis=1)
    diag = tf.nn.softplus(diag)  # +1e-7 #Use this correction to avoid numerical underflow
    cholesky = diag_soft_plus.forward(cholesky)
    vcv = tf.matmul(cholesky, cholesky, transpose_b=True)
    # The Jacobian of the transformation of the Cholesky samples
    cholesky_factor = (N - np.arange(N))[np.newaxis, :]
    jacobian_chol = tf.reduce_sum(cholesky_factor * tf.math.log(diag), axis=1) + N * np.log(2)
    # The Jacobian of the softplus on diagonal elements
    jacobian_softplus = tf.reduce_sum(tf.math.log(tf.nn.sigmoid(diag)), axis=1)
    # The Jacobian, overall
    jacobian_particles = jacobian_softplus + jacobian_chol
    return mu, vcv, cholesky, jacobian_particles


def mu_vcv_linearize(mu, vcv):
    """
    Convert numpy states mu,vcv into linearized vectors
    :param mu: The mean vectors, dim=batch_size,n
    :param vcv: The covariance matrix, dim=batch_size,n,n
    :return: A line vector, dim=batch_size,num_parameters_mean_cholesky(n)
    """
    vcv_line = matrix2line_diagFunc_timeseries(vcv)
    line = np.concatenate([mu, vcv_line], axis=1)
    return line


def mean_covariance_llkl(y, mu, vcv=None, vcv_tril=None, eta=None, eta_tril=None):
    """Compute the log-likelihood of a mean covariance model."""
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


def empirical_tvp_prior(mu_vcv_line_center, scale=1.0):
    """
    Compute an empirical prior over a linearized vector of a mean,covariance series of states.
    :param mu_vcv_line_center: line vector with dim=t,num_parameters_mean_cholesky(n)
    :param scale: Scale of the Gaussian prior
    :return: A function evaluating the empirical gaussian prior
    """
    mu_dline, vcv_dline = line_center_prior_d(mu_vcv_line_center, scale)
    mu_line, vcv_line = line_center_prior(mu_vcv_line_center, scale)
    pdf_dline = tfp.distributions.MultivariateNormalFullCovariance(mu_dline, vcv_dline * (scale ** 2))
    pdf_line = tfp.distributions.MultivariateNormalFullCovariance(mu_line, vcv_line * scale ** 2)

    def prior(line, *args):
        return tf.concat([pdf_line.log_prob(line[:1]), pdf_dline.log_prob(line[1:] - line[:-1])], axis=0)
        # return tf.concat([np.float32([0.0]), pdf_dline.log_prob(line[1:] - line[:-1])], axis=0)

    return prior


def line_center_prior_d(mu_vcv_line_center, scale, use_lw=True):
    """
    Compute an empirical prior over the variations of a linearized vector
    of a mean,covariance series of states.
    :param mu_vcv_line_center: line vector with dim=t,num_parameters_mean_cholesky(n)
    :param scale: Scale of the Gaussian prior
    :return: A function evaluating the empirical gaussian prior over the linearized variations.
    """
    dline = mu_vcv_line_center[1:] - mu_vcv_line_center[:-1]
    if use_lw:
        vcv_dline = np.float32(LedoitWolf().fit(dline).covariance_)
    else:
        vcv_dline = np.float32(np.cov(np.transpose(dline, [1, 0])))
    if np.linalg.det(vcv_dline) < 1e-16:
        print('Correction term added to singular prior')
        vcv_dline += np.eye(dline.shape[1]) * 1e-3 * scale
    mu_dline = dline.mean(axis=0)
    vcv_dline *= scale
    return mu_dline, vcv_dline


def line_center_prior(mu_vcv_line_center, scale, use_lw=True):
    """
    Compute an empirical prior over the values of a linearized vector
    of a mean,covariance series of states.
    :param mu_vcv_line_center: line vector with dim=t,num_parameters_mean_cholesky(n)
    :param scale: Scale of the Gaussian prior
    :return: A function evaluating the empirical gaussian prior over the linearized values directly.
    """
    if use_lw:
        vcv_line = np.float32(LedoitWolf().fit(mu_vcv_line_center).covariance_)
    else:
        vcv_line = np.float32(np.cov(np.transpose(mu_vcv_line_center, [1, 0])))
    if np.linalg.det(vcv_line) < 1e-16:
        print('Correction term added to singular prior')
        vcv_line += np.eye(mu_vcv_line_center.shape[1]) *1e-1 * scale
    mu_line = mu_vcv_line_center.mean(axis=0)
    vcv_line *= scale
    return mu_line, vcv_line


def no_prior():
    """Return a function evaluating a flat prior."""

    # pdf = tfp.distributions.Normal(np.float32(0.0),np.float32(1000.0))
    def prior(*args):
        return tf.zeros(1, dtype=tf.float32)

    return prior


if __name__ == '__main__':
    N = 3
    d = num_parameters_mean_cholesky(N)
    mc = 10
    x = np.float32(np.random.randn(mc, d))
    mu, vcv, chol, lj = shape_mu_vcv(x, N)
    prior = empirical_tvp_prior(x)
    pi_values = prior(x)
    print(mu.shape, vcv.shape, pi_values)
