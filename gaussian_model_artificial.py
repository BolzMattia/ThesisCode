import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as sps
import matplotlib.pyplot as plt
import seaborn as sns
# local imports
from utils import indexes_librarian, matrix2line_diagFunc
from evaluators import tester_performances


class ToyMuEtaProblem:
    """
    Represent an artificial problem for Bayesian inference.
    The problem uses a conjugate Gaussian-Wishart prior with Gaussian likelihood.
    """
    def __init__(self,
                 T=None,
                 N=None,
                 real_mu=None,
                 real_eta=None,
                 p0_mu=None,
                 p0_k=None,
                 p0_nu=None,
                 p0_eta=None,
                 center_prior=False):
        '''
        Construct a toy Bayesian problem with Gaussian density.
        :param T: The number of observations in data
        :type T: int
        :param N: The dimension of the observations
        :type N: int
        :param real_mu: The vector of the observations mean
        :type real_mu: np.array
        :param real_eta: The matrix of the observations precision
        :type real_eta: np.array, 2-dim
        :param p0_mu: The vector of the prior mean
        :type p0_mu: np.array
        :param p0_k: The prior mean dispersion
        :type p0_k: float
        :param p0_nu: The prior precision dispersion
        :type p0_nu: float
        :param p0_eta: The matrix of the prior precision
        :type p0_eta: np.array, 2-dim
        '''
        if T is None: T = 20
        if N is None: N = 10
        if real_mu is None:
            real_mu = np.zeros(N, dtype=np.float32)
        if real_eta is None:
            real_eta = np.eye(N)#np.float32(sps.wishart.rvs(N*T, np.eye(N) / N*T))

        # Data generation
        y = np.float32(np.random.randn(T, N))
        y = y @ np.linalg.cholesky(np.linalg.inv(real_eta)) + real_mu
        if center_prior:
            if p0_mu is None:
                p0_mu = np.float32(np.zeros(N))
            if p0_k is None:
                p0_k = T
            if p0_nu is None:
                p0_nu = N
            if p0_eta is None:
                p0_eta = np.float32(np.eye(N))
        else:
            if p0_mu is None:
                p0_mu = np.float32(np.random.randn(N))
            if p0_k is None:
                p0_k = T
            if p0_nu is None:
                p0_nu = N
            if p0_eta is None:
                p0_eta = np.float32(sps.wishart.rvs(p0_nu, np.eye(N) / p0_nu) / p0_nu)
        self.p1_mu, self.p1_k, self.p1_nu, self.p1_eta = self.posterior_representation(
            y, p0_mu, p0_k, p0_nu, p0_eta)
        self.p0_mu, self.p0_k, self.p0_nu, self.p0_eta = p0_mu, p0_k, p0_nu, p0_eta
        self.y = y

    # @tf.function
    def prior(self, mu, eta=None, eta_tril=None, **kwargs):
        '''
        Compute the prior value on a set of parameters.
        All the parameters must have the first dimension equal to the number of samples.
        The other dimensions refers to the parameters dimension.
        :param mu: The mean parameter
        :type mu: tf.Tensor
        :param eta: The precision parameter
        :type eta: tf.Tensor
        :param eta_tril: The Cholesky decomposition of eta
        :type eta_tril: tf.Tensor
        :param kwargs: Other parameters
        :type kwargs: -
        :return: A tensor with the prior values
        :rtype: tf.Tensor
        '''
        p0_mu, p0_k, p0_nu, p0_eta = self.p0_mu, self.p0_k, self.p0_nu, self.p0_eta
        if eta_tril is None:
            vcv = tf.linalg.inv(eta)
            p0pdf_eta = tfp.distributions.Wishart(np.float32(p0_nu), np.float32(p0_eta),
                                                  input_output_cholesky=False)
            p0pdf_mu = tfp.distributions.MultivariateNormalFullCovariance(np.float32(p0_mu), vcv / p0_k)

            p0_mu = p0pdf_mu.log_prob(mu)
            p0_eta = p0pdf_eta.log_prob(eta)
            # log_det_vcv = tf.math.log(tf.linalg.det(vcv))
        else:
            tril = tf.linalg.inv(eta_tril)
            p0pdf_eta = tfp.distributions.Wishart(np.float32(p0_nu), np.float32(p0_eta),
                                                  input_output_cholesky=True)
            p0pdf_mu = tfp.distributions.MultivariateNormalTriL(np.float32(p0_mu), tril / np.sqrt(p0_k))

            p0_mu = p0pdf_mu.log_prob(mu)
            p0_eta = p0pdf_eta.log_prob(eta_tril)
            # log_det_vcv = 2 * tf.math.log(tf.linalg.det(tril))

        # jacobian_correction = log_det_vcv + np.log(2)
        p0 = p0_mu + p0_eta  # - jacobian_correction
        return p0

    def posterior_representation(self, y, p0_mu, p0_k, p0_nu, p0_eta):
        """
        Compute the posterior parameters.
        :param y: Observations array, dim T,N
        :param p0_mu: The prior mean
        :param p0_k: The prior mean-dispersion
        :param p0_nu: The prior Eta dispersion
        :param p0_eta: The prior Eta mean value
        :return: Posterior parameters as numpy arrays
        """
        T, N = y.shape

        ym = y.mean(axis=0)
        yc = y - np.tile(ym.reshape([1, N]), [T, 1])
        C = yc.transpose([1, 0]) @ yc
        y_cov = np.cov(y.transpose([1, 0]))

        p1_mu = (p0_k * p0_mu + T * ym) / (p0_k + T)
        p1_k = p0_k + T
        p1_nu = p0_nu + T
        p1_eta = np.linalg.inv(
            np.linalg.inv(p0_eta) + C + p0_k * T / (p0_k + T) * (ym - p0_mu).reshape([N, 1]) @ (ym - p0_mu).reshape(
                [1, N]))

        return p1_mu, p1_k, p1_nu, p1_eta

    def samples_p1(self, size):
        """
        Samples particles distributed as the posterior
        :param size: Number of particles to sample
        :return: The sampled particles as two numpy array (mean and covariance)
        """
        p1_mu, p1_k, p1_nu, p1_eta = self.p1_mu, self.p1_k, self.p1_nu, self.p1_eta
        N = p1_mu.shape[0]
        V = sps.wishart.rvs(df=p1_nu, scale=p1_eta, size=size)
        iV = np.linalg.inv(V)
        mus = np.zeros([size, N])
        for s in range(size):
            mus[s] = np.random.multivariate_normal(p1_mu, iV[s] / p1_k, size=1)
        return mus, V

    def samples_p0(self, size):
        """
        Samples particles distributed as the prior
        :param size: Number of particles to sample
        :return: The sampled particles as two numpy array (mean and covariance)
        """
        p0_mu, p0_k, p0_nu, p0_eta = self.p0_mu, self.p0_k, self.p0_nu, self.p0_eta
        N = p0_mu.shape[0]
        V = sps.wishart.rvs(df=p0_nu, scale=p0_eta, size=size)
        iV = np.linalg.inv(V)
        mus = np.zeros([size, N])
        for s in range(size):
            mus[s] = np.random.multivariate_normal(p0_mu, iV[s] / p0_k, size=1)
        return mus, V

    def p0_value(self, mu, eta):
        '''Evaluate the prior density'''
        p0_mu, p0_k, p0_nu, p0_eta = self.p0_mu, self.p0_k, self.p0_nu, self.p0_eta

        pdf_eta = sps.wishart.pdf(eta, p0_nu, p0_eta)
        pdf_mu = sps.multivariate_normal.pdf(mu, p0_mu, np.linalg.inv(eta) / p0_k)
        return pdf_mu * pdf_eta

    def p1_value(self, mu, eta):
        '''Evaluate the posterior density'''
        p1_mu, p1_k, p1_nu, p1_eta = self.p1_mu, self.p1_k, self.p1_nu, self.p1_eta

        pdf_eta = sps.wishart.pdf(eta, p1_nu, p1_eta)
        pdf_mu = sps.multivariate_normal.pdf(mu, p1_mu, np.linalg.inv(eta) / p1_k)
        return pdf_mu * pdf_eta

    def print_exact_posterior(self, MC=100000):
        '''
        This function prints some posterior statistics (mean of mu and determinant of eta),
        returns samples from its distribution.
        '''
        mus, etas = self.samples_p1(MC)
        _det_etas = np.linalg.det(etas)

        def log_bayes_factor():
            _MC = 2  # uses two particles for double-check
            mus, etas = self.samples_p1(_MC)
            V = np.linalg.inv(etas)
            bf = np.zeros(_MC)
            for _ in range(_MC):
                Lp = np.log(self.p0_value(mus[_], etas[_])) + np.log(
                    sps.multivariate_normal.pdf(self.y, mus[_], V[_])).sum()
                p1 = np.log(self.p1_value(mus[_], etas[_]))
                bf[_] = Lp - p1

            assert np.round(bf[0], 2) == np.round(bf[1], 2), 'Error in computing Bayes factor'
            return bf[0]

        _bf2 = log_bayes_factor()
        V = np.linalg.inv(etas)
        print(
            f'theta: (mean,std)\ndet eta:{_det_etas.mean(), _det_etas.std()}\nmu:{mus.mean(axis=0), mus.std(axis=0)}\nlog bayes factor:{_bf2}')
        return _bf2, {'mu': mus, 'eta': etas}


class shaper_mu_eta:
    def __init__(self, mu0=None, eta0=None, y=None):
        '''
        This class represents the operation of shaping a line tensor into a dictionary with model parameters.
        The transformation place the 0 vector to the prior mu and eta parameters.
        If the prior is not available, center the transormation in the data mean and inverse-covariance
        :param mu0: The prior mean
        :type mu0: np.array
        :param eta0: The prior precision
        :type eta0: np.array
        :param y: The data
        :type y: np.array
        '''
        if mu0 is None:
            mu0 = y.mean(axis=0)
        if eta0 is None:
            eta0 = np.eye(y.shape[1], dtype=np.float32)
        self.mu0 = mu0
        self.eta0 = eta0

    def shape(self, particles):
        '''
        Shape the line of particles as a dictionary of parameters.
        Returns also the log_density and the variables of the transformation.
        :param particles: The line of particles
        :type particles: tf.Tensor
        :return: The dictionary with the name of parameters, the tensor with the log density and the variables.
        :rtype: (dict[str: tf.Tensor], tf.Tensor, list)
        '''
        N = self.mu0.shape[0]
        # mu shaping
        mu = particles[:, :N] + self.mu0[np.newaxis, :]
        # eta shaping
        i_eta0 = matrix2line_diagFunc(self.eta0)
        diagSoftPlus = tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())

        samples_ieta_line = i_eta0[np.newaxis, :] + particles[:, N:]
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
        # The Jacobian, overall
        jacobian_particles = jacobian_softplus + jacobian_chol

        states = {'mu': mu, 'eta': eta, 'eta_tril': samples_etaCholesky}
        return states, jacobian_particles, []

    def shape_3dim(self, particles):
        """
        Same as shape but with particles that have an extra dimensions.
        The first dimension is considered the extra dimension and is intended as a batch dimension.
        :param particles: The line of particles, with dim batch,T,N
        :type particles: tf.Tensor
        :return: The dictionary with the name of parameters, the tensor with the log density and the variables.
        :rtype: (dict[str: tf.Tensor], tf.Tensor, list)
        """
        N = self.mu0.shape[0]
        # mu shaping
        mu = particles[:, :, :N] + self.mu0[np.newaxis, np.newaxis, :]
        # eta shaping
        i_eta0 = matrix2line_diagFunc(self.eta0)
        diagSoftPlus = tfp.bijectors.TransformDiagonal(tfp.bijectors.Softplus())
        samples_ieta_line = i_eta0[np.newaxis, np.newaxis, :] + particles[:, :, N:]
        samples_ietaCholesky = tfp.math.fill_triangular(samples_ieta_line)
        samples_ietaDiag = tf.gather(samples_ieta_line, indexes_librarian(N).spiral_diag, axis=-1)
        samples_etaDiag = tf.nn.softplus(samples_ietaDiag)  # +1e-7 #Avoids numerical underflow
        samples_etaCholesky = diagSoftPlus.forward(samples_ietaCholesky)
        eta = tf.matmul(samples_etaCholesky, samples_etaCholesky, transpose_b=True)

        # The Jacobian of the transformation of the Cholesky samples
        cholesky_factor = (N - np.arange(N))[np.newaxis, np.newaxis, :]
        jacobian_chol = tf.reduce_sum(cholesky_factor * tf.math.log(samples_etaDiag), axis=-1) + N * np.log(2)
        # The Jacobian of the softplus on diagonal elements
        jacobian_softplus = tf.reduce_sum(tf.math.log(tf.nn.sigmoid(samples_ietaDiag)), axis=-1)
        # The Jacobian, overall
        jacobian_particles = jacobian_softplus + jacobian_chol

        states = {'mu': mu, 'eta': eta, 'eta_tril': samples_etaCholesky}
        return states, jacobian_particles, []


def lkl_mu_eta(y, mu, vcv=None, vcv_tril=None, eta=None, eta_tril=None, **kwargs):
    '''
    Computes the likelihood of the Gaussian model, given the parameters and the observations y.
    :param y: observations
    :type y: np.array
    :param mu: Mean parameters
    :type mu: tf.Tensor
    :param eta: Precision parameters
    :type eta: tf.Tensor
    :param eta_tril: Cholesky decomposition of eta
    :type eta_tril: tf.Tensor
    :param kwargs: Other parameters for the model
    :type kwargs: -
    :return: Return a tensor with the likelihood for every set of parameters
    :rtype: tf.Tensor
    '''
    T, _ = y.shape
    if vcv is not None:
        pdf = tfp.distributions.MultivariateNormalFullCovariance(mu, vcv)
    elif vcv_tril is not None:
        pdf = tfp.distributions.MultivariateNormalTriL(mu, vcv_tril)
    elif eta_tril is not None:
        tril = tf.linalg.inv(eta_tril)
        pdf = tfp.distributions.MultivariateNormalTriL(mu, tril)
    else:
        vcv = tf.linalg.inv(eta)
        pdf = tfp.distributions.MultivariateNormalFullCovariance(mu, vcv)
    return tf.reduce_mean(pdf.log_prob(y[:, np.newaxis]), axis=0) * T


def plot_results(df_results, save_path=None, save_formats=['eps', 'png']):
    """
    Plot the results in df_results and save the plot in save_path with the formats in save_formats
    :param df_results: pandas Dataframe with the results (columns time,distance,method -> required)
    :param save_path: str with the directory to save the plot
    :param save_formats: Which formats to save the plot files
    :return: None
    """
    moms = df_results.moment.unique()
    fig, axes = plt.subplots(1, len(moms), figsize=[10, 4])

    i = 0
    for mom in moms:
        ax = axes[i]
        df = df_results.loc[(df_results.moment == mom)]
        sns.lineplot(x='time', y='distance', data=df, hue='method', ax=ax)
        ax.set_xlabel('time (minutes)')
        ax.set_title(f'{mom}')
        ax.set_yscale('log')
        i += 1
    if save_path is not None:
        for save_format in save_formats:
            fig.savefig(f'{save_path}.{save_format}', format=save_format)
    else:
        fig.show()


class gaussian_problem:
    def __init__(self, N, T, center_prior=False, **kwargs):
        '''
        Wrapper that creates a Gaussian problem.
        :param N: Dimension of the observations
        :type N: int
        :param T: Number of observations
        :type T: int
        :param kwargs: Other parameters
        :type kwargs: -
        '''
        d = int(N * (N + 1) / 2) + N
        problem = ToyMuEtaProblem(T, N, center_prior=center_prior)
        bayes_factor, truth = problem.print_exact_posterior()
        self.bayes_factor = bayes_factor
        self.d = d
        self.truth = truth
        self.y = problem.y
        self.problem = problem
        # shaper
        self.parameters_shaper = shaper_mu_eta(y=self.y).shape
        self.lkl_func = lkl_mu_eta
        self.prior_func = problem.prior

    def get_evaluator(self):
        """Create and evaluator withe self.truth as true posterior particles."""
        return tester_performances(self.truth, mean_squared_error)

    def plot_results(self, evaluator, save_path=None):
        """Plot the results stored in the evaluator and save them to save_path"""
        results = evaluator.get_pandas()
        if results is not None:
            plot_results(results, save_path)


if __name__ == '__main__':
    config = {
        'kind': 'gaussian',
        'N': 8,
        'T': 50,
    }
    problem = gaussian_problem(**config)
    evaluator = problem.get_evaluator()
    N = config['N']
    d = int(N * (N + 1) / 2) + N
    MC = 10000
    particles = np.random.rand(MC, d)

    theta, log_j, variables = problem.parameters_shaper(particles)
    print(np.mean(log_j))
