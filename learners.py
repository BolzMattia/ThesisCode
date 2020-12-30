import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from utils import print_formatted_values
import timeseries_transformations as T
import stochastic_transformations as S
import mean_covariance_models as M
import optimizers as O
from econometrics_problem import DatasetFactory
from hvi import hierarchical_sampler
from sklearn.decomposition import PCA


def decorator_multi_particles_predict(predict):
    """
    Decorator of a predict function of a probabilistic forecaster.
    Runs the predict n_particles time and combine the forecast with equal probability.
    :param predict: The predict function
    :return: The decorated predict operation
    """

    def multi_particles_predict(n_particles, **kwargs):
        llkls_all = []
        prior_all = []
        regularization_penalty_all = []
        states_all = []
        for i in range(n_particles):
            states, theta, llkl, prior, regularization_penalty = predict(**kwargs)
            llkls_all.append(llkl)
            prior_all.append(prior)
            regularization_penalty_all.append(regularization_penalty)
            states_all.append(states)
        log_n_particles = np.log(n_particles)
        llkl = tf.reduce_logsumexp(np.stack(llkls_all, axis=0), axis=0) - log_n_particles
        prior = tf.reduce_logsumexp(np.stack(prior_all, axis=0), axis=0) - log_n_particles
        penalty = tf.reduce_logsumexp(np.stack(regularization_penalty_all, axis=0), axis=0) - log_n_particles
        return states, theta, llkl, prior, penalty

    return multi_particles_predict


class AbstractCholeskyForecaster:
    """This class represent a probabilistic forecaster that maps the input to a line vector
    then reconstruct mean-cvoariance state variables reparametrizing the line vector.
    The parameters learning happens through SGD optimization and a prior computation is managed."""

    def __init__(self, shaper, prior, optimizer, x_col, y_col, states0_line_col,
                 jacobian_penalty=False, n_particles_forecast=1):
        self.shaper = shaper
        self.prior = prior
        self.optimizer = optimizer
        self.x_col = x_col
        self.y_col = y_col
        self.states0_line_col = states0_line_col
        self.n_particles_forecast = n_particles_forecast
        self.jacobian_penalty = jacobian_penalty

    def fit(self, dsf, epochs=1000, batch_size=16, verbose=0):
        """
        Fit method that trains the internal parameters, using the observations and features in dsf.
        :param dsf: wrapperPandasCsv object
        :param epochs: How many train epochs
        :param batch_size: How many observation for every training batch
        :param verbose: Control how many information to print
        :return: A list with the score+prior+regularization term at every train step.
        """
        elbos = []
        for i in range(epochs):
            elbos.append(self.fit_step(dsf, batch_size, verbose=verbose))
        return elbos

    def posterior(self, x, y, states_line):
        """
        Evaluate the model posterior
        :param x: The input features
        :param y: The target variables
        :param states_line: The center states of the model
                            (in the form of a line vector with the correct number of parameters)
        :return: (log-likelihood, log-prior, regularization term,
                  mu particles, vcv particles, the variables affecting the particles)
        """
        y_hat, regularization_penalty, variables = self.shaper(x, 0.0, [])
        y_hat += states_line
        # Model
        mu, vcv, _, log_J = M.shape_mu_vcv(y_hat, y.shape[1])
        if self.jacobian_penalty:
            regularization_penalty -= log_J
        llkl = M.mean_covariance_llkl(y, mu, vcv)
        lprior = self.prior(y_hat, variables)

        return llkl, lprior, regularization_penalty, mu, vcv, variables

    def fit_step(self, dsf, batch_size, verbose=0):
        """The internal function for training. Compute only one train step with SGD."""
        # Read the batch data to fit
        batch, index_batch, times_batch = dsf.get_batch(
            self.x_col, self.y_col, self.states0_line_col, size=batch_size)
        x_t, y_t, states0_line = batch
        with tf.GradientTape() as t:
            # Computes the model posterior
            llkl, lprior, regularization_penalty, _, _, variables = self.posterior(x_t, y_t, states0_line)
            # Correct the log-likelihood estimator wrt the batch size
            elbo = (llkl + lprior) * dsf.get_length()[0][0] - regularization_penalty
            elbo = tf.reduce_mean(elbo)
            _elbo = elbo.numpy()
            if verbose > 0:
                print_formatted_values(elbo=_elbo,
                                       llkl=np.mean(llkl.numpy()),
                                       lprior=np.mean(lprior.numpy()),
                                       penalty=np.mean(regularization_penalty.numpy()))
            g = t.gradient(-elbo, variables)
        # optimize
        self.optimizer.optimize(g, variables)
        return _elbo

    def density_forecast(self,
                         dsf=None,
                         x=None, y=None, states0=None,
                         on_test=True):
        """
        Forecast the density of the target on the train or test set
        :param dsf: wrapperPandasCsv object
        :param x: vector of features
        :param y: vector of target observations
        :param states0: the central state for the model
        :param on_test: If the forecast should be done on the test set
        :return: (States particles, variables affecting the particles,
                  log-likelihood, log-prior, regularization term)
        """
        if dsf is not None:
            x, y, states0 = dsf.select(
                [self.x_col, self.y_col, self.states0_line_col], train=not on_test, test=on_test)
        elif (x is None) | (y is None) | (states0 is None):
            raise Exception("Must provide all the inputs or a dataframe wrapper")
        llkl, lprior, ldensity, mu, vcv, variables = self.posterior(x, y, states0)
        states = {'mu': mu, 'vcv': vcv}
        return states, variables, llkl, lprior, ldensity

    def density_forecast_multi_particles(self,
                                         dsf=None,
                                         x=None, y=None, states0=None,
                                         on_test=True,
                                         n_particles=None):
        """The same as density_forecast, but uses multiple particles.
        Combine the forecast assigning equal probability to them with equal probability"""
        if n_particles is None:
            n_particles = self.n_particles_forecast
        return decorator_multi_particles_predict(self.density_forecast)(
            dsf=dsf, x=x, y=y, states0=states0, on_test=on_test, n_particles=n_particles)


class CholeskyMLP(AbstractCholeskyForecaster):
    """Represent a Cholesky forecaster with an MLP network"""

    def __init__(self, dsf, x_col, y_col, states0_line_col,
                 hidden_layers_dim=[10, 10], gaussian_posterior=False,
                 empirical_prior=True,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0_line = dsf.select([x_col, y_col, states0_line_col], train=True, test=False)
        n = y.shape[1]
        d = states0_line.shape[1]
        assert d == int(n * (n + 1) / 2 + n)

        # Creates the network for the estimates
        if gaussian_posterior:
            linear_projector = S.LinearProjectorGaussianPosterior
            n_particles_forecast = 300
            jacobian_penalty = True
        else:
            linear_projector = S.LinearProjector
            n_particles_forecast = 1
            jacobian_penalty = False
        mlp, layers = S.MLP(x.shape[1], hidden_layers_dim, states0_line.shape[1],
                            init_scale=init_scale, linear_projector=linear_projector)
        optimizer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        if empirical_prior:
            prior = M.empirical_tvp_prior(states0_line, scale=3.0)
        else:
            prior = M.no_prior()

        # Build the whole learner
        self.shaper = mlp
        AbstractCholeskyForecaster.__init__(self, mlp, prior, optimizer, x_col, y_col, states0_line_col,
                                            jacobian_penalty=jacobian_penalty,
                                            n_particles_forecast=n_particles_forecast)


class CholeskyAutoEncoder(AbstractCholeskyForecaster):
    """Represent a Cholesky forecaster with an Autoencoder+MLP network"""

    def __init__(self, dsf, x_col, y_col, states0_line_col,
                 encoder_layers_dim=[10, 10], encode_dim=5, decoder_layers_dim=[10, 10],
                 forecaster_layers_dim=[10, 10], variational_reparametrization=False,
                 empirical_prior=True, PCA_center=False,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0_line = dsf.select([x_col, y_col, states0_line_col], train=True, test=False)
        n = y.shape[1]
        d = states0_line.shape[1]
        assert d == int(n * (n + 1) / 2 + n)

        # Creates the networks for the estimates:
        # 1- The Autoencoder
        if variational_reparametrization:
            auto_encoder_class = S.VariationalAutoEncoder
            linear_projector = S.LinearProjectorGaussianPosterior
            n_particles_forecast = 300
            jacobian_penalty = True
        else:
            auto_encoder_class = S.AutoEncoder
            linear_projector = S.LinearProjector
            n_particles_forecast = 1
            jacobian_penalty = False
        if PCA_center:
            pca = PCA(n_components=encode_dim)
            pca.fit(x, x)
            encoder_center = pca.transform
            decoder_center = pca.inverse_transform
        else:
            encoder_center = None
            decoder_center = None
        AE = auto_encoder_class(x.shape[1], encoder_layers_dim, encode_dim, decoder_layers_dim,
                                encoder_center=encoder_center, decoder_center=decoder_center)
        # 2- The forecaster
        mlp, _ = S.MLP(encode_dim, forecaster_layers_dim, states0_line.shape[1], init_scale=init_scale,
                       linear_projector=linear_projector)

        optimizer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        if empirical_prior:
            prior = M.empirical_tvp_prior(states0_line, scale=3.0)
        else:
            prior = M.no_prior()
        # 3- Composition of both
        shaper = S.compose(AE, mlp)
        self.shaper = shaper

        AbstractCholeskyForecaster.__init__(self, shaper, prior, optimizer, x_col, y_col, states0_line_col,
                                            jacobian_penalty=jacobian_penalty,
                                            n_particles_forecast=n_particles_forecast)


class CholeskyLSTM(AbstractCholeskyForecaster):
    """Represent a Cholesky forecaster with an LSTM+MLP network"""

    def __init__(self, dsf, x_col, y_col, states0_line_col,
                 recurrent_dim=5, mlp_post_lstm_layers_dim=[10, 10],
                 empirical_prior=True, gaussian_posterior=False,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0_line = dsf.select([x_col, y_col, states0_line_col], train=True, test=False)
        n = y.shape[1]
        d = states0_line.shape[1]
        assert d == int(n * (n + 1) / 2 + n)

        # Creates the networks for the estimates:
        lstm = S.LSTM(x.shape[1], recurrent_dim, mlp_post_lstm_layers_dim, d,
                      gaussian_posterior=gaussian_posterior, init_scale=init_scale)
        shaper = lstm

        optimizer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)

        if empirical_prior:
            prior = M.empirical_tvp_prior(states0_line, scale=3.0)
        else:
            prior = M.no_prior()

        self.shaper = shaper
        if gaussian_posterior:
            n_particles_forecast = 300
            jacobian_penalty = True
        else:
            n_particles_forecast = 1
            jacobian_penalty = False
        AbstractCholeskyForecaster.__init__(self, shaper, prior, optimizer, x_col, y_col, states0_line_col,
                                            jacobian_penalty=jacobian_penalty,
                                            n_particles_forecast=n_particles_forecast)


class HVIstaticMuVcv(AbstractCholeskyForecaster):
    """This class represent a VI estimator of the mean-covariance matrix,
    obtained using HVI on an initial linearized estimate of a mean-covariance matrix"""

    def __init__(self, dsf, y_col, states0_line_col,
                 hvi_layers_num=3,
                 empirical_prior=True,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        y, states0_line = dsf.select([y_col, states0_line_col], train=True, test=False)
        n = y.shape[1]
        d = states0_line.shape[1]
        assert d == int(n * (n + 1) / 2 + n)

        # Creates the networks for the estimates:
        sampler = hierarchical_sampler(d, layers=hvi_layers_num, init_scale=init_scale)

        def shaper(x, regularization_penalty=0.0, variables=[]):
            y, logJ, v = sampler.sample(1)
            regularization_penalty -= logJ
            variables += v
            return y, regularization_penalty, variables

        optimizer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)

        if empirical_prior:
            prior = M.empirical_tvp_prior(states0_line, scale=3.0)
        else:
            prior = M.no_prior()

        # 3- Composition of both
        n_particles_forecast = 100
        jacobian_penalty = True

        AbstractCholeskyForecaster.__init__(self, shaper, prior, optimizer, y_col, y_col, states0_line_col,
                                            jacobian_penalty=jacobian_penalty,
                                            n_particles_forecast=n_particles_forecast)


if __name__ == '__main__':
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
    # dsf.withColumn('dcc', *T.dcc(*dsf['y']))
    dsf.withColumn('crosses', *T.elements_cross(*dsf['y']))
    dsf.withColumn('x_timesteps', *T.lagify(*dsf['y'], collapse=False))
    dsf.withColumn('x', *T.lagify(*dsf['y'], collapse=True))
    dsf.withColumn('lw', *T.staticLedoitWolf(*dsf['y']))
    mu_vcv_line_col = 'mu_vcv_line'
    baseline_mu_vcv = 'lw'
    states_line = [M.mu_vcv_linearize(**states) for states in dsf[baseline_mu_vcv]]
    dsf.withColumn(mu_vcv_line_col, *states_line)

    dsf.dropna()

    # Learners testing

    # bayesian mlp
    m = HVIstaticMuVcv(dsf, 'x', 'y', mu_vcv_line_col, init_scale=1e-8, learning_rate=1e-6)
    m.fit(dsf, 1, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)
    print(llkl, p0, log_p)

    import pdb

    pdb.set_trace()

    # bayesian mlp
    m = CholeskyMLP(dsf, 'x', 'y', mu_vcv_line_col, gaussian_posterior=True, init_scale=1e-6, learning_rate=1e-8)
    m.fit(dsf, 1, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)
    print(llkl, p0, log_p)

    # lstm
    m = CholeskyLSTM(dsf, 'x_timesteps', 'y', mu_vcv_line_col)
    m.fit(dsf, 10, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)

    # autoencoder
    m = CholeskyAutoEncoder(dsf, 'x', 'y', mu_vcv_line_col)
    m.fit(dsf, 10, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)

    # mlp
    m = CholeskyMLP(dsf, 'x', 'y', mu_vcv_line_col)
    m.fit(dsf, 10, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)
