from econometrics_problem import DatasetFactory
import timeseries_transformations as T
import gaussian_multivariate_model as G
import optimizers as O
from evaluators import evaluators_timeseries
from datetime import datetime
from utils import format_number, asymmetric_mix_dict
import shapers as S
from variational_inference import HVI
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from autoregressive_model import DCC
from variational_inference import hierarchical_sampler


class naive_sampler_global:
    def __init__(self, n):
        self.n = n

    def sample(self, mc=1):
        n = self.n
        x = np.zeros([mc, n], dtype=np.float32)
        return x, tf.constant(x[:, 0]), []


def decorator_multi_particles_predict(predict):
    def multi_particles_predict(n_particles, **kwargs):
        llkls_all = []
        p0_all = []
        log_p_all = []
        states_all = []
        theta_all = []
        for i in range(n_particles):
            states, theta, llkl, p0, log_p = predict(**kwargs)
            llkls_all.append(llkl)
            p0_all.append(p0)
            log_p_all.append(log_p)
            states_all.append(states)
            theta_all.append(theta)
        log_n_particles = np.log(n_particles)
        llkls = tf.reduce_logsumexp(np.stack(llkls_all, axis=0), axis=0) - log_n_particles
        p0s = tf.reduce_logsumexp(np.stack(p0_all, axis=0), axis=0) - log_n_particles
        log_ps = tf.reduce_logsumexp(np.stack(log_p_all, axis=0), axis=0) - log_n_particles
        return states, theta, llkls, p0s, log_ps

    return multi_particles_predict


class WrapperAbstract:
    def __init__(self, model, shaper, inferer, x_col, y_col, states0_col, sampler_global):
        self.model = model
        self.shaper_local = shaper
        self.inferer = inferer
        self.x_col = x_col
        self.y_col = y_col
        self.states0_col = states0_col
        self.sampler_global = sampler_global
        self.n_particles_forecast = 100

    def fit(self, dsf, epochs=1000, batch_size=16, verbose=0):
        elbos = []
        for i in range(epochs):
            elbos.append(self.fit_step(dsf, batch_size, verbose=verbose))
        return elbos

    def fit_step(self, dsf, batch_size, verbose=0):
        # Read the batch data to fit
        batch, index_batch, times_batch = dsf.get_batch(
            [self.x_col], self.y_col, self.states0_col, size=batch_size)
        x_t, y_t, states0 = batch
        with tf.GradientTape() as t:
            # Computes the model posterior
            particles_global, log_p_global, variables_global = self.sampler_global.sample()
            particles_t, log_p, variables_shaper = self.shaper_local(*x_t, particles_global=particles_global)
            states, theta, llkl, p0, log_jacobian, variables_model = self.model.model_evaluation(
                y_t, particles_t, particles_global, **states0)
            p1 = llkl - log_p - log_p_global
            elbo = tf.reduce_mean(p1)
            _elbo = format_number(elbo.numpy())
            if verbose > 0:
                print(f'elbo: {_elbo}, llkl: {np.mean(llkl.numpy())}, log_p: {np.mean(log_p.numpy())}')
            variables = variables_shaper + variables_model + variables_global
            g = t.gradient(-elbo, variables)
        # optimize
        self.inferer.optimize(g, variables)
        return _elbo

    def density_forecast(self,
                         dsf=None,
                         x=None, y=None, states0=None,
                         on_test=True):
        if dsf is not None:
            x, y, states0 = dsf.select(
                [self.x_col, self.y_col, self.states0_col], train=not on_test, test=on_test)
        elif (x is None) | (y is None) | (states0 is None):
            raise Exception("Must provide all the inputs or a dataframe wrapper")
        particles_global, log_p_global, variables_global = self.sampler_global.sample()
        particles_t, log_p, variables = self.shaper_local(x)
        states, theta, llkl, p0, _, _ = self.model.model_evaluation(
            y, particles_t, particles_global, **states0
        )
        log_p = log_p + log_p_global
        return states, theta, llkl, p0, log_p

    def density_forecast_multi_particles(self,
                                         dsf=None,
                                         x=None, y=None, states0=None,
                                         on_test=True,
                                         n_particles=None):
        if n_particles is None:
            n_particles = self.n_particles_forecast
        return decorator_multi_particles_predict(self.density_forecast)(
            dsf=dsf, x=x, y=y, states0=states0, on_test=on_test, n_particles=n_particles)


class CholeskyVariationalAutoencoder(WrapperAbstract):
    def __init__(self, dsf, x_col, y_col, states0_col, encoding_dim=5,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0 = dsf.select([x_col, y_col, states0_col], train=True, test=False)
        n = y.shape[1]
        d = int(n * (n + 1) / 2 + n)
        model = G.gaussian_timeseries_centered_model(states0)
        shaper = S.transformer_variational_autoencoder(
            example=x, encoding_dim=encoding_dim, out_dim=d, init_scale=init_scale)
        inferer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        sampler_global = naive_sampler_global(model.global_dim)
        self.n_particles_forecast = 200
        WrapperAbstract.__init__(self, model, shaper, inferer, x_col, y_col, states0_col, sampler_global)


class CholeskyAutoencoder(WrapperAbstract):
    def __init__(self, dsf, x_col, y_col, states0_col, encoding_dim=5,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0 = dsf.select([x_col, y_col, states0_col], train=True, test=False)
        n = y.shape[1]
        d = int(n * (n + 1) / 2 + n)
        model = G.gaussian_timeseries_centered_model(states0)
        shaper = S.transformer_autoencoder(
            example=x, encoding_dim=encoding_dim, out_dim=d, init_scale=init_scale)
        inferer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        sampler_global = naive_sampler_global(model.global_dim)
        self.n_particles_forecast = 1
        WrapperAbstract.__init__(self, model, shaper, inferer, x_col, y_col, states0_col, sampler_global)


class CholeskyLSTM(WrapperAbstract):
    def __init__(self, dsf, x_col, y_col, states0_col,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0 = dsf.select([x_col, y_col, states0_col], train=True, test=False)
        n = y.shape[1]
        d = int(n * (n + 1) / 2 + n)
        model = G.gaussian_timeseries_centered_model(states0)
        shaper = S.transformer_lstm(example=x, out_dim=d, init_scale=init_scale)
        inferer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        sampler_global = naive_sampler_global(model.global_dim)
        self.n_particles_forecast = 1
        WrapperAbstract.__init__(self, model, shaper, inferer, x_col, y_col, states0_col, sampler_global)


class CholeskyMLP(WrapperAbstract):
    def __init__(self, dsf, x_col, y_col, states0_col,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0 = dsf.select([x_col, y_col, states0_col], train=True, test=False)
        n = y.shape[1]
        d = int(n * (n + 1) / 2 + n)
        model = G.gaussian_timeseries_centered_model(states0)
        shaper = S.transformer_mlp(example=x, out_dim=d, init_scale=init_scale)
        inferer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        sampler_global = naive_sampler_global(model.global_dim)
        self.n_particles_forecast = 1
        WrapperAbstract.__init__(self, model, shaper, inferer, x_col, y_col, states0_col, sampler_global)


class CholeskyMLPBayesian(WrapperAbstract):
    def __init__(self, dsf, x_col, y_col, states0_col,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0 = dsf.select([x_col, y_col, states0_col], train=True, test=False)
        n = y.shape[1]
        d = int(n * (n + 1) / 2 + n)
        model = G.gaussian_timeseries_centered_model(states0)
        shaper = S.transformer_mlp_Bayesian(example=x, out_dim=d, init_scale=init_scale)
        inferer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        sampler_global = naive_sampler_global(model.global_dim)
        self.n_particles_forecast = 200
        WrapperAbstract.__init__(self, model, shaper, inferer, x_col, y_col, states0_col, sampler_global)


hvi_config = {
    "init_scale": 0.1,
    "epsilon_config": {},
    "layers_config": [
        {
            "kind": "affine"
        },
        {
            "kind": "affine"
        },
        {
            "kind": "affine"
        }
    ]
}


class DCC_HVI:
    def __init__(self, x_col, y_col, n, p=1, q=1, learning_rate=1e-2, beta_1=0.9, **sampler_config):
        sampler_config = asymmetric_mix_dict(hvi_config, sampler_config)
        self.model = DCC(n=n, p=p, q=q)
        self.sampler_global = hierarchical_sampler(self.model.dim_all, **sampler_config)
        self.inferer = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        self.x_col = x_col
        self.y_col = y_col
        self.p = p
        self.q = q
        self.n_particles_forecast = 200

    def fit(self, dsf, epochs=1000, batch_size=16, verbose=0):
        if self.model.pars_dict is None:
            self.model.infer_parameters0(*dsf[self.y_col])
        elbos = []
        for i in range(epochs):
            elbos.append(self.fit_step(dsf, batch_size, verbose=verbose))
        return elbos

    def fit_step(self, dsf, batch_size, verbose=0):
        # Read the batch data to fit
        batch, index_batch, times_batch = dsf.get_batch(self.x_col, self.y_col, size=batch_size)
        x, y = batch
        with tf.GradientTape() as t:
            # Computes the model posterior
            particles_global, log_p, variables_sampler = self.sampler_global.sample(batch_size)
            states, theta, llkl, p0, log_jacobian, variables_model = self.model.model_evaluation(
                y, particles=particles_global, particles_t=x)
            p1 = llkl + p0 - log_p
            elbo = tf.reduce_mean(p1)
            _elbo = format_number(elbo.numpy())
            if verbose > 0:
                print(f'elbo: {_elbo}')
            variables = variables_sampler + variables_model
            g = t.gradient(-elbo, variables)
            # g = [tf.clip_by_value(x, -1e4, 1e4) for x in g]
            # optimize
        self.inferer.optimize(g, variables)
        return _elbo

    def density_forecast(self,
                         dsf=None,
                         x=None, y=None,
                         on_test=True):
        if dsf is not None:
            x, y = dsf.select(
                [self.x_col, self.y_col], train=not on_test, test=on_test)
        elif (x is None) | (y is None):
            raise Exception("Must provide all the inputs or a dataframe wrapper")
        particles_global, log_p, variables_sampler = self.sampler_global.sample(1)
        particles_global = tf.repeat(particles_global, y.shape[0], axis=0)
        states, theta, llkl, p0, log_jacobian, variables = self.model.model_evaluation(
            y, particles_t=x, particles=particles_global
        )
        return states, theta, llkl, p0, log_p - log_jacobian

    def density_forecast_multi_particles(self,
                                         dsf=None,
                                         x=None, y=None,
                                         on_test=True,
                                         n_particles=None):
        if n_particles is None:
            n_particles = self.n_particles_forecast
        return decorator_multi_particles_predict(self.density_forecast)(
            dsf=dsf, x=x, y=y, on_test=on_test, n_particles=n_particles)


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
    dsf.withColumn('ledoitWolf_static', *T.staticLedoitWolf(*dsf['y']))

    dsf.dropna()

    # concrete wrappers testing

    ##Bayesian mlp
    m = CholeskyMLPBayesian(dsf, 'x', 'y', 'ledoitWolf_static')
    m.fit(dsf, 10, 8)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)

    # DCC
    m = DCC_HVI('x_timesteps', 'y', n=dsf.df.shape[1])
    m.fit(dsf, 10, 4, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf=dsf)

    ## variational autoencoder
    m = CholeskyVariationalAutoencoder(dsf, 'x', 'y', 'ledoitWolf_static')
    m.fit(dsf, 10, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)

    # autoencoder
    m = CholeskyAutoencoder(dsf, 'x', 'y', 'ledoitWolf_static')
    m.fit(dsf, 10, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)

    # lstm
    m = CholeskyLSTM(dsf, 'x_timesteps', 'y', 'ledoitWolf_static')
    m.fit(dsf, 10, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)

    # mlp
    m = CholeskyMLP(dsf, 'x_timesteps', 'y', 'ledoitWolf_static')
    m.fit(dsf, 10, 16)
    states, theta, llkl, p0, log_p = m.density_forecast(dsf)
    m.density_forecast_multi_particles(dsf)
