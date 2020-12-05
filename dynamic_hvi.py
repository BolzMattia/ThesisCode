import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import tensorflow_probability as tfp
from sklearn.covariance import LedoitWolf
import optimizers as O
import matplotlib.pyplot as plt
from wrappers import decorator_multi_particles_predict
from hvi import PReluActivation


def mlp(out_dim):
    """A single multilinear projector with dim=100. Returns a Keras.layers.Sequential instance."""
    dim = 100
    net = Sequential([
        Dense(dim, activation='selu'),
        Dense(dim, activation='selu'),
        Dense(out_dim),
    ])
    return net


def shape_A_b(x, n, center_A, scale_A, scale_b):
    """Takes a tensor x with dimension [batch, n_x] and returns:
        - a tensor, A, with dimension [batch, n*n], scaled by scale_A and centered on center_A
        - a tensor, b, with dimension [batch, 1, n], scaled by scale_b
    """
    return tf.reshape(x[:, n:], [x.shape[0], n, n]) * scale_A + center_A, x[:, tf.newaxis, :n] * scale_b


def shape_A_b_MC(x, n, center_A, scale_A, scale_b):
    """Takes a tensor x with dimension [batch, mc, n_x] and returns:
        - a tensor, A, with dimension [batch, mc, n*n], scaled by scale_A and centered on center_A
        - a tensor, b, with dimension [batch, mc, 1, n], scaled by scale_b
    """
    return tf.reshape(x[:, :, n:], [x.shape[0], x.shape[1], n, n]) * scale_A + center_A, \
           x[:, :, tf.newaxis, :n] * scale_b


#The PRelu activation function, unique instance for the module.
_prelu_activation = PReluActivation(0.2)


def proj(y, A, b):
    """
    Apply an Affine projection to y, returning y@A + b
    :param y: The input of the affine transformation, with dim=k,n,1
    :param A: The affine linear projection, with dim=1,d,n
    :param b: The translation of the affine transformation, with dim=1,d
    :return: A tensor with dim=k,d
    """
    transformed = tf.matmul(y, A) + b
    #transformed = _prelu_activation(transformed)
    log_jacobian = tf.math.log(tf.math.abs(tf.linalg.det(A)))
    #log_jacobian += tf.reduce_sum(_prelu_activation.jacobian_vector(transformed), axis=[-2, -1])
    return transformed, log_jacobian


# Standard gaussian initialized just once
_pdf_standard_gaussian = tfp.distributions.Normal(0.0, 1.0)


def prior_standard_gaussian(*args):
    """Compute a standard Gaussian log-density on all the elements in input.
    Returns the sum of all the log-densities."""
    p = [tf.reduce_sum(_pdf_standard_gaussian.log_prob(x)) for x in args]
    p = tf.reduce_sum(tf.stack(p))
    return p


class HDE_mu_vcv:
    """
    NOT YET TESTED. Shouldn't be used.
    """
    def __init__(self, dsf, x_col, y_col, states0_col, layers=1,
                 init_scale=1e-2, learning_rate=1e-4, beta_1=0.1):
        x, y, states0 = dsf.select([x_col, y_col, states0_col], train=True, test=False)
        n = y.shape[1]
        self.n = n
        self.x_col = x_col
        self.y_col = y_col
        self.states0_col = states0_col
        self.center_A = np.eye(n, dtype=np.float32)[np.newaxis]
        self.scale_A = init_scale
        self.scale_b = init_scale
        self.x2theta = [mlp(n * n + n) for i in range(layers)]
        self.log_prior = prior_standard_gaussian
        self.optim = O.optimizer_adam(learning_rate=learning_rate, beta_1=beta_1)
        self.y_memory = y
        self.x_memory = x
        self.states_memory = states0
        self.n_particles_forecast = 1

    def shape_layer(self, line_layer):
        return shape_A_b(line_layer, self.n, self.center_A, self.scale_A, self.scale_b)

    def shape_layer_MC(self, line_layer):
        return shape_A_b_MC(line_layer, self.n, self.center_A, self.scale_A, self.scale_b)

    def fit(self, dsf, epochs=1000, batch_size=16, verbose=0):
        elbos = []
        for i in range(epochs):
            elbos.append(self.fit_step(dsf, batch_size, verbose=verbose))
        return elbos

    def fit_step(self, dsf, batch_size, verbose=0):
        batch, index_batch, times_batch = dsf.get_batch(
            self.x_col, self.y_col, self.states0_col, size=batch_size)
        x_t, y_t, states0 = batch
        # The center of the distribution
        pdf = tfp.distributions.MultivariateNormalFullCovariance(states0['mu'], states0['vcv'])
        with tf.GradientTape() as t:
            lkl, p0, variables = self.posterior(y_t, x_t, pdf)
            elbo = tf.reduce_mean(lkl) + p0
            # If required, print the Elbo
            if verbose > 0:
                print(f'elbo: {elbo.numpy()}')
            g = t.gradient(-elbo, variables)
        # SGD on parameters
        self.optim.optimize(g, variables)
        return lkl

    def parameters(self):
        variables = []
        for l in self.x2theta:
            variables += l.weights
        return variables

    def log_likelihood_MC(self, y, x, pdf):
        x = x[tf.newaxis, :, :]
        y_hat = y[:, tf.newaxis, tf.newaxis, :]
        log_j = tf.zeros([y_hat.shape[0], x.shape[1]], dtype=tf.float32)
        for l in self.x2theta:
            y_hat, lj = proj(y_hat, *self.shape_layer_MC(l(x)))
            log_j += lj
        lkl = pdf.log_prob(y_hat[:, :, 0, :]) + log_j

        return lkl

    def log_likelihood(self, y, x, pdf):
        y_hat = y[:, tf.newaxis, :]
        log_j = tf.zeros(y_hat.shape[0], dtype=tf.float32)
        for l in self.x2theta:
            y_hat, lj = proj(y_hat, *self.shape_layer(l(x)))
            log_j += lj
        lkl = pdf.log_prob(y_hat[:, 0, :]) + log_j
        return lkl

    def posterior(self, y, x, pdf):
        """Computes the model predictive distribution"""
        lkl = self.log_likelihood(y, x, pdf)
        variables = self.parameters()
        # Prior on variables <-> parameters
        p0 = self.log_prior(*variables)
        return lkl, p0, variables

    def density_forecast(self,
                         dsf=None,
                         x=None, y=None, states0=None,
                         on_test=True):
        if dsf is not None:
            x, y, states0 = dsf.select(
                [self.x_col, self.y_col, self.states0_col], train=not on_test, test=on_test)
        elif (x is None) | (y is None) | (states0 is None):
            raise Exception("Must provide all the inputs or a dataframe wrapper")
        pdf = tfp.distributions.MultivariateNormalFullCovariance(states0['mu'], states0['vcv'])
        lkl, p0, variables = self.posterior(y, x, pdf)
        mu, vcv = self.mu_vcv_memory(x, states0)
        states = {'mu': mu, 'vcv': vcv}
        # todo: implement Bayesian net with density
        log_p = tf.zeros(1, tf.float32)
        return states, variables, lkl, p0, log_p

    def mu_vcv_memory(self, x, states):
        pdf = tfp.distributions.MultivariateNormalFullCovariance(states['mu'][tf.newaxis], states['vcv'][tf.newaxis])
        # dim(llkls) = [t_Memory, x.shape[0]]
        llkls = self.log_likelihood_MC(self.y_memory, x, pdf)

        # Create a matrix with the cross-products of every y in memory
        matrix_y_memory = self.y_memory.reshape(*self.y_memory.shape, 1)
        cross_product_memory = matrix_y_memory @ (matrix_y_memory.transpose(0, 2, 1))

        # dim(lk) = [t_Memory, x.shape[0]]
        lk = tf.exp(llkls - tf.reduce_logsumexp(llkls, axis=0)[tf.newaxis, :])
        # dim(vcv) = [x.shape[0], y_memory.shape[0], y_memory.shape[0]]
        vcv = tf.reduce_sum(lk[:, :, tf.newaxis, tf.newaxis] * cross_product_memory[:, np.newaxis, :, :], axis=0)
        # dim(mu) = [x.shape[0], y_memory.shape[0]]
        mu = tf.reduce_sum(lk[:, :, tf.newaxis] * self.y_memory[:, np.newaxis, :], axis=0)
        return mu, vcv

    def density_forecast_multi_particles(self,
                                         dsf=None,
                                         x=None, y=None, states0=None,
                                         on_test=True,
                                         n_particles=None):
        if n_particles is None:
            n_particles = self.n_particles_forecast
        return decorator_multi_particles_predict(self.density_forecast)(
            dsf=dsf, x=x, y=y, states0=states0, on_test=on_test, n_particles=n_particles)


if __name__ == '__main__':
    from experiment_econometrics import DatasetFactory
    import timeseries_transformations as T

    x_col = 'x'
    y_col = 'y'
    states0_col = 'ledoitWolf_static'
    dsf = DatasetFactory.get(name='FRED_raw')
    # Train test split
    train_time = dsf.split_train_test(sequential=True, test_size=0.3)

    # Features computation
    dsf.withColumn('y', *T.standardize(*dsf['y']))
    dsf.withColumn('x', *T.lagify(*dsf['y'], lag=12, collapse=True))
    dsf.withColumn('ledoitWolf_static', *T.staticLedoitWolf(*dsf['y']))
    dsf.dropna()
    x, y, states0 = dsf.select([x_col, y_col, states0_col], train=True, test=False)
    x_test, y_test, states0_test = dsf.select([x_col, y_col, states0_col], train=False, test=True)
    model = HDE_mu_vcv(dsf, 'x', 'y', 'ledoitWolf_static', learning_rate=1e-1)
    llkls = []
    for i in range(10):
        print(i)
        pdf = tfp.distributions.MultivariateNormalFullCovariance(
            states0['mu'][:, tf.newaxis, :], states0['vcv'][:, tf.newaxis, :, :])
        model.log_likelihood_MC(y, x_test, pdf)
        states, theta, lkl, p0, log_p = model.density_forecast(dsf)
        llkls.append(np.mean(lkl.numpy()))
        plt.plot(model.mu_vcv_memory(x_test, states0_test)[0][:, 0])
        model.fit(dsf, epochs=10, batch_size=16, verbose=0)
    plt.show()
    plt.plot(llkls)
    plt.show()
