import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.metrics import mean_absolute_error as mae
from evaluators import tester_performances
import matplotlib.pyplot as plt


def sample_mixture(N, mixture):
    n_component = len(mixture)
    truth = []
    ni = np.round(N * np.random.dirichlet(N * np.ones(n_component)))
    for i in range(n_component):
        n = ni[i]
        mix = mixture[i]
        truth.append(mix.sample(n).numpy())
    truth = np.concatenate(truth, axis=0)

    return truth


def naive_shaper(particles):
    states = {'x': particles}
    jacobian_particles = np.zeros(particles.shape[:-1])
    return states, jacobian_particles, []


class bidimensional_scatter_qualitative(tester_performances):
    def evaluate(self, particles, **kwargs):
        pass

    def callback(self, theta):
        n_obsvs = 100
        x = theta['particles']['x']
        tr = self.truth['x']
        mc = np.random.choice(np.arange(tr.shape[0]), size=np.min([x.shape[0], tr.shape[0]]), replace=False)
        mc = np.random.choice(np.arange(tr.shape[0]), size=n_obsvs, replace=False)
        tr = tr[mc]
        plt.scatter(x[-n_obsvs:, 0], x[-n_obsvs:, 1])
        plt.scatter(tr[:, 0], tr[:, 1])
        plt.show()

    def get_pandas(self):
        return None


class mixture_problem:
    def __init__(self, N, **kwargs):
        d = N
        scale = 3
        m1 = np.float32(np.ones(d))
        v1 = np.float32(np.eye(d)) / scale
        target_Gaussian1 = tfp.distributions.MultivariateNormalFullCovariance(m1, v1)
        m2 = np.float32(np.random.randn(d)) / scale * 2
        v2 = np.float32(np.eye(d)) / scale
        target_Gaussian2 = tfp.distributions.MultivariateNormalFullCovariance(m2, v2)
        mixture = [target_Gaussian1, target_Gaussian2]

        def lkl_func(y, x):
            ret = 0.0
            for target in mixture:
                ret += target.prob(x)
            return tf.math.log(ret)

        def prior_func(x):
            return tf.constant(np.zeros(x.shape[:-1], dtype=np.float32))

        truth = {'x': sample_mixture(1000, mixture)}
        bayes_factor = np.log(len(mixture))

        self.mixture = mixture
        self.d = d
        self.parameters_shaper = naive_shaper
        self.y = np.random.rand(1, d)
        self.truth = truth
        self.bayes_factor = bayes_factor
        self.prior_func = prior_func
        self.lkl_func = lkl_func

    def get_evaluator(self):
        return bidimensional_scatter_qualitative(self.truth, mae)

    def plot_results(self, evaluator):
        pass
