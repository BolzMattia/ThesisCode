import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from shapers import hierarchical_sampler
import matplotlib.pyplot as plt
from utils import dict_tf2numpy, concat_dict
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import dict_tf2numpy, concat_dict


class HVI:
    """Represent the combination of a Hierarchical sampler with an SGD optimization procedure
        to conduct general optimization of the semi-parametric sampler."""
    def __init__(self,
                 d: int,
                 sampler_config: dict,
                 optim_config: dict,
                 parameters_shaper):
        sampler = hierarchical_sampler(d, **sampler_config)
        optimizer = tf.keras.optimizers.Adam(**optim_config)
        self.sampler = sampler
        self.optimizer = optimizer
        self.parameters_shaper = parameters_shaper

    def sample(self, MC=None):
        """
        Sample particles and compute their log-density
        :param MC: How many particles
        :return: (tensor (dim=MC,d) , tensor (dim=MC,1), list of variables)
        """
        if MC is None:
            theta0, log_p, variables = self.sampler.sample(1)
            theta0, log_p = theta0[0], log_p[0]
        else:
            theta0, log_p, variables = self.sampler.sample(MC)
        theta, log_J_shaping, _ = self.parameters_shaper(theta0)
        log_p = log_p - log_J_shaping
        return theta, log_p, variables

    def estimate_elbo(self, y, theta, lkl_func, prior_func, log_p):
        """ Evaluate the ELBO expression on a set of particles"""
        # model
        lkl = lkl_func(y, **theta)
        p0 = prior_func(**theta)
        # VI
        elbo = tf.reduce_mean(lkl + p0 - log_p)
        return elbo, lkl, p0

    def fit_step(self, MC, y, lkl_func, prior_func):
        """A single SGD optimization step on a given pair of log-lkl and log-prior functions"""
        with tf.GradientTape() as t:
            theta, log_p, variables = self.sample(MC)
            elbo, lkl, p0 = self.estimate_elbo(y, theta, lkl_func, prior_func, log_p)
            # gradient
            g = t.gradient(-elbo, variables)
        # g = [tf.clip_by_value(x, -1e4, 1e4) for x in g]
        # optimize
        self.optimizer.apply_gradients(zip(g, variables))
        return theta, lkl, p0, log_p, elbo


def inferenceMethod_HVI(d: int,
                        sampler_config: dict,
                        epochs: int,
                        optim_config: dict,
                        parameters_shaper,
                        y: np.array,
                        lkl_func,
                        prior_func,
                        MC: int,
                        bayes_factor,
                        save_path=None):
    """
    Wrapper of an entire inference process with HVI on a given abstract problem.
    :param d: The problem dimension
    :param sampler_config: A dictionary with the hierarchical sampler config
    :param epochs: How many train steps
    :param optim_config: A dictionary with the SGD optimizer config
    :param parameters_shaper: A function shaping R^d into the parameters space.
    :param y: The data
    :param lkl_func: The log-likelihood function
    :param prior_func: The log-prior function
    :param MC: How many particles to sample for a single SGD step (batch_size)
    :param bayes_factor: If known, the exact Bayes factor to plot it compared to ELBO
    :param save_path: The str path where to save the results
    :return: The particles produced during the inference
    """
    elbos = []
    hvi_particles = None
    hvi = HVI(d, sampler_config=sampler_config, optim_config=optim_config, parameters_shaper=parameters_shaper)

    for i in range(epochs):
        theta, lkl, p0, log_p, elbo = hvi.fit_step(MC, y, lkl_func, prior_func)

        if (i % 100) == 0:
            print(f'epoch {i}: elbo {elbo.numpy()}, E[lkl] {np.mean(lkl.numpy())}, E[p0] {np.mean(p0.numpy())}')
        elbos.append(elbo.numpy())
        # evaluation
        theta_np = dict_tf2numpy(theta)

        hvi_particles = concat_dict(hvi_particles, theta_np)
        # hvi_particles_history.append(hvi_particles)

    # Diagnostic plot
    plt.clf()
    fig = plt.figure(figsize=[10, 5])
    plt.plot(elbos)
    plt.axhline(bayes_factor, color='r')
    if save_path is not None:
        fig.savefig(save_path)

    return hvi_particles


class inference_optimizer_sgd:
    """DEPRECATED"""

    def __init__(self, optim_config: dict):
        optimizer = tf.keras.optimizers.Adam(**optim_config)
        self.optimizer = optimizer

    def sample(self, MC=None):
        if MC is None:
            theta0, log_p, variables = self.sampler.sample(1)
            theta0, log_p = theta0[0], log_p[0]
        else:
            theta0, log_p, variables = self.sampler.sample(MC)
        theta, log_J_shaping, _ = self.parameters_shaper(theta0)
        log_p = log_p - log_J_shaping
        return theta, log_p, variables

    def estimate_elbo(self, y, theta, lkl_func, prior_func, log_p):
        # model
        lkl = lkl_func(y, **theta)
        p0 = prior_func(**theta)
        # VI
        elbo = tf.reduce_mean(lkl + p0 - log_p)
        return elbo, lkl, p0

    def fit_step(self, MC, y, lkl_func, prior_func):
        with tf.GradientTape() as t:
            theta, log_p, variables = self.sample(MC)
            elbo, lkl, p0 = self.estimate_elbo(y, theta, lkl_func, prior_func, log_p)
            # gradient
            g = t.gradient(-elbo, variables)
        # g = [tf.clip_by_value(x, -1e4, 1e4) for x in g]
        # optimize
        self.optimizer.apply_gradients(zip(g, variables))
        return theta, lkl, p0, log_p, elbo


if __name__ == '__main__':
    d = 5
    optim_config = {
        "learning_rate": 0.001,
        "beta_1": 0.9
    }
    sampler_config = {
        "init_scale": 1.0,
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
    vi = HVI(d, sampler_config=sampler_config, optim_config=optim_config, parameters_shaper=None)
    print(vi)
