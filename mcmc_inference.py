import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from utils import dict_tf2numpy


class MCMC_diagnostic:
    """Collects the results of an MCMC run in tensorflow,
    for easy access of its internal state"""
    def __init__(self, kernel_results):
        self.kr = kernel_results

    def summary(self):
        """Print the cumulative acceptance-ratio"""
        is_accepted = self.kr.is_accepted.numpy().astype(int)
        print(f'Cumulative acceptance-ratio: {is_accepted.mean()}')

    def plot(self, save_path=None):
        """Plot the cumulative acceptance-ratio"""
        _kr = self.kr
        is_accepted = _kr.is_accepted.numpy().astype(int)

        ar = (np.cumsum(is_accepted, axis=0) / np.tile(
            1 + np.arange(is_accepted.shape[0]).reshape([-1, 1]),
            [1, 1]))
        plt.clf()
        plt.plot(ar[10:])  # exclude the fi  rst ten samples of the cumulative mean
        plt.xlabel('particles sampled')
        plt.ylabel('cumulative acceptance-ratio')
        plt.title('MCMC trace-plot')
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()


class MCMC:
    """Wrapper of a tensorflow MCMC object."""
    def __init__(self,
                 num_chains=3,
                 num_results=10,
                 num_burnin_steps=2,
                 num_steps_between_results=1,
                 step_size=1e-3,
                 num_leapfrog_steps=3,
                 choice='hamiltonian'):
        self.num_chains = num_chains
        self.num_results = num_results
        self.num_burnin_steps = num_burnin_steps
        self.num_steps_between_results = num_steps_between_results
        self.step_size = step_size
        self.num_leapfrog_steps = num_leapfrog_steps
        self.choice = choice

    def compile_inference(self, d, lkl, p0, y, theta_shaper):
        """Prepare the posterior numerator function and the sampling MCMC mechanism."""
        def Lp0_fun(x):
            states, log_p, _ = theta_shaper(x)
            ll = lkl(y, **states)
            prior = p0(**states)
            return ll + prior - log_p

        if self.choice == 'hamiltonian':
            mcmc_choice = tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=Lp0_fun,
                step_size=self.step_size,
                num_leapfrog_steps=self.num_leapfrog_steps)
        elif self.choice == 'uniform':
            mcmc_choice = tfp.mcmc.random_walk_uniform_fn(
                scale=self.step_size, name=None
            )
        else:
            raise NotImplementedError(f'MCMC methodology not implemented: {self.choice}')

        init_state = np.zeros([self.num_chains, d], dtype=np.float32)

        samples_unshaped, kernel_results = tfp.mcmc.sample_chain(
            num_results=self.num_results,
            current_state=init_state,
            kernel=mcmc_choice,
            num_burnin_steps=self.num_burnin_steps,
            num_steps_between_results=self.num_steps_between_results,
        )

        samples_unshaped = tf.reshape(samples_unshaped, [self.num_chains * self.num_results, -1])
        particles = theta_shaper(samples_unshaped)

        self.particles = particles
        self.kernel_results = kernel_results


def inferenceMethod_MCMC(d, lkl_func, prior_func, y, parameters_shaper, save_path=None, bayes_factor=None, **config):
    """Effectively run the MCMC inference, launching the sampling scheme."""
    mcmc = MCMC(**config)
    # MCMC sampling
    mcmc.compile_inference(d, lkl_func, prior_func, y, parameters_shaper)
    mcmc_particles = dict_tf2numpy(mcmc.particles[0])
    # diagnostic
    MCMC_diagnostic(mcmc.kernel_results).plot(save_path)
    return mcmc_particles
