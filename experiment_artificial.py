from variational_inference import inferenceMethod_HVI
from mcmc_inference import inferenceMethod_MCMC
from gaussian_model_artificial import gaussian_problem
from mixture_problem import mixture_problem
import numpy as np
import os
from datetime import datetime
import json
from utils import equal_interval_creation


def problem_builder(kind, **config):
    """Access point for the problem sampler creation."""
    if kind == 'gaussian':
        problem = gaussian_problem(**config)
    elif kind == 'mixture':
        problem = mixture_problem(**config)
    else:
        raise Exception(f'kind of problem ({kind}) unknown')
    return problem


def experiment(
        experiment_dir,
        seed,
        problem,
        methods):
    """
    This method wraps all the operation necessary
    to evaluate posterior approximators and save the results.
    :param experiment_dir: The directory where to save the results, str
    :param seed: the seed of the random problem creation, int
    :param problem: A dictionary with the problem configurations
    :param methods: A dictionary with the approximators configurations
    :return: None
    """
    np.random.seed(seed)
    time_intervals = 100

    problem = problem_builder(**problem)
    evaluator = problem.get_evaluator()

    dts = {}
    all_particles = {}
    for method in methods:
        t1 = datetime.now()
        method_name = method['name']
        # Paths
        results_path = f'{experiment_dir}/results.csv'
        plot_inference_path = f'{experiment_dir}/trace_inference.png'
        method_diagnostic_path = f'{experiment_dir}/diagnostic_{method_name}.png'
        # Every methodology got its own function
        methodology = {
            'hvi': inferenceMethod_HVI,
            'mcmc': inferenceMethod_MCMC,
        }[method['kind']]
        all_particles[method_name] = methodology(
            d=problem.d,
            **method['config'],
            parameters_shaper=problem.parameters_shaper,
            y=problem.y,
            lkl_func=problem.lkl_func,
            prior_func=problem.prior_func,
            save_path=method_diagnostic_path,
            bayes_factor=problem.bayes_factor
        )
        t2 = datetime.now()
        dt = (t2 - t1).total_seconds() / 60.0
        dts[method_name] = dt

    ###EVALUATION
    # compare run times and creates equal length intervals
    th = min(list(dts.values()))
    for method_name in all_particles:
        particles = all_particles[method_name]
        particles_intervals = equal_interval_creation(
            particles,
            fields=list(problem.truth.keys()),
            time_intervals=time_intervals,
            time_horizon=th,
            time_passed=dts[method_name]
        )
        for particles in particles_intervals:
            evaluator.callback(particles)
            evaluator.evaluate(**particles, method=method_name)

    print(dts)

    # plot results
    problem.plot_results(evaluator, plot_inference_path)

    # save the results
    evaluator.log_results(results_path)


def setup_experiment(experiment_dir, config):
    """
    Create the required folders to run the experiment, if they don't exists.
    :param experiment_dir: The directory to create
    :param config: Dictionary with extra configurations
    :return: directory_created,dictionary of all the configurations
    """
    experiment_dir = f'./experiments/artificial/{experiment_name}'
    configJson_dir = f'{experiment_dir}/config.json'
    # Save and log directory
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
    if config is None:
        # Read the config from json
        with open(configJson_dir) as json_file:
            config = json.load(json_file)
    else:
        # Dump the config to json
        with open(configJson_dir, 'w') as fp:
            json.dump(config, fp)
    return experiment_dir, config


if __name__ == '__main__':
    number_of_runs = 3
    config = {
        "seed": 0,
        "problem": {
            "kind": "gaussian",
            "N": 5,
            "T": 50,
            "center_prior": False,
        },
        "methods": [
            {
                "kind": "mcmc",
                "name": "mcmc - uniform",
                "choice": "uniform",
                "config": {
                    "step_size": 0.085,
                    "num_results": 1400,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            },
            {
                "kind": "hvi",
                "name": "hvi-q=5",
                "config": {
                    "MC": 32,
                    "epochs": 9000,
                    "optim_config": {
                        "learning_rate": 0.01,
                        "beta_1": 0.9
                    },
                    "sampler_config": {
                        "init_scale": 0.2,
                        "epsilon_config": {
                            "d": 5
                        },
                        "layers_config": [
                            {
                                "kind": "affine",
                                "d": 5
                            },
                            {
                                "kind": "affine",
                                "d": 5
                            },
                            {
                                "kind": "affine_large_dim",
                                "d1": 5
                            }
                        ]
                    }
                }
            },
            {
                "kind": "hvi",
                "name": "hvi",
                "config": {
                    "MC": 32,
                    "epochs": 9000,
                    "optim_config": {
                        "learning_rate": 0.01,
                        "beta_1": 0.9
                    },
                    "sampler_config": {
                        "init_scale": 0.1,
                        "epsilon_config": {},
                        "layers_config": [
                            {
                                "kind": "affine",
                            },
                            {
                                "kind": "affine",
                            },
                            {
                                "kind": "affine",
                            }
                        ]
                    }
                }
            },
            {
                "kind": "mcmc",
                "name": "mcmc - hamiltonian",
                "choice": "hamiltonian",
                "config": {
                    "step_size": 0.09,
                    "num_results": 1400,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            }
        ]
    }
    # Set config to None ro read the file in the experiment directory
    # config = None
    for i in range(1, number_of_runs + 1):
        experiment_name = f'N5_run_{i}'
        config['problem']['scale_variability'] = 2*i - 1
        experiment_dir, config = setup_experiment(experiment_name, config)
        experiment(experiment_dir, **config)

    config = {
        "seed": 0,
        "problem": {
            "kind": "gaussian",
            "N": 10,
            "T": 50,
            "center_prior": False,
        },
        "methods": [
            {
                "kind": "mcmc",
                "name": "mcmc - uniform",
                "choice": "uniform",
                "config": {
                    "step_size": 0.04,
                    "num_results": 2800,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            },
            {
                "kind": "hvi",
                "name": "hvi-q=5",
                "config": {
                    "MC": 32,
                    "epochs": 10000,
                    "optim_config": {
                        "learning_rate": 0.0022,
                        "beta_1": 0.9
                    },
                    "sampler_config": {
                        "init_scale": 1.0,
                        "epsilon_config": {
                            "d": 5
                        },
                        "layers_config": [
                            {
                                "kind": "affine",
                                "d": 5
                            },
                            {
                                "kind": "affine",
                                "d": 5
                            },
                            {
                                "kind": "affine_large_dim",
                                "d1": 5
                            }
                        ]
                    }
                }
            },
            {
                "kind": "hvi",
                "name": "hvi",
                "config": {
                    "MC": 32,
                    "epochs": 10000,
                    "optim_config": {
                        "learning_rate": 0.0012,
                        "beta_1": 0.9
                    },
                    "sampler_config": {
                        "init_scale": 1.0,
                        "epsilon_config": {},
                        "layers_config": [
                            {
                                "kind": "affine",
                            },
                            {
                                "kind": "affine",
                            },
                            {
                                "kind": "affine",
                            }
                        ]
                    }
                }
            },
            {
                "kind": "mcmc",
                "name": "mcmc - hamiltonian",
                "choice": "hamiltonian",
                "config": {
                    "step_size": 0.04,
                    "num_results": 2800,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            }
        ]
    }
    # Set config to None ro read the file in the experiment directory
    # config = None
    for i in range(1, number_of_runs + 1):
        experiment_name = f'N10_run_{i}'
        config['problem']['scale_variability'] = 2*i - 1
        experiment_dir, config = setup_experiment(experiment_name, config)
        #experiment(experiment_dir, **config)

    config = {
        "seed": 0,
        "problem": {
            "kind": "gaussian",
            "N": 15,
            "T": 50,
            "center_prior": False,
        },
        "methods": [
            {
                "kind": "mcmc",
                "name": "mcmc - uniform",
                "choice": "uniform",
                "config": {
                    "step_size": 0.012,
                    "num_results": 5000,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            },
            {
                "kind": "hvi",
                "name": "hvi-q=5",
                "config": {
                    "MC": 32,
                    "epochs": 13000,
                    "optim_config": {
                        "learning_rate": 0.00125,
                        "beta_1": 0.9
                    },
                    "sampler_config": {
                        "init_scale": 0.1,
                        "epsilon_config": {
                            "d": 5
                        },
                        "layers_config": [
                            {
                                "kind": "affine",
                                "d": 5
                            },
                            {
                                "kind": "affine",
                                "d": 5
                            },
                            {
                                "kind": "affine_large_dim",
                                "d1": 5
                            }
                        ]
                    }
                }
            },
            {
                "kind": "hvi",
                "name": "hvi",
                "config": {
                    "MC": 32,
                    "epochs": 13000,
                    "optim_config": {
                        "learning_rate": 0.00052,
                        "beta_1": 0.9
                    },
                    "sampler_config": {
                        "init_scale": 0.1,
                        "epsilon_config": {},
                        "layers_config": [
                            {
                                "kind": "affine",
                            },
                            {
                                "kind": "affine",
                            },
                            {
                                "kind": "affine",
                            }
                        ]
                    }
                }
            },
            {
                "kind": "mcmc",
                "name": "mcmc - hamiltonian",
                "choice": "hamiltonian",
                "config": {
                    "step_size": 0.0122,
                    "num_results": 5000,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            }
        ]
    }
    # Set config to None ro read the file in the experiment directory
    # config = None
    for i in range(1, number_of_runs + 1):
        experiment_name = f'N15_run_{i}'
        config['problem']['scale_variability'] = 2*i - 1
        experiment_dir, config = setup_experiment(experiment_name, config)
        experiment(experiment_dir, **config)
