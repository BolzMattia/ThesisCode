from experiment_artificial import experiment, setup_experiment

if __name__ == '__main__':
    number_of_runs = 1
    config = {
        "seed": 0,
        "problem": {
            "kind": "gaussian",
            "N": 5,
            "T": 50,
            "center_prior": True,
        },
        "methods": [
            {
                "kind": "mcmc",
                "name": "mcmc - uniform",
                "choice": "uniform",
                "config": {
                    "step_size": 0.132,
                    "num_results": 700,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            },
            {
                "kind": "hvi",
                "name": "hvi_low_dim",
                "config": {
                    "MC": 32,
                    "epochs": 4500,
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
                "name": "hvi_standard",
                "config": {
                    "MC": 32,
                    "epochs": 4500,
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
                    "step_size": 0.132,
                    "num_results": 700,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            }
        ]
    }
    # Set config to None ro read the file in the experiment directory
    # config = None
    for i in range(number_of_runs):
        experiment_name = f'N5_center_prior_run_{i}'
        config['seed'] = i
        experiment_dir, config = setup_experiment(experiment_name, config)
        experiment(experiment_dir, **config)

    config = {
        "seed": 0,
        "problem": {
            "kind": "gaussian",
            "N": 10,
            "T": 50,
            "center_prior": True,
        },
        "methods": [
            {
                "kind": "mcmc",
                "name": "mcmc - uniform",
                "choice": "uniform",
                "config": {
                    "step_size": 0.1001,
                    "num_results": 1400,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            },
            {
                "kind": "hvi",
                "name": "hvi_low_dim",
                "config": {
                    "MC": 32,
                    "epochs": 5000,
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
                "name": "hvi_standard",
                "config": {
                    "MC": 32,
                    "epochs": 5000,
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
                    "step_size": 0.0806,
                    "num_results": 1400,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            }
        ]
    }
    # Set config to None ro read the file in the experiment directory
    # config = None
    for i in range(number_of_runs):
        experiment_name = f'N10_center_prior_run_{i}'
        config['seed'] = i
        experiment_dir, config = setup_experiment(experiment_name, config)
        experiment(experiment_dir, **config)

    config = {
        "seed": 2020,
        "problem": {
            "kind": "gaussian",
            "N": 15,
            "T": 50,
            "center_prior": True,
        },
        "methods": [
            {
                "kind": "mcmc",
                "name": "mcmc - uniform",
                "choice": "uniform",
                "config": {
                    "step_size": 0.0042,
                    "num_results": 5000,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            },
            {
                "kind": "hvi",
                "name": "hvi_low_dim",
                "config": {
                    "MC": 32,
                    "epochs": 17000,
                    "optim_config": {
                        "learning_rate": 0.0025,
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
                "name": "hvi_standard",
                "config": {
                    "MC": 32,
                    "epochs": 17000,
                    "optim_config": {
                        "learning_rate": 0.0072,
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
                    "step_size": 0.0032,
                    "num_results": 5000,
                    "num_chains": 100,
                    "num_burnin_steps": 1
                }
            }
        ]
    }
    # Set config to None ro read the file in the experiment directory
    # config = None
    for i in range(number_of_runs):
        experiment_name = f'N15_center_prior_run_{i}'
        config['seed'] = i
        experiment_dir, config = setup_experiment(experiment_name, config)
        experiment(experiment_dir, **config)
