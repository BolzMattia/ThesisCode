from experiment_econometrics import main_experiment
from econometrics_problem import DatasetFactory
from learners import CholeskyMLP, CholeskyLSTM, CholeskyAutoEncoder
from wrappers import DCC_HVI
import mean_covariance_models as M
import timeseries_transformations as T


if __name__ == '__main__':
    experiment_name = 'testing_main'
    dataset_name = 'PTF'
    baseline_model = 'dcc'
    hidden_layer_dim = 32
    batch_size = 16
    total_epochs = 50
    plot_points = 2

    # Data reading
    dsf = DatasetFactory.get(name=dataset_name)
    # Train test split
    train_time = dsf.split_train_test(sequential=True, test_size=0.3)

    baseline_estimator = {
        'dcc': T.dcc,
        'lw': T.LedoitWolf,
    }[baseline_model]
    baseline_model_line = f'{baseline_model}_line'

    # Features computation
    dsf.withColumn('y', *T.standardize(*dsf['y']))
    dsf.withColumn(baseline_model, *baseline_estimator(*dsf['y']))
    states_line = [M.mu_vcv_linearize(**states) for states in dsf[baseline_model]]
    dsf.withColumn(baseline_model_line, *states_line)
    dsf.withColumn('crosses', *T.elements_cross(*dsf['y']))
    dsf.withColumn('x_timesteps', *T.lagify(*dsf['y'], lag=12, collapse=False))
    dsf.withColumn('x', *T.lagify(*dsf['y'], lag=12, collapse=True))
    dsf.dropna()

    models = {
        'DCC + vae': CholeskyAutoEncoder(
            dsf, 'x', 'y', baseline_model_line,
            variational_reparametrization=True,
            encoder_layers_dim=[hidden_layer_dim for i in range(5)],
            encode_dim=5,
            forecaster_layers_dim=[hidden_layer_dim for i in range(5)],
            init_scale=3e-2, learning_rate=3e-2),
        'DCC + Bayesian mlp': CholeskyMLP(dsf, 'x', 'y', baseline_model_line,
                                          hidden_layers_dim=[hidden_layer_dim for i in range(5)],
                                          gaussian_posterior=True,
                                          init_scale=5e-2, learning_rate=3e-4, beta_1=0.9),
    }

    main_experiment(dsf,
                    experiment_name=experiment_name,
                    baseline_model='dcc',
                    debug=True,
                    models=models,
                    batch_size=batch_size,
                    total_epochs=total_epochs,
                    plot_points=plot_points)
