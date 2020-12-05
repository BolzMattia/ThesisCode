from experiment_econometrics import main_experiment
from econometrics_problem import DatasetFactory
from learners import CholeskyMLP, CholeskyLSTM, CholeskyAutoEncoder
from wrappers import DCC_HVI
import mean_covariance_models as M
import timeseries_transformations as T

if __name__ == '__main__':
    experiment_name = 'ptf_dcc_LSTM_1010'
    dataset_name = 'PTF'
    baseline_model = 'dcc'
    hidden_layer_dim = 32
    batch_size = 16
    total_epochs = 30000
    plot_points = 30

    # Data from .csv
    dsf = DatasetFactory.get(name=dataset_name)
    # Train test split
    train_time = dsf.split_train_test(sequential=True, test_size=0.3)
    # Features computation
    dsf.withColumn('y', *T.standardize(*dsf['y']))
    dsf.withColumn('crosses', *T.elements_cross(*dsf['y']))
    dsf.withColumn('x_timesteps', *T.lagify(*dsf['crosses'], lag=6, collapse=False))
    # The baseline model
    baseline_estimator = {
        'dcc': T.dcc,
        'lw': T.LedoitWolf,
    }[baseline_model]
    baseline_model_line = f'{baseline_model}_line'
    dsf.withColumn(baseline_model, *baseline_estimator(*dsf['y']))
    states_line = [M.mu_vcv_linearize(**states) for states in dsf[baseline_model]]
    dsf.withColumn(baseline_model_line, *states_line)
    # Drop rows with NA (empty features)
    dsf.dropna()

    models = {
        'DCC + VI-lstm': CholeskyLSTM(dsf,
                                            'x_timesteps', 'y', baseline_model_line,
                                            gaussian_posterior=True,
                                            recurrent_dim=hidden_layer_dim,
                                            mlp_post_lstm_layers_dim=[hidden_layer_dim for i in range(2)],
                                            init_scale=1e0, learning_rate=5e-4),
        'DCC + p-lstm': CholeskyLSTM(dsf,
                                             'x_timesteps', 'y', baseline_model_line,
                                             recurrent_dim=hidden_layer_dim,
                                             mlp_post_lstm_layers_dim=[hidden_layer_dim for i in range(2)],
                                             init_scale=5e-1, learning_rate=5e-5),
        'DCC + lstm': CholeskyLSTM(dsf,
                                   'x_timesteps', 'y', baseline_model_line,
                                   empirical_prior=False,
                                   recurrent_dim=hidden_layer_dim,
                                   mlp_post_lstm_layers_dim=[hidden_layer_dim for i in range(2)],
                                   init_scale=5e-1, learning_rate=5e-5),
    }

    main_experiment(dsf,
                    experiment_name=experiment_name,
                    baseline_model=baseline_model,
                    debug=False,
                    models=models,
                    batch_size=batch_size,
                    total_epochs=total_epochs,
                    plot_points=plot_points)
