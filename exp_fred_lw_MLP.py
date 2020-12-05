from experiment_econometrics import main_experiment
from econometrics_problem import DatasetFactory
from learners import CholeskyMLP, CholeskyLSTM, CholeskyAutoEncoder
from wrappers import DCC_HVI
import mean_covariance_models as M
import timeseries_transformations as T

if __name__ == '__main__':
    experiment_name = 'fred_lw_MLP_1010'
    dataset_name = 'FRED'
    baseline_model = 'lw'
    hidden_layer_dim = 32
    batch_size = 16
    total_epochs = 40000
    plot_points = 30

    # Data from .csv
    dsf = DatasetFactory.get(name=dataset_name)
    # Train test split
    train_time = dsf.split_train_test(sequential=True, test_size=0.3)
    # Features computation
    dsf.withColumn('y', *T.standardize(*dsf['y']))
    dsf.withColumn('crosses', *T.elements_cross(*dsf['y']))
    dsf.withColumn('x', *T.lagify(*dsf['crosses'], lag=6, collapse=True))
    # The baseline model
    baseline_estimator = {
        'dcc': T.dcc,
        'lw': T.staticLedoitWolf,
    }[baseline_model]
    baseline_model_line = f'{baseline_model}_line'
    dsf.withColumn(baseline_model, *baseline_estimator(*dsf['y']))
    states_line = [M.mu_vcv_linearize(**states) for states in dsf[baseline_model]]
    dsf.withColumn(baseline_model_line, *states_line)
    # Drop rows with NA (empty features)
    dsf.dropna()

    models = {
        'LW + VI-mlp': CholeskyMLP(dsf, 'x', 'y', baseline_model_line,
                                          hidden_layers_dim=[hidden_layer_dim for i in range(2)],
                                          gaussian_posterior=True, empirical_prior=True,
                                          init_scale=3e-1, learning_rate=5e-5, beta_1=0.9),
        'LW + p-mlp': CholeskyMLP(dsf, 'x', 'y', baseline_model_line,
                                           hidden_layers_dim=[hidden_layer_dim for i in range(2)],
                                           gaussian_posterior=False, empirical_prior=True,
                                           init_scale=5e-2, learning_rate=4e-5, beta_1=0.9),
        'LW + mlp': CholeskyMLP(dsf, 'x', 'y', baseline_model_line,
                                 hidden_layers_dim=[hidden_layer_dim for i in range(2)],
                                 gaussian_posterior=False, empirical_prior=False,
                                 init_scale=5e-2, learning_rate=1e-5, beta_1=0.9),
    }

    main_experiment(dsf,
                    experiment_name=experiment_name,
                    baseline_model=baseline_model,
                    debug=False,
                    models=models,
                    batch_size=batch_size,
                    total_epochs=total_epochs,
                    plot_points=plot_points)
