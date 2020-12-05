from experiment_econometrics import main_experiment
from econometrics_problem import DatasetFactory
from learners import CholeskyMLP, CholeskyLSTM, CholeskyAutoEncoder, HVIstaticMuVcv
from wrappers import DCC_HVI
import mean_covariance_models as M
import timeseries_transformations as T

if __name__ == '__main__':
    experiment_name = 'fred_hvi_1010'
    dataset_name = 'FRED'
    batch_size = 16
    total_epochs = 100000
    plot_points = 30

    # Data from .csv
    dsf = DatasetFactory.get(name=dataset_name)
    # Train test split
    train_time = dsf.split_train_test(sequential=True, test_size=0.3)
    # Features computation
    dsf.withColumn('y', *T.standardize(*dsf['y']))
    dsf.withColumn('crosses', *T.elements_cross(*dsf['y']))
    dsf.withColumn('x_timesteps', *T.lagify(*dsf['y'], lag=36, collapse=False))
    dsf.withColumn('x', *T.lagify(*dsf['y'], lag=12, collapse=True))
    # The baseline model
    baseline_estimators = {
        'dcc': T.dcc,
        'lw': T.staticLedoitWolf,
    }
    baseline_models_line = {}
    all_baseline = list(baseline_estimators.keys())
    for baseline_model in all_baseline:
        baseline_model_line = f'{baseline_model}_line'
        baseline_models_line[baseline_model] = baseline_model_line
        baseline_estimator = baseline_estimators[baseline_model]
        dsf.withColumn(baseline_model, *baseline_estimator(*dsf['y']))
        states_line = [M.mu_vcv_linearize(**states) for states in dsf[baseline_model]]
        dsf.withColumn(baseline_model_line, *states_line)

    # Drop rows with NA (empty features)
    dsf.dropna()

    models = {
        'HVI(LW)': HVIstaticMuVcv(
            dsf, 'y', baseline_models_line['lw'],
            init_scale=5e-1, learning_rate=1e-5),
        # Not yet tested
        # 'HVI(DCC)': DCC_HVI('x_timesteps', 'y', n=dsf.df.shape[1], init_scale=3e-1, learning_rate=2e-6),
    }

    main_experiment(dsf,
                    experiment_name=experiment_name,
                    baseline_model=all_baseline,
                    debug=False,
                    models=models,
                    batch_size=batch_size,
                    total_epochs=total_epochs,
                    plot_points=plot_points)
