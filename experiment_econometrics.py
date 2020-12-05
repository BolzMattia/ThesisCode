from econometrics_problem import DatasetFactory
import timeseries_transformations as T
from evaluators import evaluators_timeseries
from datetime import datetime
from utils import runtime_print
import matplotlib.pyplot as plt
from gaussian_multivariate_model import log_lkl_timeseries
import os
import numpy as np
import pandas as pd
import sys

# Size of the figure to be used for latex import
_figsize_latex = [12, 6]


@runtime_print
def train(dsf, m, evaluator, wrapper_name,
          plot_points, single_round_epochs, batch_size, verbose):
    """
    Train a single model and evaluate its performances with evaluator.observe.
    :param dsf: An object of type WrapperPandasCsv
    :param m: The model object
    :param evaluator: An object of type timeseries_evaluator
    :param wrapper_name: The name of the model, str
    :param plot_points: How many times evaluate the performances during the train process, int
    :param single_round_epochs: How many train steps between two performances evaluations
    :param batch_size: The size of the training batch
    :param verbose: Control the verbosity of the m.fit, int
    :return: The list of the evaluated performances
    """
    elbos = []
    for i in range(plot_points):
        print(f'{wrapper_name}, round: {i}')
        # Fit round
        _elbos = m.fit(dsf, epochs=single_round_epochs, batch_size=batch_size, verbose=verbose)
        elbos.append(np.mean(_elbos))
        # Predict
        states, theta, llkl, p0, log_p = m.density_forecast_multi_particles(dsf)
        evaluator.observe(llkl.numpy(), method=wrapper_name, time=datetime.now(), epoch=i)
    return elbos


def estimate_vola_first_series(m, dsf, n_particles, volatility2plot_quantiles):
    """
    Estimate volatility quantiles with multiple particle forecast.
    :param m: The model object
    :param dsf: An object of type WrapperPandasCsv
    :param n_particles: How many particles should be used to estimate the volatility distribution
    :param volatility2plot_quantiles: Which quantiles of the volatility distribution should be evaluated
    :return: The estimated quantiles of the volatility distribution
    """
    llkls = []
    volatility2plot = []
    index1 = 0
    index2 = 0
    try:
        for i in range(n_particles):
            states, theta, llkl, p0, log_p = m.density_forecast(dsf)
            v_i = states['vcv'][:, index1, index2]
            if index1 == index2:
                v_i = np.sqrt(v_i)
            volatility2plot.append(v_i)

        # Quantiles are returned, not the whole trajectories
        volatility2plot = np.quantile(np.stack(volatility2plot, axis=1), volatility2plot_quantiles, axis=1).transpose()
    except:
        volatility2plot = np.ones([dsf.indexes_test.shape[0], len(volatility2plot_quantiles)])
    return volatility2plot


def run_multiple_models(experiment_dir,
                        models,
                        dsf,
                        evaluator,
                        baseline_model,
                        plot_points, single_round_epochs, batch_size,
                        debug=False):
    """
    Function that wraps the training of various models.
    :param experiment_dir: The directory where to save the performances of the models, str
    :param models: A dictionary {'model_name': model_object}
    :param dsf: An object of type WrapperPandasCsv
    :param evaluator: An object of type timeseries_evaluator
    :param baseline_model: A str or a list with the baseline model name within the dsf feature's
    :param plot_points: How many times evaluate the performances during the train process, int
    :param single_round_epochs: How many train steps between two performances evaluations
    :param batch_size: The size of the training batch
    :param debug: If True set the models.fit method as verbose and stops the training when an error occurs.
    :return: The estimated volatility's quantiles on the train set, for every method,
             as a dictionary {'model_name':quantile}
    """
    # Baseline model
    if type(baseline_model) == str:
        baseline_model_features = [baseline_model]
    else:
        baseline_model_features = baseline_model
    for baseline_model_feature in baseline_model_features:
        y, baseline = dsf.select(['y', baseline_model_feature], train=False, test=True)
        llkl_baseline = log_lkl_timeseries(y, **baseline)
        for i in range(plot_points + 1):
            evaluator.observe(llkl_baseline.numpy(), method=baseline_model_feature, time=datetime.now(), epoch=i)

    # Other models
    elbos = {}
    volatilities = {}
    volatility2plot_quantiles = [0.05, 0.5, 0.95]
    for name in models:
        def main():
            model = models[name]
            elbos[name] = train(
                dsf, model, evaluator, name, plot_points, single_round_epochs, batch_size, verbose=int(debug)
            )
            volatility_last_epoch = estimate_vola_first_series(model, dsf,
                                                               model.n_particles_forecast, volatility2plot_quantiles)
            if model.n_particles_forecast > 1:
                volatilities = {f'{name} q: {volatility2plot_quantiles[i]}': volatility_last_epoch[:, i]
                                for i in range(volatility_last_epoch.shape[1])}
            else:
                volatilities = {name: volatility_last_epoch[:, 0]}
            # Plot volatility first series
            fig, ax = plt.subplots(1, 1, figsize=_figsize_latex)
            ax.plot(dsf.indexes_test, volatility_last_epoch)
            period = int(dsf.indexes_test.shape[0] / 10)
            for i, l in enumerate(ax.get_xticklabels()):
                val = str(dsf.indexes_test[i])
                if (i % period) == 0:
                    l.set_visible(True)
                else:
                    l.set_visible(False)

            fig.savefig(f'{experiment_dir}/{name}_vola_last_epoch.png')
            plt.close(fig)
            # save
            evaluator.save_results(experiment_dir)
            # trace-plot
            fig = plt.figure(figsize=_figsize_latex)
            plt.plot(elbos[name])
            plt.savefig(f'{experiment_dir}/{name}_elbo_trace.png')
            plt.close(fig)
            # Performances plot
            fig = plt.figure(figsize=_figsize_latex)
            evaluator.plot_time_advances(f'{experiment_dir}/trace_score.png', methods=[*baseline_model_features, name])
            plt.ylim(-31, 21)
            [plt.savefig(f'{experiment_dir}/{name}_performances.{extension}', format=extension)
             for extension in ['eps', 'png']]
            plt.close(fig)
            return volatilities

        if debug:
            volatilities = {**volatilities, **main()}
        else:
            try:
                volatilities = {**volatilities, **main()}
            except:
                e = sys.exc_info()[0]
                print(f'Error in inference for {name}.\n')
                print(e)

    volatilities = pd.DataFrame(volatilities)
    volatilities.index = dsf.indexes_test
    return volatilities


def main_experiment(dsf,
                    experiment_name,
                    baseline_model='dcc',
                    debug=True,
                    models={},
                    batch_size=16,
                    total_epochs=5000,
                    plot_points=200):
    """
    The main function to run an experiment of chapter 4.
    :param dsf: An object of type WrapperPandasCsv
    :param experiment_name: The name of the experiment, str
    :param baseline_model: A str or a list with the baseline model name within the dsf feature's
    :param debug: Set to True to print training information and to stop if any error occurs
    :param models: A dictionary {'model_name': model_object}
    :param batch_size: The size of the training batch
    :param total_epochs: How many train steps should be done, for every model
    :param plot_points: How many times evaluate the performances during the train process, int
    :return: None
    """
    # plot & log results
    experiment_dir = f'./experiments/econometrics/{experiment_name}'
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)

    # Train it in separated steps to check progresses
    single_round_epochs = int(total_epochs / plot_points)

    evaluator = evaluators_timeseries(dsf.indexes_test)

    ## run the models
    volatilities = run_multiple_models(experiment_dir, models, dsf, evaluator, baseline_model,
                                       plot_points, single_round_epochs, batch_size, debug=debug)

    # final plot of the results
    fig = plt.figure(figsize=_figsize_latex)
    if type(baseline_model) == str:
        baseline_model = [baseline_model]
    method2plot = list(models.keys()) + baseline_model
    evaluator.plot_time_advances(f'{experiment_dir}/{baseline_model}_performances',
                                 methods=method2plot)
    plt.close(fig)

    # final plot of the volatilities
    fig, ax = plt.subplots(1, 1, figsize=_figsize_latex)
    volatilities.plot(ax=ax)
    # plt.title(f'{dataset_name} - {baseline_model}')
    for extension in ['eps', 'png']:
        plt.savefig(f'{experiment_dir}/{baseline_model}_vola.{extension}', format=extension)
    plt.close(fig)


if __name__ == '__main__':
    from wrappers import CholeskyVariationalAutoencoder

    # plot & log results
    experiment_name = 'testing'
    experiment_dir = f'./experiments/econometrics/{experiment_name}'
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)

    # Data reading
    dsf = DatasetFactory.get(name='FRED_raw')
    # Train test split
    train_time = dsf.split_train_test(sequential=True, test_size=0.3)

    # Features computation
    dsf.withColumn('y', *T.standardize(*dsf['y']))
    # dsf.withColumn('dcc', *T.dcc(*dsf['y']))
    dsf.withColumn('crosses', *T.elements_cross(*dsf['y']))
    dsf.withColumn('x_timesteps', *T.lagify(*dsf['y'], lag=12, collapse=False))
    dsf.withColumn('x', *T.lagify(*dsf['y'], lag=12, collapse=True))
    dsf.withColumn('ledoitWolf_static', *T.staticLedoitWolf(*dsf['y']))
    dsf.dropna()

    # evaluators
    evaluator = evaluators_timeseries(dsf.indexes_test)

    # EXPERIMENTS
    batch_size = 16
    total_epochs = 50
    plot_points = 5
    # Train it in separated steps to check progresses
    single_round_epochs = int(total_epochs / plot_points)

    # baseline model
    baseline_model_name = 'ledoitWolf_static'

    # Bayesian models
    models = {
        # 'bayesian DCC': DCC_HVI('x_timesteps', 'y', n=dsf.df.shape[1]),
        'regularized variational autoencoder': CholeskyVariationalAutoencoder(dsf, 'x', 'y', baseline_model_name,
                                                                              init_scale=1e-1),
        # 'standard projection': CholeskyMLP(dsf, 'x', 'y', baseline_model_name),
        # 'regularized autoencoder': CholeskyAutoencoder(dsf, 'x', 'y', baseline_model_name),
        # 'lstm': CholeskyLSTM(dsf, 'x_timesteps', 'y', baseline_model_name),
    }

    ## run the models
    run_multiple_models(experiment_dir, models, dsf, evaluator, baseline_model_name, plot_points, single_round_epochs,
                        batch_size)
