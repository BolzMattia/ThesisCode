import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from logger import create_folder


def quantile_func(q):
    """Closure function for np.quantile"""
    def f(x):
        return np.quantile(x, q)

    return f


class tester_performances:
    """
    This class collects particles estimate of a distribution and compare truth_particles moments.
    """
    def __init__(self, truth_particles, metrics):
        """
        Instantiat the class
        :param truth_particles: The truth_particles, distributed as the target, numpy with dim=mc,d
        :param metrics: The comparison metric (as mse or mae)
        """
        self.moments = {'avg': np.mean, 'std': np.std}
        self.truth = truth_particles
        self.metrics = metrics
        self.results = []

    def evaluate(self, particles, **kwargs):
        """
        Observe a set of particles with dim=mc,d and compare the moments to self.truth_particles
        :param particles: numpy with dim=mc,d
        :param kwargs: key-args dictionary of extra tags for the evaluated particles.
        :return:
        """
        truth = self.truth
        metrics = self.metrics
        moments = self.moments
        results = []
        # for key in truth:
        th_hvi = np.concatenate([particles[key].reshape(particles[key].shape[0], -1) for key in truth], axis=1)
        tr = np.concatenate([truth[key].reshape(truth[key].shape[0], -1) for key in truth], axis=1)
        for mom_name in moments:
            mom_func = moments[mom_name]
            dist = metrics(mom_func(tr, axis=0), mom_func(th_hvi, axis=0))
            result = {'moment': mom_name, 'distance': dist, **kwargs}
            self.results.append(result)
        return results

    def get_pandas(self):
        """Return a pandas with the error metrics"""
        return pd.DataFrame(self.results)

    def log_results(self, path):
        """Save the results to a .csv"""
        pd.DataFrame(self.results).to_csv(path)

    def callback(self, theta):
        pass


class evaluators_timeseries:
    """
    This class collects loss values on a test set and computes plots and statistics to evaluate them.
    """
    _results = []
    _scores = []
    statistics = {
        'avg': np.mean,
        'std': np.std,
        'q05': quantile_func(0.05),
        #'q25': quantile_func(0.25),
        'q50': quantile_func(0.50),
        #'q75': quantile_func(0.75),
        'q95': quantile_func(0.95),
    }

    def __init__(self, index_scores):
        """The index_scores, will be used to plot results"""
        self.index_scores = index_scores

    def observe(self, scores, **fields):
        """
        Observe a set of scores with dim t,1
        :param scores: a numpy of scores obtained on the test set, dim t,1
        :param fields: Extra tags of the model that obtained the scores
        :return: None
        """
        [self._scores.append({'value': s, 'index': i, **fields}) for s, i in zip(scores, self.index_scores)]
        for s in self.statistics:
            v = self.statistics[s](scores)
            self._results.append({'statistics': s, 'value': v, **fields})

    def get_results(self, methods: list = None):
        """Return a pandas with the results with tag methods in methods"""
        df = pd.DataFrame(self._results)
        if (methods is not None) & ('method' in df.columns):
            df = df.loc[[x in methods for x in df.method.values]]
        return df

    def get_scores(self):
        """Return all the scores as a pandas object"""
        return pd.DataFrame(self._scores)

    def get_pivot_last_epoch(self):
        """Return a pandas with the performances of the last train epoch, for every model."""
        df = self.get_results()
        df_max_epoch = df.groupby('method')['epoch'].max().reset_index()
        df_max_epoch = df.merge(df_max_epoch, on=['method', 'epoch'], how='inner')
        df_pivot = df_max_epoch.pivot('method', 'statistics', 'value')
        return df_pivot

    def save_results(self, path):
        """Save the pandas results as a .csv"""
        create_folder(path)
        self.get_scores().to_csv(path + r'/scores.csv', index=False)
        self.get_results().to_csv(path + r'/results.csv', index=False)
        self.get_pivot_last_epoch().to_csv(path + r'/pivot_last_epoch.csv', index=True)

    def plot_time_advances(self, path: str = None, methods: list = None, title=None):
        """
        Plot different statistics about performances during the training process.
        :param path: The path to save the figure
        :param methods: The methods to include in the plot
        :param title: The title of the plot
        :return: None
        """
        df = self.get_results(methods)
        ax = plt.axes()
        sns.lineplot(x='epoch', y='value', data=df, hue='statistics', style='method', ax=ax)
        plt.ylim(-31, 11)
        plt.ylabel('test statistics of log-lkl')
        lgd = plt.legend(loc='upper left', bbox_to_anchor=[1.01, -0.1, 0.2, 0.8], ncol=1)
        ax.set_position([0.1, 0.1, 0.75, 0.8])
        if title is not None:
            ax.set_title(title)
        if path is not None:
            for extension in ['eps', 'png']:
                plt.savefig(f'{path}.{extension}', format=extension)


if __name__ == '__main__':
    # Testing evaluator
    from sklearn.metrics import mean_squared_error

    truth_particles = {'a': np.random.randn(10000, 5), 'b': np.random.rand(10000, 3)}
    evaluator = tester_performances(truth_particles, mean_squared_error)
    for i in range(10):
        mc_particles = {'a': np.random.randn(1000, 5), 'b': np.random.rand(1000, 3)}
        evaluator.evaluate(mc_particles)
        print(i)
        print(evaluator.get_pandas())

    # Testing evaluator timeseries
    t = 100
    index_scores = np.arange(t)
    e = evaluators_timeseries(index_scores)
    t0 = datetime.now()
    for i in range(10):
        e.observe(np.random.randn(100), method='prova', time=datetime.now(), epoch=i)
        e.observe(np.random.randn(100), method='prova2', time=datetime.now(), epoch=i)
    print(e.get_results())
    print(e.get_scores())
    e.plot_time_advances()
    plt.show()
    df = e.get_results()

    # pivoting results
    df = e.get_pivot_last_epoch()
    print(df)
