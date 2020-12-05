import numpy as np
import pandas as pd
from econometrics_problem import DatasetFactory, datasets
import timeseries_transformations as T
import seaborn as sns
import matplotlib.pyplot as plt
import os
from matplotlib import transforms
from logger import experiment_logger


# Table of moments
def moments_table(df):
    """
    Format the moments of a dataframe for a Latex table.
    :param df: pandas Dataframe
    :return: pandas Dataframe
    """
    stats = df.describe().round(4).transpose()
    stats = stats.drop('count', axis=1)
    return stats


def view_dataset(dsf: Data, experiment_dir):
    """
    Save a table and two figures about the pandas dataframe in dsf.df
    :param dsf: an object of class DataframeWrapperPandas
    :param experiment_dir: The directory path to save the results
    :return: None
    """
    train_time = dsf.split_train_test(sequential=True, test_size=0.3)
    mom_test_train = moments_table(dsf.df)
    mom_test_train.to_csv(f'{experiment_dir}/test_train_moments.csv')
    mom_train = moments_table(dsf.df.loc[dsf.indexes_train])
    mom_train.to_csv(f'{experiment_dir}/train_moments.csv')
    mom_test = moments_table(dsf.df.loc[dsf.indexes_test])
    mom_test.to_csv(f'{experiment_dir}/test_moments.csv')
    # Plot series
    fig, axes = plt.subplots(2, 1, figsize=[15, 7.5], sharex=True)
    price0 = 100.0
    prices = np.exp(dsf.df.cumsum()) * price0
    axes[0].set_ylabel('Raw values (base 100)')
    lprices = prices.plot(ax=axes[0])
    axes[1].set_ylabel('Log-variations')
    lreturns = dsf.df.plot(ax=axes[1], legend=False)
    lgd = axes[0].legend( loc="upper left", ncol=3)
    # plt.legend(lprices, dsf.df.columns.values, loc='lower center', bbox_to_anchor=(0, -0.1, 1, 1),bbox_transform=plt.gcf().transFigure)
    for save_format in ['eps', 'png']:
        fig.savefig(f'{experiment_dir}/samplefigure.{save_format}', bbox_extra_artists=(lgd,), format=save_format,
                    )


if __name__ == '__main__':
    for dataset_name in datasets:
        # plot & log results
        # experiment_name = 'fred_monthly'
        experiment_dir = f'./explorations/{dataset_name.lower()}'
        if not os.path.isdir(experiment_dir):
            os.mkdir(experiment_dir)

        # Data reading
        dsf = DatasetFactory.get(name=dataset_name)
        # Data exploration
        view_dataset(dsf, experiment_dir)
