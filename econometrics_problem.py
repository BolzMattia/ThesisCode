import pandas as pd
import numpy as np
from utils import decorator_multilevel
from datetime import datetime
from operator import itemgetter


def pad_column(col):
    '''Pad a column negative values using the last positive element.
    If the series starts with a negative value, pads with the first
    non-negative element of the future.'''
    col = col.copy()
    last_val = (col > 0).argmax()
    last_val = col[last_val]
    for r in range(col.shape[0]):
        if col.iloc[r] < 0:
            col.iloc[r] = last_val
        else:
            last_val = col.iloc[r]
    return col


def pad_nan_and_negative(y_df):
    # Cleans the NaN and negative values
    y_df = y_df.copy()
    y_df[y_df.isnull()] = -9999.0
    y_df = y_df.loc[(y_df < 0).mean(axis=1) < 0.1, :]
    y_df = y_df.loc[:, (y_df < 0).mean(axis=0) < 0.005]

    for c in y_df:
        col = pad_column(y_df[c])
        y_df[c] = col

    return y_df


def select_timeseries_dict(df_dict, key2select, index2select):
    if type(df_dict) != dict:
        x = df_dict[index2select]
    else:
        x = {k: (
            select_timeseries_dict(df_dict[k], df_dict[k].keys(), index2select)
        ) for k in key2select}
    return x


def select_timeseries_dict_size(df_dict, key2select, size):
    # Any k0 is ok
    k0 = list(df_dict.keys())[0]
    indexes = np.random.choice(df_dict[k0].index, size=size, replace=False)
    df_batch = select_timeseries_dict(df_dict, key2select, indexes)
    return df_batch


def select_dict(df_dict, index2select):
    if type(df_dict) != dict:
        x = df_dict[index2select]
    else:
        print(type(df_dict))
        x = {k: (
            select_dict(df_dict[k], index2select)
        ) for k in df_dict.keys()}
    return x


def select_size(x, size: int, sequential=True):
    indexes = np.random.choice(np.arange(x.shape[0]), size=size, replace=False)
    values = x[indexes]
    return indexes, values


def select_dict_size(df_dict, size):
    # Any k0 is ok
    k0 = list(df_dict.keys())[0]
    indexes = np.random.choice(np.arange(df_dict[k0].shape[0]), size=size, replace=False)
    df_batch = select_dict(df_dict, indexes)
    return df_batch


def select_timeseries_dict_split(df_dict, key2select, time_split):
    # Any k0 is ok
    k0 = list(df_dict.keys())[0]
    indexes = df_dict[k0].index < time_split
    ret = [
        # Before split time
        select_timeseries_dict(df_dict, key2select, indexes),
        # After split time
        select_timeseries_dict(df_dict, key2select, ~indexes)
    ]
    return ret


def select_timeseries_size(df: pd.DataFrame, size: int):
    indexes = np.random.choice(df.index, size=size, replace=False)
    df_batch = df.loc[indexes]
    return df_batch


def filter_dict(d: dict, indexok: np.ndarray):
    ret = {}
    for k in d:
        if type(d[k]) == pd.DataFrame:
            ret[k] = d[k].loc[d[k].index[indexok]]
        elif type(d[k]) == np.ndarray:
            ret[k] = d[k][indexok]
        elif type(d[k]) == dict:
            ret[k] = filter_dict(d[k], indexok)
        else:
            raise Exception(f"Type {type(d[k])} not managed")
    return ret

def fill_dict(d: dict, val: float):
    ret = {}
    for k in d:
        ret[k] = d[k]
        if type(d[k]) == pd.DataFrame:
            ret[k] = ret[k].fillna(val)
        elif type(d[k]) == np.ndarray:
            ret[k] = np.nan_to_num(ret[k],val)
        elif type(d[k]) == dict:
            ret[k] = fill_dict(d[k], val)
        else:
            raise Exception(f"Type {type(d[k])} not managed")
    return ret


def indexok_dict(d: dict):
    i = None
    for k in d:
        if (type(d[k]) == pd.DataFrame):
            nulls = d[k].isnull().any(1)
        elif (type(d[k]) == pd.Series):
            nulls = d[k].isnull()
        elif type(d[k]) == np.ndarray:
            nulls = np.isnan(d[k])
            for _ in range(len(d[k].shape) - 1):
                nulls = nulls.any(1)
        elif type(d[k]) == dict:
            nulls = ~ indexok_dict(d[k])
        else:
            raise Exception(f"Type {type(d[k])} not managed")

        if i is None:
            i = nulls
        else:
            i = np.logical_or(i, nulls)
    i = ~ i
    return i


def select_timeseries_split(df, time_split):
    indexes = df.index < time_split
    x_train = df.loc[indexes]
    x_test = df.loc[~indexes]
    return x_train, x_test


class wrapperPandasCsv:
    """
        Class that wrap a .csv and allows to do common operations as:
        - train vs test split
        - Features storage and retrieval
        - Cleaning operations as .dropna, .fillna
    """

    def __init__(self,
                 path,
                 sep=',',
                 pad_negative=True,
                 height=None,
                 width=None,
                 log_return=False):

        self.path = path
        self.df = pd.read_csv(path, header=0, index_col=0, sep=sep)
        if pad_negative:
            self.df = pad_nan_and_negative(self.df)
        if log_return:
            self.df = np.log(self.df.iloc[1:, :] / self.df.values[:-1, :])

        if height is not None:
            if isinstance(height, int):
                height = np.arange(height)

        if width is not None:
            if isinstance(width, int):
                width = np.arange(width)

    def split_train_test(self,
                         test_size=0.3,
                         sequential=True,
                         seed=None,
                         ):
        """
            Separate the wrapped dataframe into train and test.
            :param sizes: List of proportions of data within train and test
                            (e.g. [0.7,0.3] for 30% test size)
            :param sequential: Set =True to use a sequential split (e.g. for time-series).
                               Set =False for random split (e.g. independent rows)
            :param seed: The seed for the random split
            :return: None
        """
        df = self.df
        if seed is not None:
            np.random.seed(seed)
        if sequential:
            # Train set stays before validation
            t1 = int(df.shape[0] * (1 - test_size))
            t1 = df.index[t1]
        else:
            raise Exception("Non-sequential train test split not implemented yet.")
        self.split_time = t1
        self.init_train_test()
        return self.split_time

    def withColumn(self, name, df_train=None, df_test=None, df=None):
        """
            Store the input features in *args with feature's name=name.
            *args should contain two values: one for train and one for test.
            :param name:
            :param args:
            :return: None
        """
        assert not ((df is None) & (df_train is None))
        if (df_train is None) & (df_test is None):
            df_train, df_test = select_timeseries_split(self.df, self.split_time)
        self.x_train[name] = df_train
        self.x_test[name] = df_test

    def get_length(self):
        """Returns the shape of the target variable, on both train and test."""
        return self.x_train['y'].shape, self.x_test['y'].shape

    def init_train_test(self):
        """Split train and test, keeping in memory the indexes of the split."""
        df_train, df_test = select_timeseries_split(self.df, self.split_time)
        self.indexes_train = df_train.index
        self.indexes_test = df_test.index
        self.x_train = {'y': decorator_multilevel(np.float32)(df_train.values)}
        self.x_test = {'y': decorator_multilevel(np.float32)(df_test.values)}

    def __getitem__(self, key: str):
        """Selects the features with name in keys. Extract both train and test."""
        return self.x_train[key], self.x_test[key]

    def select(self, keys, train=True, test=False):
        """Given a list of feature's name=keys, extract them. If train=True return train features,
        if test=True selects test features. If both return both."""
        assert (train | test), "Select at least train or test"
        ret = []
        if train:
            @decorator_multilevel
            def selecting(k):
                return self.x_train[k]

            ret.append(selecting(keys))
        if test:
            @decorator_multilevel
            def selecting(k):
                return self.x_test[k]

            ret.append(selecting(keys))
        if train & test:
            return ret
        else:
            return ret[0]

    def _get_batch_index(self, size: int, sequential=True):
        return select_size(self.indexes_train, size, sequential)

    def get_batch(self, *keys, size: int, sequential=True):
        """Extract a batch of features with name=keys. size of the batch=size.
        If sequential=True extract the observations sequentially."""
        i, t = self._get_batch_index(size, sequential)

        @decorator_multilevel
        def selecting(k):
            @decorator_multilevel
            def subindexing(x):
                return x[i]

            return subindexing(self.x_train[k])

        return selecting(keys), i, t



    def fillna(self, value=0.0, train=True, test=True):
        """Fill every NaN value in every feature row with the value=value."""
        if train:
            self.x_train = fill_dict(self.x_train, value)
        if test:
            self.x_test = fill_dict(self.x_test, value)
        return self.x_train, self.x_test


    def dropna(self, train=True, test=True):
        """Drops every row with at least a NaN in at least a feature.
        Drops such rows from every feature."""
        if train:
            indexok = indexok_dict(self.x_train)
            self.x_train = filter_dict(self.x_train, indexok=indexok)
            self.indexes_train = self.indexes_train[indexok]
        if test:
            indexok = indexok_dict(self.x_test)
            self.x_test = filter_dict(self.x_test, indexok=indexok)
            self.indexes_test = self.indexes_test[indexok]
        return self.x_train, self.x_test

#The list of datasets used as examples
datasets = {
        'FRED_raw': r'./econometrics_data/FRED_edited.csv',
        'FRED': r'./econometrics_data/FRED_deltas.csv',
        'PTF': r'./econometrics_data/PTF_indexes.csv',
}
class DatasetFactory:
    """This class is a simple access point to access and instantiate the  examples datasets."""
    def get(name, **kwargs):
        if name in datasets:
            dataset = wrapperPandasCsv(datasets[name], pad_negative=False, **kwargs)
        elif name == 'PTF_daily':
            dataset = wrapperPandasCsv(r'./econometrics_data/dati_2.3_Analisi_portafoglio_MA.csv', sep=';',
                                       pad_negative=True,
                                       log_return=True, **kwargs)
        else:
            raise ValueError(f'Dataset {name} non riconosciuto dalla factory')
        return dataset


if __name__ == '__main__':
    data_config = {
        'name': 'FRED'
    }
    train_test_config = {
        'sequential': True,
        'test_size': 0.3,
        'seed': 0
    }
    dsf = DatasetFactory.get(**data_config)
    train_time = dsf.split_train_test(**train_test_config)

    import timeseries_transformations as T

    dsf.withColumn('s', *T.staticLedoitWolf(*dsf['y']))

    features = {'a': ['y', 'y'], 'b': 'y', 'c': 's'}
    x_batch, index_batch, time_batch = dsf.get_batch(features, size=10, sequential=True)
    features = ['y', 'y', 's']
    x, index_batch, time_batch = dsf.get_batch(features, size=10, sequential=True)
    print(len(x))

    dsf.fillna(0.0)
    dsf.dropna()

    # Print batch
    print(x_batch)
    # Select train & test
    x_train, x_test = dsf.select(features, test=True)
    x, y, s = dsf.select(features, test=False)
    # print((x_train.keys()))
