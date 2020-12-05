import pandas as pd
import os
import json
from datetime import datetime


def filter_df_byDict(df, sel_dict, out=False):
    if (df is None) or (df.shape[0] == 0):
        return df
    mask = ((df == pd.Series(sel_dict)) * 1).sum(axis=1) == len(sel_dict)
    if out:
        mask = ~mask
    return df.loc[mask]


def create_folder(name, root_path='.'):
    folder_path = f'{root_path}//{name}'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


class experiment_logger:
    """
    DEPRECATED.
    Represent objects used to log an experiment results as a tree of folders."""
    def __init__(self, name, root_path='experiments'):
        root_folder = create_folder(root_path)
        self.root_folder = create_folder(name, root_folder)
        self.folders = [self.root_folder]
        self.active_level = 0
        self.selected_folder = self.folders[self.active_level]
        self.default_results_name = 'results'
        self.default_configuration_name = 'experiment.conf'
        self.read_history()

    def new_level(self, name, level=None):
        # If level is not specified, starts from the lowest one
        if level is None:
            level = self.active_level + 1

        # Creates the required folder for the new level
        self.selected_folder = create_folder(name, self.folders[level - 1])

        # Adds the new folder to the tree
        if level >= len(self.folders):
            self.folders.append(self.selected_folder)
        else:
            self.folders[level] = self.selected_folder

        # Selects the new level
        self.active_level = level

        print(f'New level ({level}): {name}\nPath: {self.selected_folder}')

    def _get_folder_name(self, name):
        return f'{self.selected_folder}//{name}'

    def save_csv(self, df=None, name=None):
        if name is None:
            name = self.default_results_name
        if df is None:
            df = self.get_obsvs()
        df.to_csv(self._get_folder_name(f'{name}.csv'), index=False)

    def save_dictionary(self, dictionary, name=None):
        if name is None:
            name = self.default_configuration_name
        filename = self._get_folder_name(name)
        with open(filename, 'w') as f:
            json.dump(dictionary, f)

    def read_history(self, name=None):
        if name is None:
            name = self.default_results_name
        try:
            self.obsvs = pd.read_csv(self._get_folder_name(f'{name}.csv'))
        except:
            self.obsvs = None

    def log_notes(self, text, name='logs'):
        with open(self._get_folder_name(f'{name}.txt'), "a") as text_file:
            text_file.write(text)

    def clear_history_by_fields(self, **kwargs):
        self.obsvs = filter_df_byDict(self.obsvs, kwargs, out=True)

    def observe(self, **kwargs):
        kwargs['instimestamp'] = datetime.now()
        obsv = pd.DataFrame([kwargs])
        if self.obsvs is None:
            self.obsvs = obsv
        else:
            self.obsvs = pd.concat([self.obsvs, obsv], axis=0)

    def get_obsvs(self):
        return self.obsvs

    def clear(self):
        self.obsvs = None
