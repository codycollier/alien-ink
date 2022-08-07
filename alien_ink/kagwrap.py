""" Kaggle wrapper and utilities

"""

from . import config   # config must be loaded before kaggle

import os
import sys
import zipfile

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas


def get_comp_data(competition_name):
    """Download, prepare, and dataframe the data from a competition"""

    # get the zip if it doesn't yet exist
    cdata_root = os.path.join(config.kaggle["data_root"], competition_name)
    cdata_file = os.path.join(cdata_root, f"{competition_name}.zip")
    if not os.path.exists(cdata_file):
        kapi = KaggleApi()
        kapi.authenticate()
        kapi.competition_download_files(competition_name, path=cdata_root)

    # unzip if needed
    csv_trn = os.path.join(cdata_root, "train.csv")
    csv_tst = os.path.join(cdata_root, "test.csv")
    if (not os.path.exists(csv_tst)) or (not os.path.exists(csv_trn)):
        with zipfile.ZipFile(cdata_file) as z:
            z.extractall(cdata_root)

    # load
    df_trn = pandas.read_csv(csv_trn)
    df_tst = pandas.read_csv(csv_tst)
    return df_trn, df_tst



if __name__ == "__main__":

    compname = 'digit-recognizer'
    df_train, df_test = get_comp_data(compname)
    print(df_test.head())
    print(df_train.head())


