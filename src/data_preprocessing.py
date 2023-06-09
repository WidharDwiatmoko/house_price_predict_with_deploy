# from util import load_config, pickle_load, pickle_dump
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn_features.transformers import DataFrameSelector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# import src.util as utils
import util as util
import sys
sys.path.append("config")


def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Return 3 set of data
    return x_train, x_test, y_train, y_test


def dump_data(x_train, y_train, X_test_feng, y_test):
    util.pickle_dump(x_train, "../data/processed/x_train_feng.pkl")
    util.pickle_dump(y_train, "../data/processed/y_train_feng.pkl")

    util.pickle_dump(X_test_feng, "../data/processed/x_test_feng.pkl")
    util.pickle_dump(y_test, "../data/processed/y_test_feng.pkl")


if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config("../config/config.yaml")

    # 2. Load dataset
    x_train, x_test, y_train, y_test = load_dataset(config_data)

    num_cols = [col for col in x_train.columns if x_train[col].dtype in [
        'float32', 'float64', 'int32', 'int64']]
    categ_cols = [col for col in x_train.columns if x_train[col].dtype not in [
        'float32', 'float64', 'int32', 'int64']]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_cols)),  # select only these columns
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categ_pipeline = Pipeline(steps=[
        ('selector', DataFrameSelector(categ_cols)),  # select only these columns
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('OHE', OneHotEncoder(sparse=False))
    ])

    total_pipeline = FeatureUnion(transformer_list=[
        ('num_pipe', num_pipeline),
        ('categ_pipe', categ_pipeline)
    ])
    X_train_final = total_pipeline.fit_transform(x_train)

    # X_train_feng = total_pipeline.fit_transform(x_train)
    X_test_feng = total_pipeline.transform(x_test)

    # 13. Dump data
    dump_data(X_train_final, y_train, X_test_feng, y_test)
