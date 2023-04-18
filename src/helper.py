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
import src.util as utils
import sys
sys.path.append("config")


def load_dataset_prepnew(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = utils.pickle_load(config_data["train_set_path"][2])
    y_train = utils.pickle_load(config_data["train_set_path"][3])

    x_test = utils.pickle_load(config_data["test_set_path"][2])
    y_test = utils.pickle_load(config_data["test_set_path"][3])

    # Return 3 set of data
    return x_train, x_test, y_train, y_test


def preprocess_new(X_new):
    ''' This Function tries to process the new instances before predicted using Model
    Args:
    *****
        (X_new: 2D array) --> The Features in the same order
                ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                 'population', 'households', 'median_income', 'ocean_proximity']
        All Featutes are Numerical, except the last one is Categorical.

     Returns:
     *******
         Preprocessed Features ready to make inference by the Model
    '''

    config_data = utils.load_config("config/config.yaml")

    # 2. Load dataset
    x_train, x_test, y_train, y_test = load_dataset_prepnew(config_data)

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

    return total_pipeline.transform(X_new)
