import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import util as util
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn_features.transformers import DataFrameSelector
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def load_dataset(config_data: dict) -> pd.DataFrame:
    # Load every set of data
    x_train = util.pickle_load(config_data["train_set_path"][0])
    y_train = util.pickle_load(config_data["train_set_path"][1])

    x_test = util.pickle_load(config_data["test_set_path"][0])
    y_test = util.pickle_load(config_data["test_set_path"][1])

    # Concatenate x and y each set
    train_set = pd.concat([x_train, y_train], axis=1)
    test_set = pd.concat([x_test, y_test], axis=1)

    # Return 3 set of data
    return x_train, x_test, y_train, y_test


def dump_data(x_train, y_train, X_test_feng, y_test):
    util.pickle_dump(x_train, "../data/processed/x_train_feng.pkl")
    util.pickle_dump(y_train, "../data/processed/y_train_feng.pkl")

    # util.pickle_dump(valid_set.drop(columns="y"),
    #                  "data/processed/x_valid_feng.pkl")
    # util.pickle_dump(valid_set.y, "data/processed/y_valid_feng.pkl")

    util.pickle_dump(X_test_feng, "../data/processed/x_test_feng.pkl")
    util.pickle_dump(y_test, "../data/processed/y_test_feng.pkl")


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

    config_data = util.load_config()

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

    return total_pipeline.transform(X_new)


if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

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
