# ## Major Libraries
# import pandas as pd
# import os
# ## sklearn -- for pipeline and preprocessing
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.pipeline import Pipeline, FeatureUnion
# from sklearn_features.transformers import DataFrameSelector


# ## Read the CSV file using pandas
# FILE_PATH = os.path.join(os.getcwd(), 'housing.csv')
# df_housing = pd.read_csv(FILE_PATH)

# ## Replace the  (<1H OCEAN) to (1H OCEAN) -- will cause ane errors in Deploymnet
# df_housing['ocean_proximity'] = df_housing['ocean_proximity'].replace('<1H OCEAN', '1H OCEAN')

# ## Try to make some Feature Engineering --> Feature Extraction --> Add the new column to the main DF
# df_housing['rooms_per_household'] = df_housing['total_rooms'] / df_housing['households']
# df_housing['bedroms_per_rooms'] = df_housing['total_bedrooms'] / df_housing['total_rooms']
# df_housing['population_per_household'] = df_housing['population'] / df_housing['households']


# ## Split the Dataset -- Taking only train to fit (the same the model was trained on)
# X = df_housing.drop(columns=['median_house_value'], axis=1)   ## Features
# y = df_housing['median_house_value']   ## target

# ## the same Random_state (take care)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=True, random_state=42)

# ## Separete the columns according to type (numerical or categorical)
# num_cols = [col for col in X_train.columns if X_train[col].dtype in ['float32', 'float64', 'int32', 'int64']]
# categ_cols = [col for col in X_train.columns if X_train[col].dtype not in ['float32', 'float64', 'int32', 'int64']]

# ## We can get much much easier like the following
# ## numerical pipeline
# num_pipeline = Pipeline([
#                         ('selector', DataFrameSelector(num_cols)),    ## select only these columns
#                         ('imputer', SimpleImputer(strategy='median')),
#                         ('scaler', StandardScaler())
#                         ])

# ## categorical pipeline
# categ_pipeline = Pipeline(steps=[
#             ('selector', DataFrameSelector(categ_cols)),    ## select only these columns
#             ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
#             ('OHE', OneHotEncoder(sparse=False))])

# ## concatenate both two pipelines
# total_pipeline = FeatureUnion(transformer_list=[
#                                             ('num_pipe', num_pipeline),
#                                             ('categ_pipe', categ_pipeline)
#                                                ]
#                              )

# X_train_final = total_pipeline.fit_transform(X_train) ## fit

# def preprocess_new(X_new):
#     ''' This Function tries to process the new instances before predicted using Model
#     Args:
#     *****
#         (X_new: 2D array) --> The Features in the same order
#                 ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
#                  'population', 'households', 'median_income', 'ocean_proximity']
#         All Featutes are Numerical, except the last one is Categorical.

#      Returns:
#      *******
#          Preprocessed Features ready to make inference by the Model
#     '''
#     return total_pipeline.transform(X_new)

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
import os
import yaml
import util as util


def read_raw_data(config: dict) -> pd.DataFrame:
    # Create variable to store raw dataset
    raw_dataset = pd.DataFrame()

    # Raw dataset dir
    raw_dataset_dir = config["raw_dataset_dir"]

    # Look and load add CSV files
    for i in tqdm(os.listdir(raw_dataset_dir)):
        raw_dataset = pd.concat(
            [pd.read_csv(raw_dataset_dir + i, delimiter=';'), raw_dataset])

    # Return raw dataset
    return raw_dataset


def removeDuplicates(data):

    # Drop duplicate
    data = data.drop_duplicates()
    return data


def splitInputOtput(data, config_data):
    x = data[config_data["predictors"]].copy()
    y = data[config_data["label"]].copy()
    return x, y


def splitTrainTest(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test


def dumpData(x_train, y_train, x_test, y_test, config_data):
    util.pickle_dump(x_train, config_data["train_set_path"][0])
    util.pickle_dump(y_train, config_data["train_set_path"][1])

    util.pickle_dump(x_test, config_data["test_set_path"][0])
    util.pickle_dump(y_test, config_data["test_set_path"][1])


if __name__ == "__main__":
    # 1. Load configuration file
    config_data = util.load_config()

    # 2. Read raw dataset
    raw_dataset = read_raw_data(config_data)

    # 3. Remove duplicates data
    raw_dataset = removeDuplicates(raw_dataset)

    raw_dataset['rooms_per_household'] = raw_dataset['total_rooms'] / \
        raw_dataset['households']
    raw_dataset['bedroms_per_rooms'] = raw_dataset['total_bedrooms'] / \
        raw_dataset['total_rooms']
    raw_dataset['population_per_household'] = raw_dataset['population'] / \
        raw_dataset['households']

    # Replace the  (<1H OCEAN) to (1H OCEAN) -- will cause ane errors in Deploymnet
    raw_dataset['ocean_proximity'] = raw_dataset['ocean_proximity'].replace(
        '<1H OCEAN', '1H OCEAN')

    # 6. Split input output
    x, y = splitInputOtput(raw_dataset, config_data)

    # 7. Split Train Test
    x_train, x_test, y_train, y_test = splitTrainTest(x, y)

    # 9. Dump data to pickled
    dumpData(x_train, y_train, x_test, y_test, config_data)
