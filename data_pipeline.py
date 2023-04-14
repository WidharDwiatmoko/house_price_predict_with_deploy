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
