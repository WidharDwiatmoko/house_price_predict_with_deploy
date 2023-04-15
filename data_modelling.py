from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

# sklearn -- metrics
from sklearn.metrics import mean_squared_error, r2_score

# sklearn -- Models
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# Xgboost
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import joblib
import numpy as np
import json
import pandas as pd
import copy
import hashlib
import matplotlib.pyplot as plt

import src.util as util

params = util.load_config()


def load_train_feng(params: dict) -> pd.DataFrame:
    # Load train set
    x_train = util.pickle_load(params["train_feng_set_path"][0])
    y_train = util.pickle_load(params["train_feng_set_path"][1])

    return x_train, y_train


def load_test_feng(params: dict) -> pd.DataFrame:
    # Load test set
    x_test = util.pickle_load(params["test_feng_set_path"][0])
    y_test = util.pickle_load(params["test_feng_set_path"][1])

    return x_test, y_test


def load_dataset(params: dict) -> pd.DataFrame:
    # Debug message
    util.print_debug("Loading dataset.")

    # Load train set
    x_train, y_train = load_train_feng(params)

    # Load test set
    x_test, y_test = load_test_feng(params)

    # Debug message
    util.print_debug("Dataset loaded.")

    # Return the dataset
    return x_train, y_train, x_test, y_test


def checkRMSEScore(x_train, y_train, lin_reg, modelname):
    # Check scores of this Model (RMSE) using (cross_val_score)
    rmse_scores_lin = cross_val_score(estimator=lin_reg, X=x_train, y=y_train, cv=5,
                                      scoring='neg_mean_squared_error', n_jobs=-1)  # sklearn deals with error as negative
    rmse_scores_lin = -1 * rmse_scores_lin  # we want it positive
    rmse_scores_lin = np.sqrt(rmse_scores_lin)
    print(
        f'RMSE Scores Using {modelname} --- {np.round(rmse_scores_lin, 4)}')
    print(
        f'Mean of RMSE Scores Using {modelname} --- {rmse_scores_lin.mean():.4f}')

    print('****'*30)

    # Get Prediction using (cross_val_predict)
    y_pred_lin = cross_val_predict(
        estimator=lin_reg, X=x_train, y=y_train, cv=5, method='predict', n_jobs=-1)
    # You can check the (RMSE) using what model predicts and compare it with the Mean of above result -- almost the same
    # take care of this point --> don't use .predict when you are using (crossValidation)
    rmse_pred_lin = np.sqrt(mean_squared_error(y_train, y_pred_lin))
    # almost the same result :D
    return(
        f'RMSE after prediction Using {modelname} --- {rmse_pred_lin:.4f}')


def doParamTunning(estimator):
    params_best_xgb = {'n_estimators': np.arange(100, 200, 50), 'max_depth': np.arange(4, 15, 2),
                       'learning_rate': [0.1, 0.2], 'subsample': [0.8, 0.9]}


# Intitalize the GridSearchCV and Fit ti Data
    grid_xgb = GridSearchCV(estimator=xgb_reg, param_grid=params_best_xgb, cv=5,
                            scoring='neg_mean_squared_error', n_jobs=-1, verbose=6)
    grid_xgb.fit(x_train, y_train)  # train
    return grid_xgb, params_best_xgb


def getBestParam(grid_xgb):
    best_xgb_params = grid_xgb.best_params_
    print('best_xgb_params -- ', best_xgb_params)

    with open(params["best_xgb_param_log_path"], "w") as outfile:
        json.dump(best_xgb_params, outfile, default=str)


# Get the best estimator
    best_xgb = grid_xgb  # predict using this Model
    print('best_xgb -- ', best_xgb)

    with open(params["best_xgb_estimator_log_path"], "w") as outfile:
        json.dump(best_xgb, outfile, default=str)

    return best_xgb


def dump_production_model(best_xgb):
    util.pickle_dump(best_xgb, params["production_model_path"])


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset(params)

    # Lin Reg:
    lin_reg = LinearRegression()
    print("Train linear regression")
    lin_reg.fit(x_train, y_train)  # train
    print("Train linear regression done!")
    checkRMSEScore(x_train, y_train, lin_reg, "Linear Regression")

    # Ridge:
    ridge_reg = Ridge(alpha=0.9, solver='cholesky')
    print("Train Ridge")
    ridge_reg.fit(x_train, y_train)  # train the model
    print("Train Ridge done!")
    checkRMSEScore(x_train, y_train, ridge_reg, "Ridge")

    # lasso:
    lasso_reg = Lasso(alpha=1, max_iter=100000)
    print("Train Lasso")
    # train the model   and try predictions in the same way
    lasso_reg.fit(x_train, y_train)
    print("Train Lasso done")
    print("Here are Lasso coeficient")
    print(lasso_reg.coef_)

    # KNN:
    knn_reg = KNeighborsRegressor(
        n_neighbors=8, p=2,  metric='minkowski', weights='uniform')
    print("Train KNN")
    knn_reg.fit(x_train, y_train)  # train the model
    print("Train KNN Done!")
    checkRMSEScore(x_train, y_train, knn_reg, "KNN")

    # RF:
    forest_reg = RandomForestRegressor(
        n_estimators=150, max_depth=6, max_samples=0.8, random_state=42, n_jobs=-1)
    print("Train RF")
    forest_reg.fit(x_train, y_train)  # train
    print("Train RF Done!")
    checkRMSEScore(x_train, y_train, forest_reg, "Random Forest")

    # XGboost:
    xgb_reg = XGBRegressor(n_estimators=100, max_depth=6,
                           learning_rate=0.1, subsample=0.8)
    print("Train XGB")
    xgb_reg.fit(x_train, y_train)
    print("Train XGB Done!")
    checkRMSEScore(x_train, y_train, xgb_reg, "XGBoost")

    # Tuning Hyperparameter (only the best model with low RMSE -> Based on previous explore)
    print("Process Hyperparam Tunning!")
    grid_xgb, params_best_xgb = doParamTunning(estimator=xgb_reg)
    print("Process Hyperparam Tunning Done")

    # Retrain with best param
    print("Implement parameters Hyperparam tunning!")
    best_xgb = getBestParam(grid_xgb=grid_xgb)
    print("Implement parameters Hyperparam tunning done")

    # Final RMSE SCORE after hypeparam tunning and retrain with best param
    # checkRMSEScore(x_train, y_train, best_xgb, "XGBoost with Tunning")

    print("freeze model to production stage")
    dump_production_model(best_xgb)


# Define Ridge Model (Regularized Version of LinearRegression)
# ridge_reg = Ridge(alpha=0.9, solver='cholesky')
# ridge_reg.fit(x_train, y_train)  ## train the model
