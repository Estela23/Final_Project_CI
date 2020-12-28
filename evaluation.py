from sklearn import metrics
import numpy as np


def MSE(y_true, y_pred, verbose=True):
    mse = metrics.mean_squared_error(y_true, y_pred)
    if verbose:
        print(mse)
    return mse


def rMSE(y_true, y_pred, verbose=True):
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    if verbose:
        print(rmse)
    return rmse


def MAE(y_true, y_pred, verbose=True):
    mae = metrics.mean_absolute_error(y_true, y_pred)
    if verbose:
        print(mae)
    return mae


def all_errors(y_true, y_pred, verbose=True):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    if verbose:
        print("MSE: ", mse, "\trMSE: ", rmse, "\tMAE: ", mae)
    return mse, rmse, mae
