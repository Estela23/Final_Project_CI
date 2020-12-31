from lightgbm import LGBMRegressor
from sklearn import model_selection, metrics
from sklearn.model_selection import KFold
import evaluation
# import tensorflow as tf
import pandas as pd
import os
import numpy as np
from aplying_nn import load_test_data


def apply_LGBM_regression(train_data, y_train, validation_data, y_val):
    lgb = LGBMRegressor(random_state=100, n_estimators=2000)  # ,max_depth=7,n_estimators=250,learning_rate=0.12
    lgb.fit(train_data, y_train)
    prediction = lgb.predict(validation_data)
    return prediction


# path_local = 'C:/Users/Tair/Documents/MAI Semester1/CI/Project'
file_name = 'train_final_data_complete.csv'
df_data = pd.read_csv(file_name)
#  feature selection
df_reduced_data = df_data.drop(columns=["segment_id"])  # for debug only!!!!!!!!!!!!!!!!!!!!
train, val, y, y_val = model_selection.train_test_split(df_reduced_data[df_reduced_data.columns[:-1]],
                                                        df_reduced_data[df_reduced_data.columns[-1]],
                                                        test_size=0.2, shuffle=True)

# apply LGBM
preds = apply_LGBM_regression(train, y, val, y_val)
mse, rmse, mae = evaluation.all_errors(y_val, preds)


# apply NN
# preds_nn = apply_NN(train, y, val)
# evaluation.rMSE(y_val, preds_nn)


def apply_kfold_LGBM(number_of_folds, train, y, validation_set, y_val):
    kfold = KFold(n_splits=number_of_folds, random_state=1, shuffle=True).split(y)
    predictions = []
    models = []

    for i, (train_i, test_i) in enumerate(kfold):
        print(f'Fold {i}')
        model = LGBMRegressor(random_state=100, n_estimators=2000)  # ,max_depth=7,n_estimators=250,learning_rate=0.12
        model.fit(train.values[train_i], y.values[train_i])
        pred = model.predict(validation_set)
        evaluation.all_errors(y_val, pred)
        predictions.append(pred)
        # print('Fold rmse', rmse(y.values[test_i], model.predict(train.values[test_i])))
        models.append(model)

    # res_predictions = predictions[0]
    # for i in range(1, number_of_folds):
    #     res_predictions += predictions[i]
    # res_predictions /= number_of_folds
    #
    # print('LGBM rmse', rmse(y_val, res_predictions))

    return models


def test_LGBM(train_data, validation_data, y_train, y_validation, folds):
    test_set = load_test_data("test_final_data_complete.csv")

    models = apply_kfold_LGBM(folds, train_data, y_train, validation_data, y_validation)  # create models using k-fold

    # predict in test set with all models
    predictions = []
    for model in models:
        pred = model.predict(test_set)
        predictions.append(pred)

    # do the average of the multiple predictions
    res_preds = predictions[0]
    for i in range(1, folds):
        res_preds += predictions[i]
    res_preds /= folds

    # export results as submission file
    sample_submission = pd.read_csv('sample_submission.csv')

    test_set['time_to_eruption'] = res_preds
    sample_submission = pd.merge(sample_submission, test_set[['segment_id', 'time_to_eruption']], on='segment_id')
    sample_submission = sample_submission.drop(['time_to_eruption_x'], axis=1)
    sample_submission.columns = ['segment_id', 'time_to_eruption']

    sample_submission.to_csv('results/first_submission_LGBM.csv', index=False)


test_LGBM(train, val, y, y_val, folds=10)
