from lightgbm import LGBMRegressor
from sklearn import model_selection, metrics
from sklearn.model_selection import KFold
import evaluation
import pandas as pd
import os
import numpy as np
from FeatureSelection import FeatureSelectionUsingCorrelation, FeatureSelectionWrapper, FeatureSelectionEmbedded


def apply_kfold_LGBM(number_of_folds, train, y, validation_set, y_val):
    kfold = KFold(n_splits=number_of_folds, random_state=1, shuffle=True).split(y)
    models = []
    for i, (train_i, test_i) in enumerate(kfold):
        print(f'Fold {i}')
        model = LGBMRegressor(random_state=100)
        model.fit(train.values[train_i], y.values[train_i])
        pred = model.predict(validation_set)
        evaluation.all_errors(y_val, pred)
        models.append(model)
    return models


def test_LGBM(train_data, validation_data, y_train, y_validation, folds, output_file_name, reduced_technique):
    test_set = pd.read_csv("test_final_data_complete.csv")
    if reduced_technique == 1:
        test_set_reduced = test_set[list_features_corr[:-1]]
        test_set_reduced["segment_id"] = test_set["segment_id"]
    elif reduced_technique == 2:
        test_set_reduced = test_set[list_features_wrapper]
        test_set_reduced["segment_id"] = test_set["segment_id"]
    else:
        test_set_reduced = test_set

    models = apply_kfold_LGBM(folds, train_data, y_train, validation_data, y_validation)  # create models using k-fold

    # predict in test set with all models
    predictions = []
    for model in models:
        pred = model.predict(test_set_reduced.drop("segment_id", 1))
        predictions.append(pred)

    # average of the multiple predictions
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

    sample_submission.to_csv(f'results/{output_file_name}.csv', index=False)


df_data = pd.read_csv("train_final_data_complete.csv")
df_data = df_data.drop(columns=["segment_id"])

# Original Data split
train, val, y, y_val = model_selection.train_test_split(df_data[df_data.columns[:-1]],
                                                        df_data[df_data.columns[-1]],
                                                        test_size=0.2, shuffle=True)

# Correlation Reduced Data split
list_features_corr, df_reduced_corr_data = FeatureSelectionUsingCorrelation(df_data, 0.05)

corr_train, corr_val, corr_y, corr_y_val = model_selection.train_test_split(df_reduced_corr_data[df_reduced_corr_data.columns[:-1]],
                                                                            df_reduced_corr_data[df_reduced_corr_data.columns[-1]],
                                                                            test_size=0.2, shuffle=True)

# Wrapper Reduced Data split
list_features_wrapper, df_reduced_wrapper_data = FeatureSelectionWrapper(df_data, 0.5)

wrapper_train, wrapper_val, wrapper_y, wrapper_y_val = model_selection.train_test_split(df_reduced_wrapper_data[df_reduced_wrapper_data.columns[:-1]],
                                                                                        df_reduced_wrapper_data[df_reduced_wrapper_data.columns[-1]],
                                                                                        test_size=0.2, shuffle=True)

# Final predictions
print("LGBM with original data")
test_LGBM(train, val, y, y_val, folds=10, output_file_name="original_data_submission_LGBM", reduced_technique=0)
print("LGBM with correlation reduced data")
test_LGBM(corr_train, corr_val, corr_y, corr_y_val, folds=10, output_file_name="correlation_reduced_data_submission_LGBM", reduced_technique=1)
print("LGBM with wrapper reduced data")
test_LGBM(wrapper_train, wrapper_val, wrapper_y, wrapper_y_val, folds=10, output_file_name="wrapper_reduced_data_submission_LGBM", reduced_technique=2)
