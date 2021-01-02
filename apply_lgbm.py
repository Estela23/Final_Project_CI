from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
import evaluation
import pandas as pd


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


def apply_LGBM(train_data, validation_data, y_train, y_validation, test, folds=10):
    # create models using k-fold
    models = apply_kfold_LGBM(folds, train_data, y_train, validation_data, y_validation)

    # predict in test set with all models
    predictions = []
    for model in models:
        pred = model.predict(test)
        predictions.append(pred)

    # average of the multiple predictions
    res_preds = predictions[0]
    for i in range(1, folds):
        res_preds += predictions[i]
    res_preds /= folds

    # export results as submission file
    sample_submission = pd.read_csv('data/sample_submission.csv')
    test_set = pd.read_csv("data/test_final_data_complete.csv")

    test_set['time_to_eruption'] = res_preds
    sample_submission = pd.merge(sample_submission, test_set[['segment_id', 'time_to_eruption']], on='segment_id')
    sample_submission = sample_submission.drop(['time_to_eruption_x'], axis=1)
    sample_submission.columns = ['segment_id', 'time_to_eruption']

    sample_submission.to_csv(f'results/submission.csv', index=False)
