import math
import pandas as pd
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error as mse
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.wrappers.scikit_learn import KerasRegressor


def load_train_data(file_name: str):  # 'train_final_data.csv'
    train_set = pd.read_csv(file_name)
    train = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
    y = train_set['time_to_eruption']

    return train, y


def load_test_data(file_name: str):
    test_set = pd.read_csv(file_name)
    test_set = test_set.drop(['segment_id'], axis=1)

    return test_set


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def create_model(learn_rate=0.001, neurons=1, dropout=0.0):
    model = tf.keras.Sequential([
        tf.keras.layers.Input((160,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(neurons, activation="sigmoid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    model.compile(
        loss=root_mean_squared_error,
        optimizer=tf.keras.optimizers.Adamax(learning_rate=learn_rate, name="Adamax")
    )
    return model


def apply_kfold(number_of_folds, train, yy, validation_set, y_val):
    kfold = KFold(n_splits=number_of_folds, random_state=1, shuffle=True).split(yy)
    predictions = []
    models = []

    for i, (train_i, test_i) in enumerate(kfold):
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0)

        print(f'Fold {i}')

        model = create_model(neurons=500, dropout=0.3)

        model.fit(
            train.values[train_i],
            yy.values[train_i],
            epochs=4000,
            batch_size=128,
            verbose=1,
            callbacks=[early_stopping]
        )

        pred = model.predict(validation_set)
        predictions.append(pred)
        print('Fold rmse', rmse(yy.values[test_i], model.predict(train.values[test_i])))
        models.append(model)

    res_predictions = predictions[0]
    for i in range(1, number_of_folds):
        res_predictions += predictions[i]
    res_predictions /= number_of_folds

    print('NN rmse', rmse(y_val, res_predictions))

    return models

"""
Best: -1.060633 using {'optimizer': 'Adamax'}
-1.103667 (0.038624) with: {'optimizer': 'SGD'}
-1.149151 (0.094356) with: {'optimizer': 'RMSprop'}
-1.249073 (0.084215) with: {'optimizer': 'Adagrad'}
-10.794811 (0.978465) with: {'optimizer': 'Adadelta'}
-1.093858 (0.045738) with: {'optimizer': 'Adam'}
-1.060633 (0.054592) with: {'optimizer': 'Adamax'}
-1.092085 (0.052264) with: {'optimizer': 'Nadam'}

Best: -0.795547 using {'dropout': 0.3, 'learn_rate': 0.001, 'neurons': 500}
NN rmse 3930.734834209182
10 folds: NN rmse 3397.546502484067
"""

def grid_search_cv(X, Y):

    model = KerasRegressor(build_fn=create_model, epochs=500, batch_size=50, verbose=1)
    # optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    learn_rate = [0.001, 0.01, 0.1, 0.3]
    dropout_rate = [0.3, 0.5, 0.7, 0.9]
    neurons = [100, 250, 500, 1000]
    param_grid = dict(learn_rate=learn_rate, dropout=dropout_rate, neurons=neurons)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    grid_result = grid.fit(X, Y)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def test():
    folds = 10
    train, y = load_train_data("train_final_data_complete.csv")
    train, val, y, y_val = train_test_split(train, y, random_state=1, test_size=0.2, shuffle=True)

    test_set = load_test_data("test_final_data_complete.csv")

    y = y.astype(np.float32)
    # grid_search_cv(train, yy)

    models = apply_kfold(folds, train, y, val, y_val)  # create models using k-fold

    # predict in test set with all models
    predictions = []
    for model in models:
        pred = model.predict(test_set)
        # pred = np.expm1(pred).reshape((pred.shape[0],))
        predictions.append(pred)

    # do the average of the multiple predictions
    res_preds = predictions[0]
    for i in range(1, folds):
        res_preds += predictions[i]
    res_preds /= folds

    # export results as submission file
    test_set = pd.read_csv("test_final_data_complete.csv")
    sample_submission = pd.read_csv('sample_submission.csv')

    test_set['time_to_eruption'] = res_preds
    sample_submission = pd.merge(sample_submission, test_set[['segment_id', 'time_to_eruption']], on='segment_id')
    sample_submission = sample_submission.drop(['time_to_eruption_x'], axis=1)
    sample_submission.columns = ['segment_id', 'time_to_eruption']

    sample_submission.to_csv('submission.csv', index=False)


test()
