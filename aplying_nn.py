import math
import pandas as pd
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error as mse
from tensorflow.python.keras.callbacks import EarlyStopping


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


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input((160,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1000, activation="sigmoid"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.6),
        tf.keras.layers.Dense(1, activation='relu')
    ])

    model.compile(
        loss=root_mean_squared_error,
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
    )
    return model


def apply_kfold(number_of_folds, train, yy, validation_set, y_val):
    kfold = KFold(n_splits=number_of_folds, random_state=1, shuffle=True).split(yy)
    predictions = []
    models = []

    for i, (train_i, test_i) in enumerate(kfold):
        early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=0)

        print(f'Fold {i}')

        model = create_model()

        model.fit(
            train.values[train_i],
            yy.values[train_i],
            epochs=4000,
            batch_size=128,
            verbose=1,
            callbacks=[early_stopping]
        )

        pred = model.predict(validation_set)
        pred = np.expm1(pred).reshape((pred.shape[0],))
        predictions.append(pred)
        print('Fold rmse', rmse(yy.values[test_i], model.predict(train.values[test_i])))
        models.append(model)

    res_predictions = predictions[0]
    for i in range(1, number_of_folds):
        res_predictions += predictions[i]
    res_predictions /= number_of_folds

    print('NN rmse', rmse(y_val, res_predictions))

    return models


def test():
    folds = 2
    train, y = load_train_data("train_final_data.csv")
    train, val, y, y_val = train_test_split(train, y, random_state=1, test_size=0.2, shuffle=True)
    yy = np.log1p(y)  # to handle precision problems

    test_set = load_test_data("test_final_data.csv")

    models = apply_kfold(folds, train, yy, val, y_val)  # create models using k-fold

    # predict in test set with all models
    predictions = []
    for model in models:
        pred = model.predict(test_set)
        pred = np.expm1(pred).reshape((pred.shape[0],))
        predictions.append(pred)

    # do the average of the multiple predictions
    res_preds = predictions[0]
    for i in range(1, folds):
        res_preds += predictions[i]
    res_preds /= folds

    # export results as submission file
    test_set = pd.read_csv("test_final_data.csv")
    sample_submission = pd.read_csv('sample_submission.csv')

    test_set['time_to_eruption'] = res_preds
    sample_submission = pd.merge(sample_submission, test_set[['segment_id', 'time_to_eruption']], on='segment_id')
    sample_submission = sample_submission.drop(['time_to_eruption_x'], axis=1)
    sample_submission.columns = ['segment_id', 'time_to_eruption']

    sample_submission.to_csv('submission.csv', index=False)


test()
