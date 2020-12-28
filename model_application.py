from lightgbm import LGBMRegressor
from sklearn import model_selection
import evaluation
# import tensorflow as tf
import numpy as np
import pandas as pd
import os


def apply_LGBM_regression(train_data, y_train, validation_data):
    lgb = LGBMRegressor(random_state=100)  # ,max_depth=7,n_estimators=250,learning_rate=0.12
    lgb.fit(train_data, y_train)
    prediction = lgb.predict(validation_data)
    return prediction


# def NN():
#     model = tf.keras.Sequential([
#         tf.keras.layers.Input((241,)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(1000, activation="sigmoid"),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.6),
#         tf.keras.layers.Dense(1, activation='relu')
#     ])
#
#     model.compile(
#         loss=rMES,
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001)
#     )
#     return model


path_local = 'C:/Users/Tair/Documents/MAI Semester1/CI/Project'
file_name = 'train_final_data.csv'
df_data = pd.read_csv(os.path.join(path_local, file_name))
#  feature selection
df_reduced_data = df_data  # for debug only!!!!!!!!!!!!!!!!!!!!
train, val, y, y_val = model_selection.train_test_split(df_reduced_data[df_reduced_data.columns[:-1]],
                                                        df_reduced_data[df_reduced_data.columns[-1]])
preds = apply_LGBM_regression(train, y, val)
evaluation.rMES(y_val, preds)
