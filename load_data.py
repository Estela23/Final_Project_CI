import pandas as pd
import numpy as np
import os
from build_features import build_features


def create_matrix(data_path):
    result_list = []
    result_features = []
    file_names_test = os.listdir(data_path)
    for file_name in file_names_test:
        df_log = pd.read_csv(os.path.join(data_path, file_name))
        if df_log.iloc[1].isnull().sum() < 2:
            train_row = []
            for ids in range(10):
                sensor_id = f'sensor_{ids + 1}'
                train_row.append(build_features(df_log[sensor_id].fillna(0), file_name, sensor_id))
            train_row = pd.concat(train_row, axis=1)
            result_features.append(train_row)
            result_list.append(file_name)
            if len(result_list) % 300 == 0:
                print(len(result_list))

    return result_list, result_features


def save_data_to_csv(data_frame, path):
    data_frame.to_csv(rf'{path}', index=False, header=True)


# path_local = 'C:/Users/Tair/Documents/CI/Project'
path_local = 'C:/Users/Estela/Desktop/MAI/MAI 20-21/(CI) Inteligencia Computacional/FinalProject/predict-volcanic-eruptions-ingv-oe'
train_path = os.path.join(path_local, 'train1')
test_path = os.path.join(path_local, 'test1')
train_list, train_matrix = create_matrix(train_path)
#save_data_to_csv(train_matrix, path_local)
test_list, test_matrix = create_matrix(test_path)


#
# print(len(test_list))
#
# mean_train =
# for line in test_features:
#     if line.isnull():
#
# def fill_NaNs():
