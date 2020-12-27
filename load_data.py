import pandas as pd
import os


def create_matrix(data_path):
    result_list = []
    file_names_test = os.listdir(data_path)
    for file_name in file_names_test:
        df_log = pd.read_csv(os.path.join(data_path, file_name))
        if df_log.iloc[1].isnull().sum() < 2:
            # feature = function for extract features (fillnans(0)
            # test_features.append(feature)
            result_list.append(file_name)
            if len(result_list) % 300 == 0:
                print(len(result_list))

    return result_list  # , matrix


path_local = 'C:/Users/Tair/Documents/CI/Project'
train_path = os.path.join(path_local, 'train')
test_path = os.path.join(path_local, 'test')
train_list, train_matrix = create_matrix(train_path)
test_list, test_matrix = create_matrix(test_path)

#
# print(len(test_list))
#
# mean_train =
# for line in test_features:
#     if line.isnull():
#
# def fill_NaNs():
