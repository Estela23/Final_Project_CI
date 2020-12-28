import pandas as pd
import os
from build_features import build_features


def create_matrix(data_path):
    result_list = []
    result_features = pd.DataFrame()
    file_names = os.listdir(data_path)
    for file_name in file_names:
        df_log = pd.read_csv(os.path.join(data_path, file_name))
        if df_log.iloc[1].isnull().sum() < 2:
            train_row = list(pd.DataFrame('file', file_name.strip('.csv')))
            for ids in range(10):
                sensor_id = f'sensor_{ids + 1}'
                train_row.append(build_features(df_log[sensor_id].fillna(0), file_name, sensor_id))
            train_row = pd.concat(train_row, axis=1)
            result_features = result_features.append(train_row)
            result_list.append(file_name)
            if len(result_list) % 300 == 0:
                print(len(result_list))
    return result_list, result_features


def save_data_to_csv(data_frame, path, name_file):
    data_frame.to_csv(rf'{path}/{name_file}', index=False, header=True)


# path_local = 'C:/Users/Tair/Documents/CI/Project'
path_local = 'C:/Users/Estela/Desktop/MAI/MAI 20-21/(CI) Inteligencia Computacional/FinalProject/predict-volcanic-eruptions-ingv-oe'
train_path = os.path.join(path_local, 'train')
test_path = os.path.join(path_local, 'test')

train_list, train_matrix = create_matrix(train_path)
df_ground_truth = pd.read_csv(os.path.join(path_local, 'train.csv'))
rslt_df = dataframe[dataframe['Percentage'] > 80]

ground_truth = [for x in df_ground_truth["segment_id"] if x in train_list]
train_matrix.append(df_ground_truth["time_to_eruption"], axis=1)
print("Dataframe for the train data created")
save_data_to_csv(train_matrix, path_local, "train_final_data.csv")
print("Dataframe for the train data saved in a .csv")

test_list, test_matrix = create_matrix(test_path)
print("Dataframe for the test data created")
save_data_to_csv(test_matrix, path_local, "test_final_data.csv")
print("Dataframe for the test data saved in a .csv")
