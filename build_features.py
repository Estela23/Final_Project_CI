import os
import numpy as np
import pandas as pd


def build_features(data, row, sensor_id):
    df = pd.DataFrame()
    fourier_f = np.fft.fft(data)
    f_real = np.real(fourier_f)
    df.loc[row, f'{sensor_id}_sum'] = data.sum()
    df.loc[row, f'{sensor_id}_mean'] = data.mean()
    df.loc[row, f'{sensor_id}_std'] = data.std()
    df.loc[row, f'{sensor_id}_min'] = data.min()
    df.loc[row, f'{sensor_id}_max'] = data.max()
    df.loc[row, f'{sensor_id}_skew'] = data.skew()
    df.loc[row, f'{sensor_id}_kurtosis'] = data.kurtosis()
    df.loc[row, f'{sensor_id}_Q99'] = np.quantile(data, 0.99)
    df.loc[row, f'{sensor_id}_Q95'] = np.quantile(data, 0.95)
    df.loc[row, f'{sensor_id}_Q55'] = np.quantile(data, 0.50)
    df.loc[row, f'{sensor_id}_Q05'] = np.quantile(data, 0.05)
    df.loc[row, f'{sensor_id}_Q01'] = np.quantile(data, 0.01)
    df.loc[row, f'{sensor_id}_fft_real_mean'] = f_real.mean()
    df.loc[row, f'{sensor_id}_fft_real_std'] = f_real.std()
    df.loc[row, f'{sensor_id}_fft_real_min'] = f_real.min()
    df.loc[row, f'{sensor_id}_fft_real_max'] = f_real.max()

    return df


def create_matrix(data_path):
    result_list = []
    result_features = pd.DataFrame()
    file_names = os.listdir(data_path)
    for file_name in file_names:
        df_log = pd.read_csv(os.path.join(data_path, file_name))
        df = pd.DataFrame(columns=["segment_id"])
        df.loc[file_name] = [str(file_name).strip(".csv")]

        train_row = [df]
        for ids in range(10):
            sensor_id = f'sensor_{ids + 1}'
            train_row.append(build_features(df_log[sensor_id].fillna(0), file_name, sensor_id))
        train_row = pd.concat(train_row, axis=1)
        result_features = result_features.append(train_row)
        result_list.append(str(file_name).strip(".csv"))
        if len(result_list) % 10 == 0:
            print(len(result_list))

    return result_list, result_features


def save_data_to_csv(data_frame, path, name_file):
    data_frame.to_csv(rf'{path}/{name_file}', index=False, header=True)


train_path = "train"
test_path = "test"

train_list, train_matrix = create_matrix(train_path)

df_ground_truth = pd.read_csv("data/train.csv")
df_ground_truth['segment_id'] = df_ground_truth['segment_id'].astype(str)

print("Dataframe for the train data created")
result = pd.merge(train_matrix, df_ground_truth, on="segment_id")

save_data_to_csv(result, "C:/Users/Alberto/Desktop", "data/train_final_data_complete.csv")
print("Dataframe for the train data saved in a .csv")

test_list, test_matrix = create_matrix(test_path)
print("Dataframe for the test data created")

save_data_to_csv(test_matrix, "C:/Users/Alberto/Desktop", "data/test_final_data_complete.csv")
print("Dataframe for the test data saved in a .csv")
