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
