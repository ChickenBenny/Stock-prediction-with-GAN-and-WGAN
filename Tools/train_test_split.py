import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def train_test_split(data, y_loc, windows, split_rate):
    x = data.iloc[:, :y_loc].values
    y = data.iloc[:, y_loc].values

    split = int(data.shape[0]* 0.8)
    train_x, test_x = x[: split, :], x[split - windows:, :]
    train_y, test_y = y[: split, ], y[split - windows:, ]

    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))

    train_x = x_scaler.fit_transform(train_x)
    test_x = x_scaler.transform(test_x)

    train_y = y_scaler.fit_transform(train_y.reshape(-1, 1))
    test_y = y_scaler.transform(test_y.reshape(-1, 1))

    print(f'train_x: {train_x.shape} train_y: {train_y.shape}')
    print(f'test_x: {test_x.shape} test_y: {test_y.shape}')
    
    return train_x, train_y, test_x, test_y, x_scaler, y_scaler