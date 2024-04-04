import pandas as pd
import os
import numpy as np
from tqdm import trange
from model import MyLinearRegression
from random import random


def train_test_split(X, y, test_size=0.1):
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (numpy.ndarray): The input features.
        y (numpy.ndarray): The target variable.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.1.

    Returns:
        X_train (numpy.ndarray): The training input features.
        X_test (numpy.ndarray): The testing input features.
        y_train (numpy.ndarray): The training target variable.
        y_test (numpy.ndarray): The testing target variable.
    """
    hist, bin_adges = np.histogram(y, bins='auto')

    train_mask = []
    test_mask = []
    for i, edge in enumerate(bin_adges):
        if i == 0:
            continue
        indices, = np.where((bin_adges[i - 1] <= y) & (y < edge))
        num_test = int(len(indices) * test_size)
        np.random.shuffle(indices)
        test_indices = indices[:num_test]
        train_indices = indices[num_test:]

        train_mask += list(train_indices)
        test_mask += list(test_indices)

    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]

    return X_train, X_test, y_train, y_test


def mean_absolute_error(actual, pred):
    return np.mean(np.abs(actual - pred))


df = pd.concat([pd.read_csv(
    os.path.join('data', p)
) for p in os.listdir('data')])  # .sample(2_000)


def replace_year(string):
    y = string[6:10]
    y = str(int(y) - 543)
    return string[:6] + y + string[10:]


df.loc[:, 'time'] = pd.to_datetime(
    df['วัน-เวลา'].apply(replace_year), format='%d/%m/%Y %H:%M')

df = df.sort_values('time')

grouped = df.groupby(
    ['เขต', pd.Grouper(key="time", freq='30min')]
)['ฝน 30 นาที'].max()

prev_time_len = 5
chunks = []
for i in trange(len(grouped.index)-prev_time_len, desc='Building data chunks'):
    if not grouped[i:i+prev_time_len].isna().any():
        indices = grouped[i:i+prev_time_len].index
        chunks.append(
            grouped[indices].values
        )

print('Total data points is', len(chunks))

data = np.array(chunks)

X = data[:, :prev_time_len-1]
y = data[:, prev_time_len-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

my_reg = MyLinearRegression(0.0003).fit(X_train, y_train, iteration=8000)

print('Mean absolute error', mean_absolute_error(my_reg.predict(X_test), y_test))
