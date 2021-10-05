import numpy as np
from sklearn.linear_model import *


def lasso_clf():
    clf = LogisticRegressionCV(
        Cs=[0.01],#, 0.02, 0.03],
        cv=10,
        penalty='l1',
        solver='saga',
        max_iter=10,
        n_jobs=-1,
        verbose=1,
        multi_class='multinomial',
        random_state=42,
    )
    return clf


def load_data(fpath):
    print(f'Load data from {fpath}')
    data_xy = np.load(fpath)
    dataX = data_xy['dataX']
    dataY = data_xy['dataY']

    label_list = sorted(np.unique(dataY))
    label_map = {val:i for i, val in enumerate(label_list)}
    print(f'Label list:', label_list)
    print(f'Label map: {label_map}')

    dataX = np.nan_to_num(dataX)
    dataY = np.array([label_map[label] for label in dataY])

    print('Data size:', dataX.shape, dataY.shape)
    return dataX, dataY, label_map


if __name__ == '__main__':
    data_adjacent_xy = './new_data/data_adjacent_xy.npz'
    X, Y, label_map = load_data(data_adjacent_xy)
