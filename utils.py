import os
import json
import glob
import json
import logging
import numpy as np
from sklearn.linear_model import *
from sklearn.model_selection import train_test_split, StratifiedKFold


Logger = logging.getLogger(__name__)


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
    Logger.info(f'Load data from {fpath}')
    data_xy = np.load(fpath, allow_pickle=True)
    dataX = data_xy['dataX']
    dataY = data_xy['dataY']

    label_list = sorted(np.unique(dataY))
    label_map = {val:i for i, val in enumerate(label_list)}
    rev_label_map = {i:val for i, val in enumerate(label_list)}
    Logger.info(f'Label list: {label_list}')
    Logger.info(f'Label map: {label_map}')

    dataX = np.nan_to_num(dataX)
    dataY = np.array([label_map[label] for label in dataY])

    Logger.info('Data size: {dataX.shape}, {dataY.shape}')
    return dataX, dataY, label_list


def split_data(fpath, exp_path, debug=False):
    X, Y, label_map = load_data(fpath)

    if debug:
        # run with only 1k features
        Logger.info('*'*40)
        Logger.info('Running SANITY TEST, with only 1000 features')
        Logger.info('*'*40)
        X = X[:, :1000]

    Logger.info('Dump label map')
    with open(f'{exp_path}/labelmap.json', 'w') as f:
        json.dump(label_map, f)

    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.33, random_state=42, stratify=Y)
    Logger.info(f'Data size: train {Xtrain.shape} test {Xtest.shape}')

    return Xtrain, Xtest, Ytrain, Ytest, label_map


def create_exp_path(fpath, debug=False):
    fname = fpath.split('/')[-1].split('.')[0]
    exps = glob.glob(f'experiments/{fname}/*/')
    if debug:
        exp_path = f'experiments/{fname}/{len(exps)}_debug'
    else:
        exp_path = f'experiments/{fname}/{len(exps)}'
    os.system(f'mkdir -p {exp_path}')
    Logger.info(f'Save experiment to {exp_path}')

    # logging to file
    fh = logging.FileHandler(f'{exp_path}/run_logs.txt', )
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logging.getLogger('').addHandler(fh)

    return exp_path, fname
