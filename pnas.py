import os
import sys
import json
import glob
import logging
import sklearn

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utils import load_data
from lasso import LASSO
from ttest import TtestPreprocessor
from moderated_ttest import R_Preprocessor


LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(level=LOGLEVEL)
Logger = logging.getLogger(__name__)


class PNAS:
    '''
    Run whole PNAS pipeline
    '''

    def __init__(self, exp_path, preprocessor, lasso_model, clf_model):
        self.exp_path = exp_path
        self.preprocessor = preprocessor
        self.lasso_model = lasso_model
        self.clf_model = clf_model

    def _train_clf(self, dataX, dataY):
        Logger.debug('Start training classification model')

        self.clf_model.fit(dataX, dataY)
        acc = self.clf_model.score(dataX, dataY)
        Logger.info(f'Classification model training Acc: {acc}')

    def _save_markers(self):
        with open(f'{self.exp_path}/final_marker_to_draw.npy', 'wb') as f:
            np.save(f, self.lasso_model.final_mask)
        Logger.info('Final markers saved')

    def run_pipeline(self, dataX, dataY):
        '''Run whole PNAS pipeline, from preprocessing to classification.
        Pipeline:
            - Preprocess: run moderated-ttest to remove excessive noise
            - Lasso: run lasso model to select markers
            - Classification: train a clf model and evaluate above markers

        Input: Data from training set
            dataX.shape == [n_samples, n_markers]
            dataY.shape == [n_samples]
        Output: List markers
            output.shape == [n_classes, n_markers]
            where 1 mean selected feature and 0 is not.

        '''
        dataX, dataY = self.preprocessor(dataX, dataY)
        dataX, dataY = self.lasso_model(dataX, dataY)
        self._train_clf(dataX, dataY)
        self._save_markers()
        return self.clf_model, self.lasso_model.final_mask

    def score(self, Xtest, Ytest, labels=None):
        Logger.info(f'Testing data size {Xtest.shape}')
        score = self.clf_model.score(Xtest, Ytest)
        Logger.info(f'Testing accuracy {score}')

        preds = self.clf_model.predict(Xtest)
        reports = sklearn.metrics.classification_report(Ytest, preds)
        Logger.info(reports)
        with open(f'{self.exp_path}/reports.txt', 'w') as f:
            print(reports, file=f)

        disp = ConfusionMatrixDisplay.from_predictions(Ytest, preds,
                                                       display_labels=labels,
                                                       xticks_rotation='vertical')
        disp.plot(xticks_rotation='vertical')
        plt.savefig(f'{self.exp_path}/confusion_matrix.png')


def get_ttest_pnas(exp_path):
    preprocessor = TtestPreprocessor(exp_path)
    lasso_model = LASSO(exp_path)
    clf_model= SGDClassifier(loss='log', verbose=0, n_jobs=-1)
    pnas = PNAS(exp_path, preprocessor, lasso_model, clf_model)
    return pnas


def get_mod_ttest_pnas(exp_path):
    preprocessor = R_Preprocessor(exp_path)
    lasso_model = LASSO(exp_path)
    clf_model= SGDClassifier(loss='log', verbose=0, n_jobs=-1)
    pnas = PNAS(exp_path, preprocessor, lasso_model, clf_model)
    return pnas


def split_data(fpath, exp_path, debug=False):
    Logger.info(f'Load data from {fpath}')
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


def create_exp_path(fpath):
    fname = fpath.split('/')[-1].split('.')[0]
    exps = glob.glob(f'experiments/{fname}/*/')
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


if __name__ == '__main__':
    fpath = sys.argv[1]
    exp_path, fname = create_exp_path(fpath)

    Xtrain, Xtest, Ytrain, Ytest, label_list = split_data(fpath, exp_path, debug=True)
    pnas = get_mod_ttest_pnas(exp_path)

    pnas.run_pipeline(Xtrain, Ytrain)
    pnas.score(Xtest, Ytest, label_list)
