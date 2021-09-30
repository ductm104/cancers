import os
import logging
import scipy
import sklearn
import seaborn
import functools
import multiprocessing

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import *
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, StratifiedKFold


LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)
Logger = logging.getLogger(__name__)


def ttest_scipy(arr1, arr2):
    '''Perform ttest on dimension 1 of two arrays'''
    tvalues, pvalues = scipy.stats.ttest_ind(arr1, arr2, axis=1, equal_var=True)
    return tvalues, pvalues


class Preprocessor:
    '''Remove excessive noise using (not yet) moderated t-test'''
    def __init__(self, ttest_fn=None,):
        if ttest_fn is None:
            self.ttest_fn = ttest_scipy

    def __call__(self, dataX, dataY):
        labels = np.unique(dataY)
        masks = []
        for label in labels:
            other = list(set(labels)-set([label]))
            Logger.debug(f'Perform T-Test ong {label} vs {other}')

            x_label = dataX[dataY==label]
            x_other = dataX[dataY!=label]
            Logger.debug(f'Data size: {x_label.shape} vs {x_other.shape}')

            mask = self._filter(x_label, x_other)
            Logger.debug(f'{label}: {np.sum(mask)}')
            masks.append(mask)
        final_mask = self._merge(masks)
        return final_mask

    def _get_support_mask(self, pvalues, alpha=0.05):
        '''Perform Bejamini-Hochberg procedure to select features'''
        n_features = len(pvalues)
        sv = np.sort(pvalues)
        selected = sv[sv <= float(alpha) / n_features * np.arange(1, n_features + 1)]
        if len(selected) == 0:
            return np.zeros_like(pvalues, dtype=bool)
        return pvalues <= selected.max()

    def _filter(self, x_label, x_other):
        x_label = x_label.transpose(1, 0)
        x_other = x_other.transpose(1, 0)

        tvalues, pvalues = self.ttest_fn(x_label, x_other)
        mask = self._get_support_mask(pvalues)
        return mask

    def _merge(self, masks):
        masks = np.array(masks).transpose(1, 0)
        out = np.array([all(s) for s in masks])
        Logger.info(f'Number of retained features: {np.sum(out)}')
        return out


class PNAS:
    def __init__(self,
            preprocessor=None,
            lasso_model=None,
            clf_model=None
    ):
        self.preprocessor = preprocessor
        if preprocessor is None:
            self.preprocessor = Preprocessor()

        self.lasso_model = lasso_model
        if lasso_model is None:
            self.lasso_model = LinearSVC(C=0.01, penalty="l1", dual=False)

        self.clf_model = clf_model
        if clf_model is None:
            self.clf_model= SGDClassifier(loss='log', verbose=0)

    def lasso_select_features_multiclass(self, dataX, dataY):
        '''Using Lasso to select features
        Return: mask of size [n_classes, n_features]
        '''
        def select_from_model(coef, threshold=1e-5):
            scores = np.abs(coef)
            mask = np.ones_like(scores, dtype=bool)
            mask[scores < threshold] = False
            return mask
        Logger.debug(f'Input data size: {dataX.shape} and {dataY.shape}')

        self.lasso_model.fit(dataX, dataY)
        Logger.debug(f'Feature selection model fiting Accuracy: {self.lasso_model.score(dataX, dataY)}')

        mask = select_from_model(self.lasso_model.coef_)
        Logger.info('Num selected features for each class')
        for class_idx in range(mask.shape[0]):
            Logger.info(f'\tClass {class_idx}, num features: {np.sum(mask[class_idx])}')
        return mask

    def transform(self, X, Y, mask):
        '''filter out not selected features
        X.shape == [n_samples, n_features]
        Y.shape == [n_samples]
        mask.shape == [n_classes, n_features]
        '''
        n_classes = mask.shape[0]
        for cls_idx in range(n_classes):
            sub_mask = mask[cls_idx]
            X[Y==cls_idx][:, sub_mask==0] = 0
        return X

    def train_clf(self, dataX, dataY):
        Logger.debug('Start training classification model')

        self.clf_model.fit(dataX, dataY)
        acc = self.clf_model.score(dataX, dataY)
        Logger.info(f'Classification model training Acc: {acc}')

        return self.clf_model

    def pipeline(self, dataX, dataY, n_folds=10):
        skf = StratifiedKFold(n_splits=n_folds)

        Logger.info(f'Start K-Fold')
        for index, (train, test) in enumerate(skf.split(dataX, dataY)):
            Logger.info(f'Fold {index}')
            Xtrain, Ytrain = dataX[train], dataY[train]
            Xtest, Ytest = dataX[test], dataY[test]

            Logger.debug(f'Training data size: {Xtrain.shape} {Ytrain.shape}')
            Logger.debug(f'Validation data size: {Xtest.shape} {Ytest.shape}')

            mask = self.lasso_select_features_multiclass(Xtrain, Ytrain)

            Xstest = self.transform(Xtest, Ytest, mask)
            Xstrain = self.transform(Xtrain, Ytrain, mask)

            clf = self.train_clf(Xstrain, Ytrain)
            acc = clf.score(Xstest, Ytest)
            Logger.info(f'Validation Acc {acc}')

        return self.clf_model

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
        mask_1st = self.preprocessor(dataX, dataY)
        dataX[:, mask_1st==0] = 0

        self.pipeline(dataX, dataY)
        return []


def load_data():
    data_adjacent_xy = './data_adjacent_xy.npz'
    data_xy = np.load(data_adjacent_xy)
    dataX = data_xy['dataX']
    dataY = data_xy['dataY']

    label_list = sorted(np.unique(dataY))
    label_map = {val:i for i, val in enumerate(label_list)}
    print(f'Label map: {label_map}')

    dataX = np.nan_to_num(dataX)
    dataY = np.array([label_map[label] for label in dataY])

    print('Data size:', dataX.shape, dataY.shape)
    return dataX, dataY, label_map

if __name__ == '__main__':
    X, Y, label_map = load_data()
    pnas = PNAS()
    pnas.run_pipeline(X, Y)
