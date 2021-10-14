import os
import sys
import logging
import sklearn

import numpy as np

from sklearn.linear_model import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold


Logger = logging.getLogger(__name__)


class LASSO:
    def __init__(self, exp_path, lasso_model=None):
        self.lasso_model = lasso_model
        if lasso_model is None:
            self.lasso_model = LinearSVC(C=0.01, penalty="l1", dual=False)

    def _select_features_multiclass(self, dataX, dataY):
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
        return mask, self.lasso_model

    def _transform(self, X, Y, mask):
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

    def __call__(self, dataX, dataY):
        skf = StratifiedKFold(n_splits=10)
        Logger.info(f'Start K-Fold')
        masks = []
        for index, (train, test) in enumerate(skf.split(dataX, dataY)):
            Logger.info(f'Fold {index}')
            Xtrain, Ytrain = dataX[train], dataY[train]
            Xtest, Ytest = dataX[test], dataY[test]

            Logger.debug(f'Training data size: {Xtrain.shape} {Ytrain.shape}')
            Logger.debug(f'Validation data size: {Xtest.shape} {Ytest.shape}')

            mask, clf = self._select_features_multiclass(Xtrain, Ytrain)
            masks.append(mask)

            acc = clf.score(Xtest, Ytest)
            Logger.info(f'Validation Acc {acc}')

        masks = np.array(masks)
        masks = np.sum(masks, axis=0)
        final_mask = masks >= 7
        self.final_mask = np.array(final_mask)

        Logger.info('Num selected features for each class')
        for class_idx in range(final_mask.shape[0]):
            Logger.info(f'\tClass {class_idx}, num features: {np.sum(final_mask[class_idx])}')

        dataX = self._transform(dataX, dataY, final_mask)
        return dataX, dataY
