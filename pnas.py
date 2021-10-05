import os
import logging
import scipy
import sklearn
import seaborn

import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import *
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from ttest import TtestPreprocessor
from utils import load_data


LOGLEVEL = os.environ.get('LOGLEVEL', 'DEBUG').upper()
logging.basicConfig(level=LOGLEVEL)
Logger = logging.getLogger(__name__)


class LASSO:
    def __init__(self, lasso_model=None):
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


class PNAS:
    def __init__(self,
            clf_model=None
    ):
        self.preprocessor = TtestPreprocessor()
        self.lasso_model = LASSO()
        self.clf_model= SGDClassifier(loss='log', verbose=0, n_jobs=-1)

    def train_clf(self, dataX, dataY):
        Logger.debug('Start training classification model')

        self.clf_model.fit(dataX, dataY)
        acc = self.clf_model.score(dataX, dataY)
        Logger.info(f'Classification model training Acc: {acc}')
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
        dataX, dataY = self.preprocessor(dataX, dataY)
        dataX, dataY = self.lasso_model(dataX, dataY)

        final_mask = self.lasso_model.final_mask
        clf = self.train_clf(dataX, dataY)
        return clf, final_mask


if __name__ == '__main__':
    fpath = './new_data/data_adjacent_xy.npz'
    X, Y, label_map = load_data(fpath)
    X=X[:, :1000]
    Xtrain, Xtest, Ytrain, Ytest = sklearn.model_selection.train_test_split(X, Y,
                test_size=0.33, random_state=42, stratify=Y)

    pnas = PNAS()
    final_clf, final_mask = pnas.run_pipeline(Xtrain, Ytrain)
    with open('./final_marker_to_draw.npy', 'wb') as f:
        np.save(f, final_mask)
    score = final_clf.score(Xtest, Ytest)
    preds = final_clf.predict(Xtest)
    Logger.info(f'Final testing score {score}')
    print(sklearn.metrics.classification_report(Ytest, preds))
    cm = confusion_matrix(Ytest, preds, labels=final_clf.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=final_clf.classes_)
    disp.plot()
    #plt.show()
    plt.savefig('cm.png')
