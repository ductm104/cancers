import os
import scipy
import logging
import numpy as np


Logger = logging.getLogger(__name__)


def ttest_scipy(arr1, arr2):
    '''Perform ttest on dimension 1 of two arrays'''
    tvalues, pvalues = scipy.stats.ttest_ind(arr1, arr2, axis=1, equal_var=True)
    return tvalues, pvalues


def pre_compute_proc(*x, fpath='./final_mask.npy'):
    final_mask = np.load(fpath)
    return final_mask


class TtestPreprocessor:
    '''Remove excessive noise using (not yet) moderated t-test'''
    def __init__(self, exp_path):
        self.exp_path = exp_path
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

        # filter out not significant features
        final_mask = self._merge(masks)
        self.final_mask = final_mask

        dataX[:, final_mask==0] = 0
        return dataX, dataY

    def _get_support_mask(self, pvalues, alpha=0.05):
        '''Perform Bejamini-Hochberg procedure to select features'''
        n_features = len(pvalues)
        sv = np.sort(pvalues)
        selected = sv[sv <= float(alpha) / n_features * np.arange(1, n_features + 1)]
        if len(selected) == 0:
            return np.zeros_like(pvalues, dtype=bool)
        return pvalues <= selected.max()

    def _filter(self, x_label, x_other):
        # transpose to n_features * n_samples
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
