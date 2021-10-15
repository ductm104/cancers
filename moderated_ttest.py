import os
import gc
import sys
import logging
import psutil
import rpy2
import numpy as np
import pandas as pd

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


logging.basicConfig(level=logging.DEBUG)
Logger = logging.getLogger(__name__)


#----------------------------------#
#----------R utils-----------------#
#----------------------------------#
def log_mem(mess=''):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024**3)
    print(f'\t{mess}.\t\t Current memory allocated {round(mem)}(GB)')


def clean_mem():
    gc = robjects.r['gc']
    gc()


def install_packages():
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector

    utils = rpackages.importr('utils')
    utils.chooseCRANmirror(ind=1) # select the first mirror in the list

    packnames = ('ggplot2', 'hexbin', 'BiocManager', 'limma', 'MKmisc')
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

try:
    R = robjects.r
    MKmisc = importr('MKmisc')
    Rbase = importr('base')
    Rstats = importr('stats')
except Exception:
    install_packages()


#----------------------------------#
#----------moderated test----------#
#----------------------------------#
def moderated_ttest(X, Y):
    '''Perform Moderated TTest using R backend
    With Adjustment method is Benjamini-HochBerg
    Params:
        - X: matrix of shape [n_feature, n_samples]
        - Y: vector of shape [n_samples]
             Y must contain only 2 unique values
             e.g. Y = [0, 1, 0, 1, 1, 0]
             The i_th value of Y represent the label of sample X_i
    Return:
        - stats: matrix of statistic
    '''
    Rbase.set_seed(42)
    nr, nc = X.shape[:2]
    X = Rbase.matrix(X, nrow=nr, ncol=nc)
    Y = Rbase.factor(Y)
    stats = MKmisc.mod_t_test(X, group=Y, paired=False, adjust_method='BH')
    stats = pd.DataFrame(stats)
    del X, Y
    gc.collect()
    clean_mem()
    return stats


class R_Preprocessor:
    '''Remove excessive noise using moderated t-test'''
    def __init__(self, exp_path, threshold=0.05):
        self.exp_path = exp_path
        self.threshold = threshold

    def __call__(self, dataX, dataY):
        dataX = dataX.transpose(1, 0)  # n_features * n_samples
        labels = np.unique(dataY)
        masks = []
        for label in labels:
            other = list(set(labels)-set([label]))
            Logger.debug(f'Perform T-Test ong {label} vs {other}')

            Y_clone = dataY.copy()
            Y_clone[Y_clone != label] = label+1

            stats = moderated_ttest(dataX, Y_clone)
            mask = np.zeros(len(stats['adj.p.value']), dtype=bool)
            mask[stats['adj.p.value'] <= self.threshold] = True
            masks.append(mask)

            Logger.debug(f'Label {label} num markers: {np.sum(mask)}')

            del mask, stats
            gc.collect()

        final_mask = self._merge(masks)
        self.final_mask = final_mask
        dataX = dataX.transpose(1, 0)
        dataX[:, final_mask==0] = 0
        return dataX, dataY

    def _merge(self, masks):
        masks = np.array(masks)
        masks = masks.transpose(1, 0)
        with open(f'{self.exp_path}/moderated_ttest_all_masks.npy', 'wb') as f:
            np.save(f, masks)

        final_mask = np.array([all(s) for s in masks])
        with open(f'{self.exp_path}/moderated_ttest_merged_final_mask.npy', 'wb') as f:
            np.save(f, final_mask)
        Logger.info(f'Number of retained features: {np.sum(final_mask)}')
        return final_mask


if __name__ == '__main__':
    input_file = sys.argv[1]
    fname = input_file.split('/')[-1].split('.')[0]

    X, Y, label_map = load_data(input_file)
    if os.environ.get('test', False):
        X = X[:, :1000]

    proc = R_Preprocessor()
    mask = proc(X, Y)
    print(sum(mask))
