import logging
import rpy2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector

Logger = logging.getLogger(__name__)

#rpy2.robjects.numpy2ri.activate()
rprint = robjects.globalenv.find("print")

R = robjects.r
MKmisc = importr('MKmisc')
Rbase = importr('base')
Rstats = importr('stats')


class R_Preprocessor:
    '''Remove excessive noise using moderated t-test'''

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
            
            qvalues = stats['adj.p.value']
            mask = qvalues <= 0.05
            
            Logger.debug(f'{label}: {np.sum(mask)}')
            masks.append(mask)
        final_mask = self._merge(masks)
        return final_mask

    def _merge(self, masks):
        masks = np.array(masks).transpose(1, 0)
        out = np.array([all(s) for s in masks])
        Logger.info(f'Number of retained features: {np.sum(out)}')
        return out
        
data_adjacent_xy = './data_adjacent_xy.npz'
def load_data():
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
    X = X[:, :1000]
    proc = R_Preprocessor()
    mask = proc(Xs, Ys)
    print(sum(mask))