import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import json

class Cluster():
    def __init__(self, X_df, y, label_to_num, mask):
        self.X_df = X_df
        self.y = y
        self.label_to_num = label_to_num
        self.mask = mask
    def __call__(self):
        list_all_markers = self.get_marker_name_from_mask(X_df, mask)
        self.draw_cluster(X_df, y, list_all_markers, label_to_num)

    
    def get_marker_name_from_mask(self, X_df, mask):
        list_all_markers = []
        for class_i in range (mask.shape[0]):
            marker_indices = np.argwhere(mask[class_i,:] == 1)
            marker_class_i = X_df.columns[marker_indices].squeeze().tolist()
            list_all_markers += marker_class_i
        list_all_markers= np.unique(list_all_markers)
        return list_all_markers
    
    def draw_cluster(self, X_df, y, list_all_markers, label_to_num):
        label_to_color = {
        0: '#5a94f4',
        1: '#ff42a4',
        2: '#994714',
        3: '#502020',
        4: '#574a65',
        5: '#cdad00',
        6: '#417029',
        7: '#655b3f'
        }
        col_colors = np.vectorize(label_to_color.get)(y)
        label_pal = sns.cubehelix_palette(len(np.unique(y)), reverse=True,)
        label_pal = sns.color_palette("hls", 8)
        label_lut = dict(zip(np.unique(y),label_pal))
        label_colors = pd.Series(y).map(label_lut)
        sns.set_theme(color_codes=True)
        g = sns.clustermap(np.transpose(X_df.loc[:, list_all_markers]),
                method='average' ,col_colors=label_colors, metric='euclidean', yticklabels=False)
        for label in np.unique(y):
            label_name = [k for k,v in label_to_num.items() if v == label]
            g.ax_col_dendrogram.bar(0, 0, color=label_lut[label],
                                    label=label_name, linewidth=0)
            g.ax_col_dendrogram.legend(loc="center", ncol=4)
        g.savefig('results/clustermap.png')


def load_data(use_old_data=False):
    if use_old_data == True:
        data_xy = np.load('data/old_data/data_adjacent_xy.npz', allow_pickle=True)
        with open('data/old_data/features_name.json') as f:      
            list_450k_features =  json.load(f)
        with open('data/old_data/labelmap.json') as f:
            label_to_num = json.load(f)
        mask = np.load('data/old_data/final_marker_to_draw.npy')
        X = data_xy['dataX']
        y = data_xy['dataY']
        y = np.vectorize(label_to_num.get)(y)
        X_df = pd.DataFrame(X, columns = list_450k_features)
        X_df.fillna(0, inplace=True)
    else:
        data_xy = np.load('data/new_data/data_tcga_450_beta.npz', allow_pickle=True)
        list_450k_features =  pd.read_csv('data/new_data/tcga_450_features_name.csv')
        with open('data/new_data/labelmap.json') as f:
            label_to_num = json.load(f)
        mask = np.load('data/new_data/final_marker_to_draw.npy')
        X = data_xy['dataX']
        y = data_xy['dataY']
        y = np.vectorize(label_to_num.get)(y)
        X_df = pd.DataFrame(X, columns = np.squeeze(list_450k_features.to_numpy()))
        X_df.fillna(0, inplace=True)
    return X_df, y, label_to_num, mask

if(__name__ == "__main__"):
    X_df, y, label_to_num, mask = load_data(use_old_data=True)
    cluster = Cluster(X_df, y, label_to_num, mask)
    cluster()
