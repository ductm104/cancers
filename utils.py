import json
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
    data_xy = np.load(fpath, allow_pickle=True)
    dataX = data_xy['dataX']
    dataY = data_xy['dataY']

    label_list = sorted(np.unique(dataY))
    label_map = {val:i for i, val in enumerate(label_list)}
    rev_label_map = {i:val for i, val in enumerate(label_list)}
    print(f'Label list:', label_list)
    print(f'Label map: {label_map}')

    dataX = np.nan_to_num(dataX)
    dataY = np.array([label_map[label] for label in dataY])

    print('Data size:', dataX.shape, dataY.shape)
    return dataX, dataY, label_list


def get_feature_names(mask_path):
    output = []
    #with open('./new_data/tcga_450_features_name.csv') as f:
        #fnames = f.readlines()[1:]
        #print(len(fnames))
        #fnames = [name.strip() for name in fnames]
    with open('./new_data/features_name.json') as f:
        fnames = json.load(f)
    fnames = np.array(fnames)

    mask = np.load(mask_path)
    print(mask.shape)

    assert mask.shape[1] == len(fnames), 'Len not equal'

    for dtype in mask:
        selected = np.where(dtype>0)[0]
        names = fnames[selected]
        output.append(names)

    print(output)
    return output

if __name__ == '__main__':
    #data_adjacent_xy = './new_data/data_adjacent_xy.npz'
    #X, Y, label_map = load_data(data_adjacent_xy)
    output = get_feature_names('./data_adjacent_xy/final_marker_to_draw.npy')

    with open('./data_adjacent_xy_cp/ttest_markers.txt', 'w') as f:
        for x in output:
            print(x, file=f)
