import pandas as pd
import numpy as np


# PNAS markers
breast_cancer_markers= ["cg01327147", "cg02680086", "cg05395187", "cg07493516",
                        "cg09819083", "cg13976210", "cg18482112", "cg20069090",
                        "cg04772948", "cg08268679", "cg14817783", "cg24732563",
                        "cg04917276", "cg08549335", "cg15412918"]
breast_normal_markers= ["cg00886954", "cg23690893"]

colon_cancer_markers= ["cg08088171", "cg13420112", "cg24583770", "cg27536151",
                       "cg14642259", "cg20973720"]
colon_normal_markers= ["cg24741563", "cg22979615"]

liver_cancer_markers= ["cg07360250", "cg08550839", "cg13499300"]
liver_normal_markers= ["ch.7.135065R", "cg14054357"]

lung_cancer_markers= ["cg01602690", "cg03993087", "cg04383154", "cg05346286",
                      "cg05784951", "cg06352912", "cg06800849", "cg07903001",
                      "cg08089301", "cg14419975", "cg15545942", "cg17894293",
                      "cg19924352", "cg20691722", "cg21845794", "cg04933208",
                      "cg07464206", "cg15963326", "cg24398479"]
lung_normal_markers= ["cg03169557", "cg04549287", "cg07649862", "cg08682723"]

# Array index of paper markers must have the same order with our mask 
paper_markers = np.array([breast_normal_markers, breast_cancer_markers,
                        colon_normal_markers, colon_cancer_markers,
                        liver_normal_markers, liver_cancer_markers,
                        lung_normal_markers, lung_cancer_markers], dtype='object')

num_to_label = {
    0 : "Breast normal",
    1 : "Breast cancer",
    2 : "Colon normal",
    3 : "Colon cancer",
    4 : "Liver normal",
    5 : "Liver cancer",
    6 : "Lung normal",
    7 : "Lung cancer"
}

def get_duplicated_markers(full_features, mask):
    """
    
    """
    duplicated_markers = {}
    for each_label_idx in range(mask.shape[0]):
        label_mask = mask[each_label_idx, :]
        label_indices = np.argwhere(label_mask==1)
        our_markers = full_features[label_indices]
        duplicated_markers[num_to_label.get(each_label_idx)] = np.intersect1d(paper_markers[each_label_idx], our_markers )
    return duplicated_markers

def load_data():
    full_features = pd.read_csv("data/new_data/tcga_450_features_name.csv").to_numpy()
    mask = np.load("data/new_data/final_marker_to_draw.npy")
    return full_features, mask

if(__name__== "__main__"):
    full_features, mask = load_data()
    duplicated_markers = get_duplicated_markers(full_features, mask)
    print(duplicated_markers)

