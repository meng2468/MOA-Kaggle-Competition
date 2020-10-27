import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from processing.data_reader import get_genes_cell_header


# from sklearn import preprocessing
# from sklearn.metrics import log_loss
# from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

### ========================================== Feature Adding

### =========== One Hot Encoding ============
def one_hot_encoding(data, column, values):
    '''
    replace columns with one-hot encoding

    Parameters:
        columns = columns name
        values = all possible values

    return dataframe
    '''
    data[column] = data[column].astype(CategoricalDtype(values))

    onehot = pd.get_dummies(data[column],prefix=column, drop_first=True)

    print(f"On-Hot Encoded {column}: Adding {len(values)-2} columns")

    ## insert
    loc = data.columns.get_loc(column)
    for c in reversed(onehot.columns):
        data.insert(loc,c, onehot[c])

    ## remove
    return data.drop(columns=column)

### =========== PCA ============
def get_PCA_features(data, n_comp, suffix="pca-", random_state=42):
    '''
    return n_comp number of PCA components
    '''
    features_pca = PCA(n_components=n_comp, random_state=random_state).fit_transform(data)
    return pd.DataFrame(features_pca, columns=[f'{suffix}{i}' for i in range(n_comp)])

### ========================================== Feature Selection

def drop_columns(data, columns):
    return data.drop(columns=columns)

### =========== Variance Encoding ============
def get_variance_encoding_columns(data, threshold=0.5):
    '''
    get feature name with variance encoding
    '''
    var_thresh = VarianceThreshold(threshold=threshold)
    var_thresh.fit(data)
    mask = var_thresh.get_support()
    return data.columns[~mask]

### =========== pipeline ============
def split_moa_train_test(data, train_len, test_len):
    """
    split train+test combined data to train and test
    """
    train_data = data[:train_len].reset_index(drop=True, inplace=False)
    test_data = data[-test_len:].reset_index(drop=True, inplace=False)
    return (train_data, test_data)

def one_hot_encode_moa(data):
    data = one_hot_encoding(data, "cp_time", [24,48,72])
    data = one_hot_encoding(data, "cp_dose", ["D1","D2"])
    return data

def add_PCA_feature(data, g_n_comp=50, c_n_comp=15):
    '''
    extend data with PCA decomposition

    Parameters:
        g_n_comp = number of components for genes
        c_n_comp = number of components for cells

    return train_features_pca, test_features_pca
    '''
    GENES, CELLS = get_genes_cell_header(data)

    # gnenes
    pca_genes = get_PCA_features(data[GENES], g_n_comp, suffix="pca_G-")
    print(f"Add #{g_n_comp} PCA G features")

    # cells
    pca_cells = get_PCA_features(data[CELLS], c_n_comp, suffix="pca_C-")
    print(f"Add #{c_n_comp} PCA C features")

    data_wiht_pca = pd.concat((data.reset_index(drop=True, inplace=False) , pca_genes, pca_cells), axis=1)

    return data_wiht_pca


def remove_variance_encoding(data, threshold=0.5):
    '''
    remove features usinf variance encoding
    '''
    remove_cols = get_variance_encoding_columns(data.iloc[:, 4:], threshold)
    cat_remain = data.columns[~data.columns.isin(remove_cols)]

    print(f"Remove #{len(remove_cols)} features via Variance Encoding (threshold={threshold})")

    return drop_columns(data, remove_cols)

def remove_ctl_vehicle(train, target):
    '''
    remove ctl_vehicle form train_data and train_target
    '''
    ctl_ids = train[train['cp_type']=='ctl_vehicle'].sig_id

    print(f"Remove {len(ctl_ids)} ctl_vehicel data")

    train = train[~train['sig_id'].isin(ctl_ids)].reset_index(drop=True)
    target = target[~target['sig_id'].isin(ctl_ids)].reset_index(drop=True)

    return train, target

def drop_cp_type(data):
    print("Dropping cp_type column")
    return drop_columns(data, "cp_type")

def preprocessing_pipeline(train_features, train_targets, test_features):

    train_len = train_features.shape[0]
    test_len = test_features.shape[0]

    data = pd.concat((train_features, test_features))

    ## TODO: save removed columns for testset transfromation
    data = remove_variance_encoding(data)

    ## TODO: save PCA value for testset transfromation
    data = add_PCA_feature(data)


    train_features, test_features = split_moa_train_test(data, train_len, test_len)

    train_features, train_targets = remove_ctl_vehicle(train_features, train_targets)

    train_features = drop_cp_type(train_features)
    train_features = one_hot_encode_moa(train_features)

    test_features = drop_cp_type(test_features)
    test_features = one_hot_encode_moa(test_features)



    return train_features, train_targets , test_features

