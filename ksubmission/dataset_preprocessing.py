import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pickle import load,dump

from dataset_reader import get_genes_cell_header


# from sklearn import preprocessing
# from sklearn.metrics import log_loss
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans

### ========================================== Feature Adding
def one_hot_encoding(data, column, values, drop_first=True, replace=False):
    '''
    replace columns with one-hot encoding

    Parameters:
        columns = columns name
        values = all possible values

    return dataframe
    '''
    data[column] = data[column].astype(CategoricalDtype(values))

    onehot = pd.get_dummies(data[column],prefix=column, drop_first=drop_first)

    print(f"On-Hot Encoded {column}: Adding {len(values)-2} columns")

    if replace:
        ## insert
        loc = data.columns.get_loc(column)
        for c in reversed(onehot.columns):
            data.insert(loc,c, onehot[c])

        ## remove
        return data.drop(columns=column)
    else:
        data = data.drop(columns=column)
        return pd.concat((data.reset_index(drop=True, inplace=False) , onehot), axis=1)


def get_PCA_features(data, n_comp, suffix="pca-", random_state=42, save_path="pca.pkl", load_path=None):
    '''
    return n_comp number of PCA components
    '''
    if load_path is None:
        pca_model = PCA(n_components=n_comp, random_state=random_state).fit(data)
        dump(pca_model, open(save_path, 'wb'))
    else:
        pca_model = load(open(load_path, 'rb'))
    features_pca = pca_model.transform(data)
    return pd.DataFrame(features_pca, columns=[f'{suffix}{i}' for i in range(features_pca.shape[1])])

def get_clustering_features(data, n_clusters, name="cluster_x", random_state=42, save_path="cluster.pkl", load_path=None):
    if load_path is None:
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=random_state).fit(data)
        dump(kmeans_model, open(save_path, 'wb'))
    else:
        kmeans_model = load(open(load_path, 'rb'))
        n_clusters = kmeans_model.n_clusters
    features_cluster = kmeans_model.predict(data)
    df = pd.DataFrame(features_cluster, columns=[name])
    df[name] = df[name].astype(CategoricalDtype(list(range(n_clusters))))
    return pd.get_dummies(df, columns=[name])

### ========================================== Feature Scaling

def gauss_rank(data, save_path='quantile.pkl', load_path = None):
    print("Apply Gauss Rank Transformation")
    if load_path is None:
        transformer = QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
        transformer.fit(data)
        dump(transformer, open(save_path, 'wb'))
    else:
        transformer = load(open(load_path, 'rb'))
    return transformer.transform(data)

### ========================================== Feature Selection
def drop_columns(data, columns):
    return data.drop(columns=columns)

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

def get_stat_feature(data):
    GSQUARE=['g-574','g-211','g-216','g-0','g-255','g-577','g-153','g-389','g-60','g-370','g-248','g-167','g-203','g-177','g-301','g-332','g-517','g-6','g-744','g-224','g-162','g-3','g-736','g-486','g-283','g-22','g-359','g-361','g-440','g-335','g-106','g-307','g-745','g-146','g-416','g-298','g-666','g-91','g-17','g-549','g-145','g-157','g-768','g-568','g-396']

    GENES, CELLS = get_genes_cell_header(data)

    df = data
    out_df = pd.DataFrame()
    # for df in train, test:
    out_df['g_sum'] = df[GENES].sum(axis = 1)
    out_df['g_mean'] = df[GENES].mean(axis = 1)
    out_df['g_std'] = df[GENES].std(axis = 1)
    out_df['g_kurt'] = df[GENES].kurtosis(axis = 1)
    out_df['g_skew'] = df[GENES].skew(axis = 1)
    out_df['c_sum'] = df[CELLS].sum(axis = 1)
    out_df['c_mean'] = df[CELLS].mean(axis = 1)
    out_df['c_std'] = df[CELLS].std(axis = 1)
    out_df['c_kurt'] = df[CELLS].kurtosis(axis = 1)
    out_df['c_skew'] = df[CELLS].skew(axis = 1)
    out_df['gc_sum'] = df[GENES + CELLS].sum(axis = 1)
    out_df['gc_mean'] = df[GENES + CELLS].mean(axis = 1)
    out_df['gc_std'] = df[GENES + CELLS].std(axis = 1)
    out_df['gc_kurt'] = df[GENES + CELLS].kurtosis(axis = 1)
    out_df['gc_skew'] = df[GENES + CELLS].skew(axis = 1)
    
    out_df['c52_c42'] = df['c-52'] * df['c-42']
    out_df['c13_c73'] = df['c-13'] * df['c-73']
    out_df['c26_c13'] = df['c-23'] * df['c-13']
    out_df['c33_c6'] = df['c-33'] * df['c-6']
    out_df['c11_c55'] = df['c-11'] * df['c-55']
    out_df['c38_c63'] = df['c-38'] * df['c-63']
    out_df['c38_c94'] = df['c-38'] * df['c-94']
    out_df['c13_c94'] = df['c-13'] * df['c-94']
    out_df['c4_c52'] = df['c-4'] * df['c-52']
    out_df['c4_c42'] = df['c-4'] * df['c-42']
    out_df['c13_c38'] = df['c-13'] * df['c-38']
    out_df['c55_c2'] = df['c-55'] * df['c-2']
    out_df['c55_c4'] = df['c-55'] * df['c-4']
    out_df['c4_c13'] = df['c-4'] * df['c-13']
    out_df['c82_c42'] = df['c-82'] * df['c-42']
    out_df['c66_c42'] = df['c-66'] * df['c-42']
    out_df['c6_c38'] = df['c-6'] * df['c-38']
    out_df['c2_c13'] = df['c-2'] * df['c-13']
    out_df['c62_c42'] = df['c-62'] * df['c-42']
    out_df['c90_c55'] = df['c-90'] * df['c-55']
    
    
    for feature in CELLS:
        out_df[f'{feature}_squared'] = df[feature] ** 2     
            
    for feature in GSQUARE:
        out_df[f'{feature}_squared'] = df[feature] ** 2        

    print(f"Add #{out_df.shape[1]} Stats features")
    return out_df    
    # data_with_stat = pd.concat((data.reset_index(drop=True, inplace=False) , out_df), axis=1)
    # return data_with_stat

def one_hot_encode_moa(data, drop_columns=True, replace=False):
    data = one_hot_encoding(data, "cp_time", [24,48,72], drop_columns, replace)
    data = one_hot_encoding(data, "cp_dose", ["D1","D2"], drop_columns, replace)
    return data

def add_PCA_feature(data, g_n_comp=50, c_n_comp=15, 
                    save_path_g="gpca.pkl", save_path_c="cpca.pkl",
                    load_path_g=None, load_path_c=None ):
    '''
    extend data with PCA decomposition

    Parameters:
        g_n_comp = number of components for genes
        c_n_comp = number of components for cells

    return train_features_pca, test_features_pca
    '''
    GENES, CELLS = get_genes_cell_header(data)

    # gnenes
    pca_genes = get_PCA_features(data[GENES], g_n_comp, suffix="pca_G-", save_path=save_path_g, load_path=load_path_g)
    print(f"Add #{pca_genes.shape[1]} PCA G features")

    # cells
    pca_cells = get_PCA_features(data[CELLS], c_n_comp, suffix="pca_C-", save_path=save_path_c, load_path=load_path_c)
    print(f"Add #{pca_cells.shape[1]} PCA C features")

    data_with_pca = pd.concat((data.reset_index(drop=True, inplace=False) , pca_genes, pca_cells), axis=1)

    return data_with_pca

def add_clustering_feature(data, g_n_cluster=22, c_n_cluster=4, pca_n_cluster=5, 
                    save_path_g="g_cluster.pkl", save_path_c="c_cluster.pkl", save_path_pca="pca_cluster.pkl",
                    load_path_g=None, load_path_c=None, load_path_pca=None ):

    GENES, CELLS = get_genes_cell_header(data)
    PCA = [col for col in data.columns if col.startswith('pca')]

    # gnenes
    clustering_genes = get_clustering_features(data[GENES], g_n_cluster, name="cluster_g", 
                                        save_path=save_path_g, load_path=load_path_g)
    print(f"Add #{clustering_genes.shape[1]} G Clustering features")

    # cells
    clustering_cells = get_clustering_features(data[CELLS], c_n_cluster, name="cluster_c", 
                                        save_path=save_path_c, load_path=load_path_c)
    print(f"Add #{clustering_cells.shape[1]} C Clustering features")

    #pca
    clustering_pca = get_clustering_features(data[PCA], pca_n_cluster, name="cluster_pca", 
                                        save_path=save_path_pca, load_path=load_path_pca)
    print(f"Add #{clustering_pca.shape[1]} PCA Clustering features")

    data_with_cluster = pd.concat((data.reset_index(drop=True, inplace=False) , clustering_genes, clustering_cells, clustering_pca), axis=1)

    return data_with_cluster

def remove_variance_encoding(data, threshold=0.5, save_path="./remove_col.pkl"):
    '''
    remove features using variance encoding
    ''' 
    remove_cols = get_variance_encoding_columns(data.iloc[:, 4:], threshold)
    dump(remove_cols, open(save_path, 'wb'))
    print(f"Remove #{len(remove_cols)} features via Variance Encoding (threshold={threshold})")

    return drop_columns(data, remove_cols)

def remove_ctl_vehicle(train, target=None):
    '''
    remove ctl_vehicle form train_data and train_target
    '''
    ctl_ids = train[train['cp_type']=='ctl_vehicle'].sig_id

    print(f"Remove {len(ctl_ids)} ctl_vehicel data")

    train = train[~train['sig_id'].isin(ctl_ids)].reset_index(drop=True)
    if target:
        target = target[~target['sig_id'].isin(ctl_ids)].reset_index(drop=True)

    return train, target

def drop_cp_type(data):
    print("Dropping cp_type column")
    return drop_columns(data, "cp_type")

def get_state_feature_tabnet(data):
    GENES, CELLS = get_genes_cell_header(data)
    out_df = pd.DataFrame()
    for stats in ["sum", "mean", "std", "kurt", "skew"]:
        out_df["g_" + stats] = getattr(data[GENES], stats)(axis = 1)
        out_df["c_" + stats] = getattr(data[CELLS], stats)(axis = 1)    
        out_df["gc_" + stats] = getattr(data[GENES + CELLS], stats)(axis = 1)
    return out_df

def preprocessing_pipeline(train_features, train_targets, test_features):

    train_len = train_features.shape[0]
    test_len = test_features.shape[0]

    data = pd.concat((train_features, test_features))

    data = remove_variance_encoding(data, 0.8)
    data = add_PCA_feature(data, g_n_comp=0.9, c_n_comp=0.9)

    data = remove_variance_encoding(data, 0.8)

    data.iloc[:,4:] = gauss_rank(data.iloc[:,4:])

    data = add_clustering_feature(data, g_n_cluster=22, c_n_cluster=4, pca_n_cluster=5)

    train_features, test_features = split_moa_train_test(data, train_len, test_len)

    train_features, train_targets = remove_ctl_vehicle(train_features, train_targets)

    train_features = drop_cp_type(train_features)
    train_features = one_hot_encode_moa(train_features)

    test_features = drop_cp_type(test_features)
    test_features = one_hot_encode_moa(test_features)
    
    print("SIZE :", "TRAIN", train_features.shape)
    return train_features, train_targets, test_features

def preprocessing_NN_meta(train_features, train_targets, test_features):

    train_len = train_features.shape[0]
    test_len = test_features.shape[0]

    data = pd.concat((train_features, test_features))
    data_ori = data.copy()

    data.iloc[:,4:] = gauss_rank(data.iloc[:,4:])
    data = add_PCA_feature(data, g_n_comp=600, c_n_comp=50)
    
    remove_cols = get_variance_encoding_columns(data.iloc[:, 4:], 0.87)
    dump(remove_cols, open('remove_col.pkl', 'wb'))

    data = add_clustering_feature(data, g_n_cluster=22, c_n_cluster=4, pca_n_cluster=5)

    data_stat = get_stat_feature(data_ori)
    data = pd.concat((data.reset_index(drop=True, inplace=False) , data_stat), axis=1)

    data = drop_columns(data, remove_cols)

    train_features, test_features = split_moa_train_test(data, train_len, test_len)

    train_features, train_targets = remove_ctl_vehicle(train_features, train_targets)

    train_features = drop_cp_type(train_features)
    test_features = drop_cp_type(test_features)

    train_features = one_hot_encode_moa(train_features, False)
    test_features = one_hot_encode_moa(test_features, False)
    
    print("SIZE :", "TRAIN", train_features.shape)
    return train_features, train_targets, test_features

def preprocessing_NN_meta_transform(test_features, paths_dict):
    for k in ['quantile', 'g_pca', 'c_pca', 'g_cluster', 'c_cluster', 'pca_cluster', 'remove_col']:
        assert k in paths_dict.keys()

    data = test_features
    data_ori = data.copy()
    data.iloc[:,4:] = gauss_rank(data.iloc[:,4:], load_path=paths_dict['quantile'])
    data = add_PCA_feature(data, load_path_g=paths_dict['g_pca'], load_path_c=paths_dict['c_pca'])
    data = add_clustering_feature(data, load_path_g=paths_dict['g_cluster'], load_path_c=paths_dict['c_cluster'], load_path_pca=paths_dict['pca_cluster'])

    data_stat = get_stat_feature(data_ori)

    remove_cols = load(open(paths_dict['remove_col'], 'rb'))

    data = pd.concat((data.reset_index(drop=True, inplace=False) , data_stat), axis=1)
    data = drop_columns(data, remove_cols)

    data = drop_cp_type(data)
    data = one_hot_encode_moa(data, False)
    return data

def preprocessing_NN_TL(train_features, train_targets, test_features):
    
    train_len = train_features.shape[0]
    test_len = test_features.shape[0]

    data = pd.concat((train_features, test_features))
    data_ori = data.copy()

    data.iloc[:,4:] = gauss_rank(data.iloc[:,4:])
    data = add_PCA_feature(data, g_n_comp=600, c_n_comp=50)
    
    remove_cols = get_variance_encoding_columns(data.iloc[:, 4:], 0.82)
    dump(remove_cols, open('remove_col.pkl', 'wb'))
    data = drop_columns(data, remove_cols)

    train_features, test_features = split_moa_train_test(data, train_len, test_len)

    train_features, train_targets = remove_ctl_vehicle(train_features, train_targets)

    train_features = drop_cp_type(train_features)
    test_features = drop_cp_type(test_features)

    train_features = one_hot_encode_moa(train_features, False)
    test_features = one_hot_encode_moa(test_features, False)
    
    print("SIZE :", "TRAIN", train_features.shape)
    return train_features, train_targets, test_features

def preprocessing_NN_TL_transform(test_features, paths_dict):
    for k in ['quantile', 'g_pca', 'c_pca', 'remove_col']:
        assert k in paths_dict.keys()

    data = test_features
    data_ori = data.copy()
    data.iloc[:,4:] = gauss_rank(data.iloc[:,4:], load_path=paths_dict['quantile'])
    data = add_PCA_feature(data, load_path_g=paths_dict['g_pca'], load_path_c=paths_dict['c_pca'])

    remove_cols = load(open(paths_dict['remove_col'], 'rb'))
    data = drop_columns(data, remove_cols)

    data = drop_cp_type(data)
    data = one_hot_encode_moa(data, False)
    return data