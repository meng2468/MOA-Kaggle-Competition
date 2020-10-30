import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA

def remove_sig_id(df_x, df_y):
    df_x = df_x.drop('sig_id', axis=1)
    df_y = df_y.drop('sig_id', axis=1)
    return df_x, df_y

def remove_ctl_v(df_x,df_y):
    features = list(df_x.columns)
    targets = list(df_y.columns)

    df = df_x
    for column in targets:
        df[column] = df_y[column]
    
    df = df[df['cp_type'] != 'ctl_vehicle']

    df_x = pd.DataFrame(columns=features)
    df_y = pd.DataFrame(columns=targets)

    df_x = df.loc[:,features[1:]]
    df_y = df.loc[:, targets]

    return df_x, df_y

def add_pca(df_x, df_y):
    g_cols = [col for col in df_x.columns if col.startswith('g-')]
    c_cols = [col for col in df_x.columns if col.startswith('c-')]
    
    gene_data = df_x.loc[:,g_cols]
    cell_data = df_x.loc[:,c_cols]

    g_pca = PCA(n_components=600).fit_transform(gene_data)
    c_pca = PCA(n_components=100).fit_transform(cell_data)

    # g_pca = pd.DataFrame(g_pca, columns=['gpca-'+str(x) for x in range(600)])
    # c_pca = pd.DataFrame(c_pca, columns=['cpca-'+str(x) for x in range(100)])
    # print(g_pca.shape)
    # print(c_pca.shape)

    # df_x = pd.concat((df_x, g_pca), axis=1)
    # print(df_x.shape)

    # df_x = pd.concat((df_x, c_pca), axis=1)
    # print(df_x.shape)

    df_x = pd.concat((df_x, pd.DataFrame(g_pca)),axis=1)
    df_x = pd.concat((df_x, pd.DataFrame(c_pca, columns=['cpca'+str(x) for x in range(100)])),axis=1)
    return df_x, df_y

def get_eng_df(df_x, df_y):
    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)

    return df_x, df_y

def to_csv():
    df_x = pd.read_csv('feature_augm_gauss.csv')
    df_y = pd.read_csv('../data/train_targets_scored.csv')

    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)
    df_x, df_y = add_pca(df_x, df_y)

    df_x.to_csv('feature_eng_gpca_x.csv', index=False)
    df_y.to_csv('feature_eng_y.csv', index=False)

to_csv()