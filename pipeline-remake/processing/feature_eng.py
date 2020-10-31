import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

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

def add_pca(df_x, gpca, cpca):
    g_cols = [col for col in df_x.columns if col.startswith('g-')]
    c_cols = [col for col in df_x.columns if col.startswith('c-')]
    
    gene_data = df_x.loc[:,g_cols]
    cell_data = df_x.loc[:,c_cols]

    g_pca = PCA(n_components=gpca).fit_transform(gene_data)
    c_pca = PCA(n_components=cpca).fit_transform(cell_data)

    df_x = df_x.reset_index()
    df_x = pd.concat((df_x, pd.DataFrame(g_pca, columns=['gp'+str(x) for x in range(gpca)])),axis=1)
    df_x = df_x.reset_index()
    df_x = pd.concat((df_x, pd.DataFrame(c_pca, columns=['cpca'+str(x) for x in range(cpca)])),axis=1)
    return df_x.loc[:,df_x.columns[2:]]

def only_pca(df_x, gpca, cpca):
    g_cols = [col for col in df_x.columns if col.startswith('g-')]
    c_cols = [col for col in df_x.columns if col.startswith('c-')]
    
    gene_data = df_x.loc[:,g_cols]
    cell_data = df_x.loc[:,c_cols]

    g_pca = PCA(n_components=gpca).fit_transform(gene_data)
    c_pca = PCA(n_components=cpca).fit_transform(cell_data)

    df_gpca = pd.DataFrame(g_pca, columns=['gp'+str(x) for x in range(gpca)])
    df_cpca = pd.DataFrame(c_pca, columns=['cpca'+str(x) for x in range(cpca)])


    df_new = pd.concat((df_gpca, df_cpca), axis=1)
    return df_new

def feature_sel(df_x, var):
    sel = VarianceThreshold(var)
    data = sel.fit_transform(df_x.values)
    df_x = pd.DataFrame(data)
    return df_x

def get_eng_df(df_x, df_y):
    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)

    return df_x, df_y

def to_csv(var, gpca, cpca):
    df_x = pd.read_csv('feature_augm.csv')
    df_y = pd.read_csv('../data/train_targets_scored.csv')

    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)
    df_x = add_pca(df_x, gpca, cpca)
    df_x = feature_sel(df_x, var)

    print("Writing csv for " + str(var) +", " + str(gpca) + ", " + str(cpca))
    print("Shape: " + str(df_x.shape))
    df_x.to_csv('./gauss_pca/v'+str(var)+'g'+str(gpca)+'c'+str(cpca)+'.csv', index=False)
    # df_y.to_csv('feature_eng_y.csv', index=False)

vars = [.5,1,2]
gpcas = [100,150,200]
cpcas = [25,50,75]

for var in vars:
    for gpca in gpcas:
        for cpca in cpcas:
            to_csv(var, gpca, cpca)