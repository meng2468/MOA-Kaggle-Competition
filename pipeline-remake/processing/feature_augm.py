import pandas as pd
import numpy as np
from sklearn import preprocessing

def cp_time(df_x):
    scaler = preprocessing.MinMaxScaler()
    df_x['cp_time'] = np.transpose(scaler.fit_transform(np.transpose([df_x['cp_time']])))[0]
    return df_x
    

def cp_dose(df_x):
    df_x['cp_dose'] = df_x['cp_dose'].replace({'D1' : 0, 'D2': 1})
    return df_x

def gene_exp(df_x):
    
    return df_x

def cell_via(df_x):
    return df_x

def gauss_trans(df_x):
    for col in (df_x.columns[4:]):
        data = np.transpose([df_x[col]])

        transformer = preprocessing.QuantileTransformer(n_quantiles=100,random_state=0, output_distribution="normal")
        data = np.transpose(transformer.fit_transform(data))[0]
        df_x[col] = data
    return df_x

def robust_scale(df_x):
    data = df_x.values
    data = np.transpose(data)
    transformer = RobustScaler().fit(data)
    data = np.transpose(transformer.transform(data))

    df_x = pd.DataFrame(data, columns=df_x.columns)
    return df_x

def get_aug_df(df_x):
    df_x = cp_time(df_x)
    df_x = cp_dose(df_x)
    df_x = gene_exp(df_x)
    df_x = cell_via(df_x)
    df_x = gauss_trans(df_x)

    return df_x

def to_csv(df_x):
    df_x = cp_time(df_x)
    df_x = cp_dose(df_x)
    df_x = gene_exp(df_x)
    df_x = cell_via(df_x)
    # df_x = gauss_trans(df_x)

    df_x.to_csv('feature_augm.csv', index=False)

to_csv(pd.read_csv('../data/train_features.csv'))