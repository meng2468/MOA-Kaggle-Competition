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

def get_aug_df(df_x):
    df_x = cp_time(df_x)
    df_x = cp_dose(df_x)
    df_x = gene_exp(df_x)
    df_x = cell_via(df_x)

    return df_x

def to_csv(df_x):
    df_x = cp_time(df_x)
    df_x = cp_dose(df_x)
    df_x = gene_exp(df_x)
    df_x = cell_via(df_x)

    df_x.to_csv('feature_augm.csv', index=False)