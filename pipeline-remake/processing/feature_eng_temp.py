import pandas as pd
import numpy as np

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

def get_eng_df(df_x, df_y):
    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)

    return df_x, df_y

def to_csv(df_x, df_y):
    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)

    df_x.to_csv('feature_eng_temp_x.csv', index=False)
    df_y.to_csv('feature_eng_temp_y.csv', index=False)


