import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler

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

def robust_scale(df_x):
    data = df_x.values
    data = np.transpose(data)
    transformer = RobustScaler().fit(data)
    data = np.transpose(transformer.transform(data))

    df_x = pd.DataFrame(data, columns=df_x.columns)
    return df_x


def get_eng_df(df_x, df_y):
    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)

    return df_x, df_y

def to_csv():
    df_x = pd.read_csv('feature_augm.csv')
    df_y = pd.read_csv('../data/train_targets_scored.csv')

    df_x, df_y = remove_sig_id(df_x, df_y)
    df_x, df_y = remove_ctl_v(df_x, df_y)
    df_x.to_csv('feature_eng_temp_x.csv', index=False)

    df_x = robust_scale(df_x)

    df_x.to_csv('feature_eng_temp_rob_x.csv', index=False)
    df_y.to_csv('feature_eng_temp_y.csv', index=False)

to_csv()
