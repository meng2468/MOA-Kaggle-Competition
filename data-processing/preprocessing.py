import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

if __name__ == "__main__":          
    df_train_x = pd.read_csv('../input/train_features.csv')
    df_train_x = df_train_x.drop('sig_id', axis=1)
    df_train_x = df_train_x.loc[:, df_train_x.columns[1:]]
    df_train_x['cp_dose'] = df_train_x['cp_dose'].replace({'D1' : 1, 'D2': 2})

    df_train_y = pd.read_csv('../input/train_targets_scored.csv')
    df_train_y = df_train_y.loc[:,df_train_y.columns[1:]]

    df_train_x.to_csv('../processed-input/clean_train_features.csv', encoding='utf-8', index=False)
    df_train_y.to_csv('../processed-input/clean_train_targets.csv', encoding='utf-8', index=False)
