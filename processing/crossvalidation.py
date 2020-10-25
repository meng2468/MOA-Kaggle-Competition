import pandas as pd
import numpy as np

def df_label_folds(df_features, df_targets, folds):
    #Shuffle
    df = df_features
    for column in df_targets:
        df[column] = df_targets[column]
    print(df.head())

    df_features['fold'] = 1
    df_targets['fold'] = 1
