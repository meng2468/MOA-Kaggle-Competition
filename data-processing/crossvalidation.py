import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

df_train_x = pd.read_csv('../processed-input/clean_train_features.csv')
df_train_y = pd.read_csv('../processed-input/clean_train_targets.csv')

skf = KFold(n_splits=5)
skf.get_n_splits(df_train_x, df_train_y)

for train_index, test_index in skf.split(df_train_x, df_train_y):
    print(train_index, test_index)