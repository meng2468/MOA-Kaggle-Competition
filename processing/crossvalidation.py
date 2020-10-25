import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


def get_folds(df_features, df_targets, folds):
    features = df_features.to_numpy()
    targets = df_targets.to_numpy()
    kf = KFold(n_splits=folds, shuffle=True)

    datasets = []
    for train_index, test_index in kf.split(features):
        fold = 0
        features_train, features_test = pd.DataFrame(features[train_index], columns=df_features.columns), pd.DataFrame(features[test_index], columns=df_features.columns)
        targets_train, targets_test = pd.DataFrame(targets[train_index], columns=df_targets.columns), pd.DataFrame(targets[test_index], columns=df_targets.columns)
        datasets.append({'train': (features_train, targets_train), 'test': (features_test, targets_test)})
    return datasets

