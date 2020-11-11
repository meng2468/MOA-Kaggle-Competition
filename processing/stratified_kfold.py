import numpy as np

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
# NOTE: Kaggle
# 1. add iterative-stratification package to input
#    link: https://www.kaggle.com/yasufuminakama/iterative-stratification
# 2. import the library from input
#    python > sys.path.append('../input/iterative-stratification/iterative-stratification-master')

def get_stratified_kfold_index(train_targets, n_split=5, random_state=42):
    """
    return array with fold index when then data row is target
    """
    kfold_indexes = np.zeros(len(train_targets), dtype=int)
    placeholder = kfold_indexes.copy()
    mskf = MultilabelStratifiedKFold(n_splits=n_split, random_state=random_state)
    for f, (t_idx, v_idx) in enumerate(mskf.split(X=placeholder, y=train_targets)):
        kfold_indexes[v_idx] = f

    return kfold_indexes