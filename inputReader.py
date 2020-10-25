import numpy as np
import pandas as pd

import processing.dataReader as reader
import processing.dataProcessing as pp


def preprocessing_pipeline(train_features, train_targets, test_features):
    
    train_len = train_features.shape[0]
    test_len = test_features.shape[0]

    data = pd.concat((train_features, test_features))
    data = pp.remove_variance_encoding(data)
    data = pp.add_PCA_feature(data)

    train_features, test_features = pp.split_moa_train_test(data, train_len, test_len)

    train_features, train_targets = pp.remove_ctl_vehicle(train_features, train_targets)

    return train_features, train_targets, test_features


train_features, train_targets, test_features, _ = reader.read_dataset('/Users/ckh/Desktop/清华TsingHua/Sem1/_code/ML/Project/lish-moa/')
train_features, train_targets, test_features = preprocessing_pipeline(train_features, train_targets, test_features)