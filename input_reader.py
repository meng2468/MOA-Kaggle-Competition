import numpy as np
import pandas as pd

import processing.data_reader as reader
from processing.data_processing import *


def preprocessing_pipeline(train_features, train_targets, test_features):
    
    train_len = train_features.shape[0]
    test_len = test_features.shape[0]
    
    data = pd.concat((train_features, test_features))
    
    ## TODO: save removed columns for testset transfromation
    data = remove_variance_encoding(data)

    ## TODO: save PCA value for testset transfromation
    data = add_PCA_feature(data)
    
    
    train_features, test_features = split_moa_train_test(data, train_len, test_len)
    
    train_features, train_targets = remove_ctl_vehicle(train_features, train_targets)
    
    print("\n=== Train ===")
    train_features = drop_cp_type(train_features)
    train_features = one_hot_encode_moa(train_features)

    print("\n=== Test ===")
    print("TODO: read raw test data")
    print("TODO: apply pipeline")
    test_features = drop_cp_type(test_features)
    test_features = one_hot_encode_moa(test_features)
    
    
    return train_features, train_targets, test_features


train_features, train_targets, test_features, _ = reader.read_dataset('/Users/ckh/Desktop/清华TsingHua/Sem1/_code/ML/Project/lish-moa/')
train_features, train_targets, test_features = preprocessing_pipeline(train_features, train_targets, test_features)