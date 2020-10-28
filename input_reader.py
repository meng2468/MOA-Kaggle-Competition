import numpy as np
import pandas as pd

import processing.data_reader as reader
from processing.data_processing import *
from processing.stratified_kfold import get_stratified_kfold_index
from models.torch_nn import *

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


X = train_features.iloc[:,1:]
Y = train_targets.iloc[:,1:]
X_test = test_features.iloc[:,1:]

PARAM = {
    ## Training
    "DEVICE" :  ('cuda' if torch.cuda.is_available() else 'cpu'),
    "EPOCHS" :  2,  #30
    "LEARNING_RATE" :  1e-3,
    "WEIGHT_DECAY" :  1e-5,
    "EARLY_STOPPING_STEPS" :  10,
    "EARLY_STOP" :  False,

    ## Dataloader
    "BATCH_SIZE" :  256,
    "NFOLDS" :  5,

    ## Model
    "NUM_FEATURE" : len(X.columns),
    "NUM_TARAGET" : len(Y.columns),
    "HIDDENT_SIZE" : 1024,
}

PARAM["DEVICE"]


fold_index = get_stratified_kfold_index(train_targets,n_split=PARAM["NFOLDS"])

seed = 0
    
for kfold in range(PARAM["NFOLDS"]):
    val_mask = fold_index == kfold
    saved_path = f"{seed}_FOLD{kfold}.pth"

    off_ = train_one_fold(kfold, X,Y, val_mask, saved_path, PARAM)
    prediction = prediction(X_test, saved_path, PARAM)