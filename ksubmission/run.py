import numpy as np
from helper import seed_everything
import dataset_reader as reader
from dataset_preprocessing import preprocessing_NN_meta_transform, preprocessing_NN_TL_transform
from model_helper import torch_inference
from model_NN_meta import Model as NN_Meta
from model_NN_meta import Model as NN_TL

seed_everything(42)
_, _, test_features_ori, _ = reader.read_dataset('../input/lish-moa/', include_non_score=False)
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')

## Model 1 - Meta
paths_dict = {
    'quantile':'quantile.pkl',
    'g_pca':'gpca.pkl',
    'c_pca':'cpca.pkl',
    'g_cluster':'g_cluster.pkl',
    'c_cluster':'c_cluster.pkl',
    'pca_cluster':'pca_cluster.pkl',
    'remove_col':'remove_col.pkl',
}

test_features = test_features_ori
test_features_Meta = preprocessing_NN_meta(test_features, paths_dict)

## Stage 1
# model1, load, predict -> unscored
# raw + unscored
test_data = test_features_Meta.iloc[:,1:]
testdataset = TestDataset(test_data.values)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=256, shuffle=False)

num_features = test_data.shape[1]
num_targets_unscored = 402
hidden_size = 2048
model = NN_Meta(num_features, num_targets_unscored, hidden_size)

predictions_meta = []
for saved_model in weight_files:
    model.load_state_dict(torch.load(saved_model))
    model.to(DEVICE)
    pred_ = torch_inference(model, testloader, DEVICE)
    predictions_meta.append(pred_)

meta_feature = np.mean(predictions_meta, axis=0)

## Stage 2
# model2, load, predict -> scored
# saved submission
test_data = pd.concat((test_features_Meta.reset_index(drop=True, inplace=False) , meta_feature), axis=1)

test_data = test_features_Meta.iloc[:,1:]
testdataset = TestDataset(test_data.values)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=256, shuffle=False)

num_features = test_data.shape[1]
num_targets=207
hidden_size = 2048
model = NN_Meta(num_features, num_targets, hidden_size)

model_meta_sub = []
for saved_model in weight_files:
    model.load_state_dict(torch.load(saved_model))
    model.to(DEVICE)
    pred_ = torch_inference(model, testloader, DEVICE)
    model_meta_sub.append(pred_)

model_meta_sub = np.mean(model_meta_sub, axis=2)



## Model 2 - TF
paths_dict = {
    'quantile':'quantile.pkl',
    'g_pca':'gpca.pkl',
    'c_pca':'cpca.pkl',
    'remove_col':'remove_col.pkl',
}

test_features = test_features_ori
test_features_TL = preprocessing_NN_TL_transform(test_features, paths_dict)

test_data = test_features_TL.iloc[:,1:]
testdataset = TestDataset(test_data.values)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=256, shuffle=False)

num_features = test_data.shape[1]
num_targets = 207
model = NN_TL(num_features, num_targets)

predictions_TL = []
for saved_model in weight_files:
    model.load_state_dict(torch.load(saved_model))
    model.to(DEVICE)
    pred_ = torch_inference(model, testloader, DEVICE)
    predictions_TL.append(pred_)

model_TL_sub = np.mean(predictions_TL, axis=2)
