import numpy as np
import pandas as pd
import random
import os
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.nn.modules.loss import _WeightedLoss

import sys
sys.path.append("..")
from processing.stratified_kfold import get_stratified_kfold_index

DEFAULT_PARAM = {
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
    "MODEL": "RESNET",
    "NUM_FEATURE" : 930,
    "NUM_TARAGET" : 206,
    "HIDDENT_SIZE" : 1024,
    "LOSS_SMOOTHING": 0.001, # remove if not apply loss smoothing
    "DROPOUT" : 0.25,
    "RELU_TYPE": "LEAKY", # "BASIC" | "LEAKY"
}

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct

class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct

def torch_train(model, optimizer, scheduler, loss_fn, dataloader, device):
    model.train()
    final_loss = 0

    for data in dataloader:
        optimizer.zero_grad()
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss /= len(dataloader)

    return final_loss

def torch_valid(model, loss_fn, dataloader, device):
    model.eval()
    final_loss = 0
    valid_preds = []

    for data in dataloader:
        inputs, targets = data['x'].to(device), data['y'].to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        final_loss += loss.item()
        valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

    final_loss /= len(dataloader)
    valid_preds = np.concatenate(valid_preds)

    return final_loss, valid_preds

def torch_inference(model, dataloader, device):
    model.eval()
    preds = []

    for data in dataloader:
        inputs = data['x'].to(device)

        with torch.no_grad():
            outputs = model(inputs)

        preds.append(outputs.sigmoid().detach().cpu().numpy())

    preds = np.concatenate(preds)

    return preds

class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout=0.5, relu_type="BASIC"):
        super(Model, self).__init__()
        self.relu_type = relu_type

        self.batch_norm1 = nn.BatchNorm1d(num_features)
#         self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x1 = self.batch_norm1(x)
#         x1 = self.dropout1(x1)
        if self.relu_type == "LEAKY":
            x1 = F.leaky_relu(self.dense1(x1))
        else:
            x1 = F.relu(self.dense1(x1))

        x2 = self.batch_norm2(x1)
        x2 = self.dropout2(x2)
        if self.relu_type == "LEAKY":
            x2 = F.leaky_relu(self.dense2(x2))
        else:
            x2 = F.relu(self.dense2(x2))

        x3 = self.batch_norm3(x2)
        x3 = self.dropout3(x3)
        x3 = self.dense3(x3)

        print(x1.shape)
        print(x2.shape)
        print(x3.shape)


        return x3

class Model_Res(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size, dropout=0.5, relu_type="BASIC"):
        super(Model_Res, self).__init__()

        self.relu_type = relu_type

        self.batch_norm1 = nn.BatchNorm1d(num_features)
#         self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))

        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

        self.batch_norm3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(dropout)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size))

#         self.conv = nn.Conv1d(hidden_size, num_targets, 3)
        self.pool = nn.AvgPool1d(3)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

    def forward(self, x):
        x1 = self.batch_norm1(x)
#         x1 = self.dropout1(x1)
        if self.relu_type == "LEAKY":
            x1 = F.leaky_relu(self.dense1(x1))
        else:
            x1 = F.relu(self.dense1(x1))

        x2 = self.batch_norm2(x1)
        x2 = self.dropout2(x2)
        if self.relu_type == "LEAKY":
            x2 = F.leaky_relu(self.dense2(x2))
        else:
            x2 = F.relu(self.dense2(x2))

        x3 = self.batch_norm3(x2)
        x3 = self.dropout3(x3)
        x3 = self.dense3(x3)

        stack = torch.stack([x1,x2,x3], dim=2)
#         y = self.conv(stack)
        y = self.pool(stack)
        y = y.squeeze(-1)


        y = self.dense4(y)

        return y

def kfold_train_valid_dataloader(X, Y, valid_mask, batch_size):
    '''
    train valid split using valid_mask
    return train_dataloader, valid_dataloader
    '''

    x_train = X[~valid_mask].reset_index(drop=True).values
    y_train = Y[~valid_mask].reset_index(drop=True).values

    x_valid = X[valid_mask].reset_index(drop=True).values
    y_valid = Y[valid_mask].reset_index(drop=True).values

    train_dataset = TrainDataset(x_train, y_train)
    valid_dataset = TrainDataset(x_valid, y_valid)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, validloader


def train_one_fold(kfold, X,Y, val_mask, saved_path, PARAM=DEFAULT_PARAM):
    trainloader, validloader = kfold_train_valid_dataloader(X, Y, val_mask, PARAM["BATCH_SIZE"])

    model_type = PARAM.get("MODEL", "NN")
    if model_type == 'RESNET':
        print("RESNET MODEL")
        model = Model_Res(
            num_features=PARAM["NUM_FEATURE"],
            num_targets=PARAM["NUM_TARAGET"],
            hidden_size=PARAM["HIDDENT_SIZE"],
            dropout=PARAM.get("DROPOUT", 0.5),
            relu_type=PARAM.get("RELU_TYPE", "BASIC"),
        )
    else:
        model = Model(
            num_features=PARAM["NUM_FEATURE"],
            num_targets=PARAM["NUM_TARAGET"],
            hidden_size=PARAM["HIDDENT_SIZE"],
            dropout=PARAM.get("DROPOUT", 0.5),
            relu_type=PARAM.get("RELU_TYPE", "BASIC"),
        )
    model.to(PARAM["DEVICE"])

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=PARAM["LEARNING_RATE"],
                                 weight_decay=PARAM["WEIGHT_DECAY"])

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=PARAM["EPOCHS"], steps_per_epoch=len(trainloader))


    loss_fn = nn.BCEWithLogitsLoss()
    loss_tr = nn.BCEWithLogitsLoss()

    loss_smooth = PARAM.get("LOSS_SMOOTHING", 0)
    if loss_smooth > 0:
        loss_tr = SmoothBCEwLogits(smoothing =loss_smooth)

    early_stopping_steps = PARAM["EARLY_STOPPING_STEPS"]
    early_step = 0

    ## training
    oof = np.zeros(Y.shape)
    best_loss = np.inf
    for epoch in range(PARAM["EPOCHS"]):

        train_loss = torch_train(model, optimizer,scheduler, loss_tr, trainloader, PARAM["DEVICE"])
        valid_loss, valid_preds = torch_valid(model, loss_fn, validloader, PARAM["DEVICE"])
        print(f"FOLD: {kfold}, EPOCH: {epoch}, train_loss: {train_loss}\tvalid_loss: {valid_loss}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            oof[val_mask] = valid_preds
            torch.save(model.state_dict(), saved_path)

        elif(PARAM["EARLY_STOP"] == True):
            early_step += 1
            if (early_step >= early_stopping_steps):
                print(f"EarlyStop @ EPOCH: {epoch}")
                break
    return oof

def run_k_fold(X,Y, X_test, seed, PARAM=DEFAULT_PARAM):
    seed_everything(seed)
    fold_index = get_stratified_kfold_index(Y,n_split=PARAM["NFOLDS"])

    train_count, target_count = Y.shape[0:2]
    test_count = X_test.shape[0]

    oof = np.zeros((train_count, target_count))
    predictions = np.zeros((test_count, target_count))

    for kfold in range(PARAM["NFOLDS"]):
        val_mask = fold_index == kfold
        saved_path = f"{seed}_FOLD{kfold}.pth"

        oof_ = train_one_fold(kfold, X,Y, val_mask, saved_path, PARAM)
        pred_ = torch_prediction(X_test, saved_path, PARAM)

        predictions += pred_
        oof += oof_

    predictions /= PARAM["NFOLDS"]
    return oof, predictions

def torch_prediction(X, saved_model, PARAM=DEFAULT_PARAM):
    testdataset = TestDataset(X.values)
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=PARAM["BATCH_SIZE"], shuffle=False)

    model_type = PARAM.get("MODEL", "NN")
    if model_type == 'RESNET':
        model = Model_Res(
            num_features=PARAM["NUM_FEATURE"],
            num_targets=PARAM["NUM_TARAGET"],
            hidden_size=PARAM["HIDDENT_SIZE"],
            relu_type=PARAM.get("RELU_TYPE", "BASIC"),
        )
    else:
        model = Model(
            num_features=PARAM["NUM_FEATURE"],
            num_targets=PARAM["NUM_TARAGET"],
            hidden_size=PARAM["HIDDENT_SIZE"],
            relu_type=PARAM.get("RELU_TYPE", "BASIC"),
        )
    model.load_state_dict(torch.load(saved_model))
    model.to(PARAM["DEVICE"])

    predictions = torch_inference(model, testloader, PARAM["DEVICE"])
    return predictions

class SmoothBCEwLogits(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth(targets:torch.Tensor, n_labels:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = targets * (1.0 - smoothing) + 0.5 * smoothing
        return targets

    def forward(self, inputs, targets):
        targets = SmoothBCEwLogits._smooth(targets, inputs.size(-1),
            self.smoothing)
        loss = F.binary_cross_entropy_with_logits(inputs, targets,self.weight)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss

def get_log_loss(Y, pred):
    if type(Y) == pd.DataFrame:
        Y = Y.values
    if type(pred) == pd.DataFrame:
        pred = pred.values
    scores = [log_loss(Y[:, i], pred[:, i]) for i in range(Y.shape[1])]
    return scores