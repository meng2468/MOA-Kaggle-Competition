### General ###
import os
import sys
import copy
import tqdm
import pickle
import random
import warnings
import optuna
from datetime import date
warnings.filterwarnings("ignore")
sys.path.append('gauss_rank_scaler/')
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

### Data Wrangling ###
import numpy as np
import pandas as pd
from scipy import stats
from gauss_rank_scaler import GaussRankScaler

### Data Visualization ###
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")

### Machine Learning ###
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import QuantileTransformer
from sklearn.feature_selection import VarianceThreshold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

### Deep Learning ###
import torch
from torch import nn
import torch.optim as optim
from torch.nn import functional as F
from torch.nn.modules.loss import _WeightedLoss
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Tabnet 
from pytorch_tabnet.metrics import Metric
from pytorch_tabnet.tab_model import TabNetRegressor
# %% [code]
seed = 42

from colorama import Fore
c_ = Fore.CYAN
m_ = Fore.MAGENTA
r_ = Fore.RED
b_ = Fore.BLUE
y_ = Fore.YELLOW
g_ = Fore.GREEN

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_seed(seed)

# %% [markdown]
# ## <font color = "green">Configuration</font>

# %% [code]
# Parameters
data_path = ""
no_ctl = True
scale = "rankgauss"
variance_threshould = 0.8
decompo = "PCA"
ncompo_genes = 600
ncompo_cells = 50
encoding = "dummy"

# %% [markdown]
# ## <font color = "green">Loading the Data</font>

# %% [code]
train = pd.read_csv(data_path + "train_features.csv")
targets = pd.read_csv(data_path + "train_targets_scored.csv")
test = pd.read_csv(data_path + "test_features.csv")

submission = pd.read_csv(data_path + "sample_submission.csv")

# %% [markdown]
# # <font color = "seagreen">Preprocessing and Feature Engineering</font>

# %% [code]
if no_ctl:
    # cp_type == ctl_vehicle
    print(b_, "not_ctl")
    train = train[train["cp_type"] != "ctl_vehicle"]
    test = test[test["cp_type"] != "ctl_vehicle"]
    targets = targets.iloc[train.index]
    train.reset_index(drop = True, inplace = True)
    test.reset_index(drop = True, inplace = True)
    targets.reset_index(drop = True, inplace = True)


# %% [code]
GENES = [col for col in train.columns if col.startswith("g-")]
CELLS = [col for col in train.columns if col.startswith("c-")]

# %% [markdown]
# ## <font color = "green">Rank Gauss Process</font>

# %% [code]
data_all = pd.concat([train, test], ignore_index = True)
cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]
mask = (data_all[cols_numeric].var() >= variance_threshould).values
tmp = data_all[cols_numeric].loc[:, mask]
data_all = pd.concat([data_all[["sig_id", "cp_type", "cp_time", "cp_dose"]], tmp], axis = 1)
cols_numeric = [feat for feat in list(data_all.columns) if feat not in ["sig_id", "cp_type", "cp_time", "cp_dose"]]

### Rank Gauss ###
print(b_, "Rank Gauss")
scaler = GaussRankScaler()
data_all[cols_numeric] = scaler.fit_transform(data_all[cols_numeric])

# %% [markdown]
# ## <font color = "green">Principal Component Analysis</font>

# %% [code]
# PCA
if decompo == "PCA":
    print(b_, "PCA")
    GENES = [col for col in data_all.columns if col.startswith("g-")]
    CELLS = [col for col in data_all.columns if col.startswith("c-")]
    
    pca_genes = PCA(n_components = ncompo_genes,
                    random_state = seed).fit_transform(data_all[GENES])
    pca_cells = PCA(n_components = ncompo_cells,
                    random_state = seed).fit_transform(data_all[CELLS])
    
    pca_genes = pd.DataFrame(pca_genes, columns = [f"pca_g-{i}" for i in range(ncompo_genes)])
    pca_cells = pd.DataFrame(pca_cells, columns = [f"pca_c-{i}" for i in range(ncompo_cells)])
    data_all = pd.concat([data_all, pca_genes, pca_cells], axis = 1)
else:
    pass

# %% [markdown]
# ## <font color = "green">One Hot</font>

# %% [code]
# Encoding
data_all = pd.get_dummies(data_all, columns = ["cp_time", "cp_dose"])

# %% [code]
GENES = [col for col in data_all.columns if col.startswith("g-")]
CELLS = [col for col in data_all.columns if col.startswith("c-")]

for stats in tqdm.tqdm(["sum", "mean", "std", "kurt", "skew"]):
    data_all["g_" + stats] = getattr(data_all[GENES], stats)(axis = 1)
    data_all["c_" + stats] = getattr(data_all[CELLS], stats)(axis = 1)    
    data_all["gc_" + stats] = getattr(data_all[GENES + CELLS], stats)(axis = 1)

# %% [markdown]
# We can confirme that the shapes of data got close to the normal distribution.

# %% [code]
with open("data_all.pickle", "wb") as f:
    pickle.dump(data_all, f)

# %% [code]
with open("data_all.pickle", "rb") as f:
    data_all = pickle.load(f)

# %% [code]
# train_df and test_df
features_to_drop = ["sig_id", "cp_type"]
data_all.drop(features_to_drop, axis = 1, inplace = True)
try:
    targets.drop("sig_id", axis = 1, inplace = True)
except:
    pass
train_df = data_all[: train.shape[0]]
train_df.reset_index(drop = True, inplace = True)
test_df = data_all[train_df.shape[0]: ]
test_df.reset_index(drop = True, inplace = True)

# %% [code]
print(f"{b_}train_df.shape: {r_}{train_df.shape}")
print(f"{b_}test_df.shape: {r_}{test_df.shape}")

# %% [code]
X_test = test_df.values
print(f"{b_}X_test.shape: {r_}{X_test.shape}")

# %% [markdown]
# # <font color = "seagreen">Modeling</font>



# %% [markdown]
# ## <font color = "green">Custom Metric</font>

# %% [code]
class LogitsLogLoss(Metric):
    """
    LogLoss with sigmoid applied
    """

    def __init__(self):
        self._name = "logits_ll"
        self._maximize = False

    def __call__(self, y_true, y_pred):
        """
        Compute LogLoss of predictions.

        Parameters
        ----------
        y_true: np.ndarray
            Target matrix or vector
        y_score: np.ndarray
            Score matrix or vector

        Returns
        -------
            float
            LogLoss of predictions vs targets.
        """
        logits = 1 / (1 + np.exp(-y_pred))
        aux = (1 - y_true) * np.log(1 - logits + 1e-15) + y_true * np.log(logits + 1e-15)
        return np.mean(-aux)

from torch.nn.modules.loss import _WeightedLoss

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
# %% [markdown]
# ## <font color = "green">Model Parameters</font>

# %% [code]


# %% [markdown]
# # <font color = "seagreen">Training</font>

# %% [code]

NB_SPLITS = 5 # 7
mskf = MultilabelStratifiedKFold(n_splits = NB_SPLITS, random_state = 0, shuffle = True)

def objective(trial):
    test_cv_preds = []
    oof_preds = []
    oof_targets = []
    scores = []
    scores_auc = []

    MAX_EPOCH = 200
    # n_d and n_a are different from the original work, 32 instead of 24
    # This is the first change in the code from the original
    tabnet_params = dict(
        n_d = trial.suggest_int('n_d', 16, 64, 6),
        n_a = trial.suggest_int('n_a', 32, 128, 12),
        # n_d = 32,
        # n_a = 32,
        n_independent = 2,
        n_shared = 2,
        n_steps = 1,
        gamma = 1.3,
        lambda_sparse = 0,
        optimizer_fn = optim.Adam,
        optimizer_params = dict(lr = trial.suggest_loguniform("learning_rate", 1e-3, 1e-2), weight_decay = 1e-5), #2e-2 1e-5
        # optimizer_params = dict(lr = 2e-2, weight_decay = 1e-5), #2e-2 1e-5
        mask_type = "entmax",
        scheduler_params = dict(
            mode = "min", patience = 5, min_lr = 1e-5, factor = 0.9),
        scheduler_fn = ReduceLROnPlateau,
        seed = seed,
        verbose = 10
    )
    
    batch_size = trial.suggest_int('batch_size', 128, 1024, 128)
    virtual_batch_size = trial.suggest_int('virtual_bs', 32, 128, 32)
    # batch_size = 1024
    # virtual_batch_size = 32

    print('Running study with ' + str(batch_size) + 'bs, ' + str(virtual_batch_size) + 'vbs')
    for fold_nb, (train_idx, val_idx) in enumerate(mskf.split(train_df, targets)):
        print(b_,"FOLDS: ", r_, fold_nb + 1)
        print(g_, '*' * 60, c_)
        
        X_train, y_train = train_df.values[train_idx, :], targets.values[train_idx, :]
        X_val, y_val = train_df.values[val_idx, :], targets.values[val_idx, :]
        ### Model ###
        model = TabNetRegressor(**tabnet_params)
            
        ### Fit ###
        # Another change to the original code
        # virtual_batch_size of 32 instead of 128
        model.fit(
            X_train = X_train,
            y_train = y_train,
            eval_set = [(X_val, y_val)],
            eval_name = ["val"],
            eval_metric = ["logits_ll"],
            max_epochs = MAX_EPOCH,
            patience = 20,
            batch_size = batch_size, #1024, 
            virtual_batch_size = virtual_batch_size, #32,
            num_workers = 1,
            drop_last = False,
            loss_fn = SmoothBCEwLogits(smoothing=5e-5)
        )
        print(y_, '-' * 60)
        
        ### Predict on validation ###
        preds_val = model.predict(X_val)
        # Apply sigmoid to the predictions
        preds = 1 / (1 + np.exp(-preds_val))
        score = np.min(model.history["val_logits_ll"])
        
        ### Save OOF for CV ###
        oof_preds.append(preds_val)
        oof_targets.append(y_val)
        scores.append(score)
        
        ### Predict on test ###
        preds_test = model.predict(X_test)
        test_cv_preds.append(1 / (1 + np.exp(-preds_test)))

    oof_preds_all = np.concatenate(oof_preds)
    oof_targets_all = np.concatenate(oof_targets)
    test_preds_all = np.stack(test_cv_preds)

    # %% [code]
    aucs = []
    for task_id in range(oof_preds_all.shape[1]):
        aucs.append(roc_auc_score(y_true = oof_targets_all[:, task_id],
                                y_score = oof_preds_all[:, task_id]
                                ))
    print(f"{b_}Overall AUC: {r_}{np.mean(aucs)}")
    print(f"{b_}Average CV: {r_}{np.mean(scores)}")
    return np.mean(scores)

study_name = 'tabnet_600g50c'
for _ in range(100):
    study = optuna.create_study(study_name=study_name, storage='sqlite:///drive/MyDrive/moa/'+study_name+'.db', load_if_exists=True)
    study.optimize(objective, n_trials=3)
    df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df.to_csv('drive/MyDrive/moa/tb_tuning'+date.today()+'.csv')

# %% [markdown]
# # <font color = "seagreen">Submission</font>

# %% [code]
# all_feat = [col for col in submission.columns if col not in ["sig_id"]]
# # To obtain the same lenght of test_preds_all and submission
# test = pd.read_csv(data_path + "test_features.csv")
# sig_id = test[test["cp_type"] != "ctl_vehicle"].sig_id.reset_index(drop = True)
# tmp = pd.DataFrame(test_preds_all.mean(axis = 0), columns = all_feat)
# tmp["sig_id"] = sig_id

# submission = pd.merge(test[["sig_id"]], tmp, on = "sig_id", how = "left")
# submission.fillna(0, inplace = True)

#submission[all_feat] = tmp.mean(axis = 0)

# Set control to 0
#submission.loc[test["cp_type"] == 0, submission.columns[1:]] = 0
# submission.to_csv("submission.csv", index = None)
# submission.head()

# print(f"{b_}submission.shape: {r_}{submission.shape}")

# %% [markdown]
# <div class = "alert alert-block alert-info">
#     <h3><font color = "red">NOTE: </font></h3>
#     <p>If you want to comment please tag me with '@' to answer more quickly.</p>
# </div>