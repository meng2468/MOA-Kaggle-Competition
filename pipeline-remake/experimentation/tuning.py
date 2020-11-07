import pandas as pd
import numpy as np
np.random.seed(1001)
import tensorflow as tf
tf.random.set_seed(221)
from crossvalidation import *
import keras
import optuna
import sys
sys.path.insert(1, '../models')
from arch_base import Model
import os

def tuning_objective(trial):
    params = {} 
    # Select data
    params['feature_csv'] = '../processing/all_gauss_pca/v0.85g450c55.csv'
    params['target_csv'] = '../processing/feature_eng_y.csv'

    # Select hyperparameters
    params['dropout'] = 0
    params['learning_rate'] = 0.001386656415113995
    params['batch_size'] = 200
    params['label_smoothing'] = 0
    params['layers'] = 1
    params['neurons'] = 1600
    
    # Select tuning
    params['batch_size'] = trial.suggest_int('batch_size', 100, 1500, 100)
    params['learning_rate'] = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    params['dropout'] = trial.suggest_float('dropout', .15, .3)
    params['label_smoothing'] = trial.suggest_loguniform('label_smoothing', 1e-6,1e-2)
    # params['layers'] = trial.suggest_int('layers', 1,5)
    # params['neurons'] = trial.suggest_int('neurons', 512,2048, 128)

    df_x = pd.read_csv(params['feature_csv'])
    df_y = pd.read_csv(params['target_csv'])
    datasets = get_strat_folds(df_x, df_y, 5, 1)
    losses = []
    aucs = []
    i = 0
    for fold in datasets:
        i += 1
        train_x, train_y = fold['train']
        test_x, test_y = fold['test']
        
        myModel = Model(len(df_x.columns), len(df_y.columns), params)
        myModel.run_training(train_x, train_y, test_x, test_y)
        
        loss, auc = myModel.get_eval(test_x, test_y)
        losses.append(loss)
        aucs.append(auc)
        trial.report(loss, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
        print("Fold " + str(i) + ": " + str(loss) + " loss, " + str(auc) + " auc")
    print(losses, aucs)

    return (sum(losses)/len(losses))

def param_tuning():
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(tuning_objective, n_trials=300)
    print(study.best_params)
    df_study = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    df_study.to_csv('overnight8545055.csv', index=False)

param_tuning()