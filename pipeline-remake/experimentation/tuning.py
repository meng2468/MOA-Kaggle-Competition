import pandas as pd
from crossvalidation import *
import keras
import optuna
import sys
sys.path.insert(1, '../models')
from arch_base import Model

df_x = pd.read_csv('../processing/feature_eng_temp_x.csv')
df_y = pd.read_csv('../processing/feature_eng_y.csv')

def tuning_objective(trial):
    params = {}
    # Select data
    params['feature_csv'] = '../processing/feature_eng_gpca_x.csv'
    params['target_csv'] = '../processing/feature_eng_y.csv'

    # Select hyperparameters
    params['dropout'] = 0.25
    params['learning_rate'] = .003305851970846805
    params['batch_size'] = 500
    params['label_smoothing'] = 0
    
    # Select tuning
    # params['batch_size'] = trial.suggest_int('batch_size', 200, 1200, 100)
    params['learning_rate'] = trial.suggest_loguniform('lr', 1e-4, 1e-2)
    # params['dropout'] = trial.suggest_float('dropout', 0.1, .4)
    # params['label_smoothing'] = trial.suggest_float('label_smoothing', 0,.002)

    datasets = get_strat_folds(df_x, df_y, 5)
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

        print("Fold " + str(i) + ": " + str(loss) + " loss, " + str(auc) + " auc")
    print(losses, aucs)

    return (sum(losses)/len(losses))

def param_tuning():
    study = optuna.create_study()
    study.optimize(tuning_objective, n_trials=50)
    print(study.best_params)

param_tuning()