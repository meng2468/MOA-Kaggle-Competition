import pandas as pd
from crossvalidation import *
import keras
import optuna
import sys
sys.path.insert(1, '../models')
from arch_base import Model

def tuning_objective(trial):
    params = {}
    # Select data
    params['feature_csv'] = '../processing/real_gauss_pca/v0.8g600c60.csv'
    params['target_csv'] = '../processing/feature_eng_y.csv'

    # Select hyperparameters
    params['dropout'] = 0.2
    # params['learning_rate'] = .003305851970846805
    # params['batch_size'] = 500
    params['label_smoothing'] = 0.001
    
    # Select tuning
    params['batch_size'] = trial.suggest_int('batch_size', 100, 1500, 100)
    params['learning_rate'] = trial.suggest_loguniform('lr', 2e-4, 5e-3)
    # params['dropout'] = trial.suggest_float('dropout', 0.1, .4)
    # params['label_smoothing'] = trial.suggest_loguniform('label_smoothing', 1e-5,1e-3)

    df_x = pd.read_csv(params['feature_csv'])
    df_y = pd.read_csv(params['target_csv'])
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
    study.optimize(tuning_objective, n_trials=10)
    print(study.best_params)

param_tuning()