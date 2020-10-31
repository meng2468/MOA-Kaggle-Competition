import pandas as pd
from crossvalidation import *
import keras
import optuna
import sys
sys.path.insert(1, '../models')
from arch_base import Model

def quick_test(params):
    df_x = pd.read_csv(params['feature_csv'])
    df_y = pd.read_csv(params['target_csv'])

    datasets = get_strat_folds(df_x, df_y, 5)
    fold_losses = []
    fold_aucs = []
    i = 0
    for fold in datasets:
        i += 1
        train_x, train_y = fold['train']
        test_x, test_y = fold['test']
        
        myModel = Model(len(df_x.columns), len(df_y.columns), params)
        myModel.dropout = params['dropout']
        myModel.learning_rate = params['learning_rate']
        myModel.batch_size = params['batch_size']
        myModel.run_training(train_x, train_y, test_x, test_y)
        
        loss, auc = myModel.get_eval(test_x, test_y)
        fold_losses.append(loss)
        fold_aucs.append(auc)

        print("Fold " + str(i) + ": " + str(loss) + " loss, " + str(auc) + " auc")
    
    print(fold_losses, fold_aucs)

def full_test(params):
    df_x = pd.read_csv(params['feature_csv'])
    df_y = pd.read_csv(params['target_csv'])

    seed_losses = []
    seed_aucs = []
    seeds = 3
    for seed in range(seeds):
        datasets = get_strat_folds(df_x, df_y, 5)
        fold_losses = []
        fold_aucs = []
        i = 0
        for fold in datasets:
            i += 1
            train_x, train_y = fold['train']
            test_x, test_y = fold['test']
            
            myModel = Model(len(df_x.columns), len(df_y.columns), params)
            myModel.run_training(train_x, train_y, test_x, test_y)
            
            loss, auc = myModel.get_eval(test_x, test_y)
            fold_losses.append(loss)
            fold_aucs.append(auc)

            print("Seed " + str(seed+1) + " Fold " + str(i) + ": " + str(loss) + " loss, " + str(auc) + " auc")
        
        seed_losses.append(sum(fold_losses)/len(fold_losses))
        seed_aucs.append(sum(fold_aucs)/len(fold_aucs))
        print('-'*40)
        print("Seed " + str(seed+1)+ ":")
        print(sum(fold_losses)/len(fold_losses), sum(fold_aucs)/len(fold_aucs))
        
    print("Average performance: " + str(sum(seed_losses)/seeds) + ", " + str(sum(seed_aucs)/seeds))
    log_evaluation(params, sum(seed_losses)/seeds, sum(seed_aucs)/seeds)

def log_evaluation(params, loss, auc):
    print('log_evaluation: writing experiment to csv')
    df = pd.read_csv('../logs/experiment_results.csv')
    params['loss'] = loss
    params['auc'] = auc
    df = df.append(params, ignore_index=True)
    df.to_csv('../logs/experiment_results.csv', index=False)

params = {}
#Select data
params['feature_csv'] = '../processing/feature_eng_ogpca_x.csv'
params['target_csv'] = '../processing/feature_eng_y.csv'

# Select hyperparameters
params['dropout'] = 0.11916016032516948
params['learning_rate'] = 0.003903996608447723
params['batch_size'] = 500
params['label_smoothing'] = 0

#Info for logging
params['extra_inf'] = ''

full_test(params)

#Current best (No robust scaling):
# params['dropout'] = 0.2093013646952079
# params['learning_rate'] = 0.003305851970846805
# params['batch_size'] = 900
# Average performance: 0.016828974584738414, 0.8263675848642985