import pandas as pd
from crossvalidation import *
import keras
import optuna
import sys
sys.path.insert(1, '../models')
from arch_base import Model

def quick_test():
    df_x = pd.read_csv('../processing/feature_eng_temp_x.csv')
    df_y = pd.read_csv('../processing/feature_eng_temp_y.csv')

    datasets = get_folds(df_x, df_y, 5)
    fold_losses = []
    fold_aucs = []
    i = 0
    for fold in datasets:
        i += 1
        train_x, train_y = fold['train']
        test_x, test_y = fold['test']
        
        myModel = Model(len(df_x.columns), len(df_y.columns))
        myModel.run_training(train_x, train_y, test_x, test_y)
        
        loss, auc = myModel.get_eval(test_x, test_y)
        fold_losses.append(loss)
        fold_aucs.append(auc)

        print("Fold " + str(i) + ": " + str(loss) + " loss, " + str(auc) + " auc")
    
    print(fold_losses, fold_aucs)

def full_test():
    df_x = pd.read_csv('../processing/feature_eng_temp_x.csv')
    df_y = pd.read_csv('../processing/feature_eng_temp_y.csv')

    seed_losses = []
    seed_aucs = []
    seeds = 3
    for seed in range(seeds):
        datasets = get_folds(df_x, df_y, 5)
        fold_losses = []
        fold_aucs = []
        i = 0
        for fold in datasets:
            i += 1
            train_x, train_y = fold['train']
            test_x, test_y = fold['test']
            
            myModel = Model(len(df_x.columns), len(df_y.columns))
            myModel.run_training(train_x, train_y, test_x, test_y)
            
            loss, auc = myModel.get_eval(test_x, test_y)
            fold_losses.append(loss)
            fold_aucs.append(auc)

            print("Seed " + str(seed+1) + " Fold " + str(i) + ": " + str(loss) + " loss, " + str(auc) + " auc")
        
        seed_losses.append(sum(fold_losses)/len(fold_losses))
        fold_aucs.append(sum(fold_aucs)/len(fold_aucs))
        print("Seed " + str(seed+1))
        print(sum(fold_losses)/len(fold_losses), sum(fold_aucs)/len(fold_aucs))
    print("Average performance: " + str(sum(seed_losses)/seeds) + ", " + str(sum(seed_aucs)/seeds))

full_test()