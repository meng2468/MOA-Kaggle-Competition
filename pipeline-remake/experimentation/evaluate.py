import pandas as pd
from crossvalidation import *
import sys
sys.path.insert(1, '../models')
from arch_base import Model

def quick_test(df_x, df_y):
    datasets = get_folds(df_x, df_y, 5)
    losses = []
    aucs = []
    i = 0
    for fold in datasets:
        i += 1
        train_x, train_y = fold['train']
        test_x, test_y = fold['test']
        
        myModel = Model(len(df_x.columns), len(df_y.columns))
        myModel.run_training(train_x, train_y, test_x, test_y)
        
        loss, auc = myModel.get_eval(test_x, test_y)
        losses.append(loss)
        aucs.append(auc)

        print("Fold " + str(i) + ": " + str(loss) + " loss, " + str(auc) + " auc")
    
    print(losses, aucs)

def run_test():
    df_x = pd.read_csv('../processing/feature_eng_temp_x.csv')
    df_y = pd.read_csv('../processing/feature_eng_temp_y.csv')
    quick_test(df_x, df_y)

run_test()