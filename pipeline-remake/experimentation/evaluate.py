import pandas as pd
import numpy as np
np.random.seed(1001)
import tensorflow as tf
tf.random.set_seed(221)
from crossvalidation import *
import optuna
import sys
sys.path.insert(1, '../models')
import keras


from arch_base import Model

def full_test(params):
    df_x = pd.read_csv(params['feature_csv'])
    df_y = pd.read_csv(params['target_csv'])

    seed_losses = []
    seed_aucs = []
    seeds = 3

    for seed in range(seeds):
        datasets = get_strat_folds(df_x, df_y, 5, seed)

        i = 0
        target_y = pd.DataFrame(columns=df_y.columns)
        pred_y = pd.DataFrame(columns=df_y.columns)
        fold_loss = -1
        fold_auc = -1
        for fold in datasets:
            i += 1
            train_x, train_y = fold['train']
            test_x, test_y = fold['test']
            
            myModel = Model(df_x, df_y, params)
            model = myModel.run_training(train_x, train_y, test_x, test_y)

            if i == 1:
                target_y = test_y
                pred_y = pd.DataFrame(model.predict(test_x), columns=df_y.columns)
            else:
                target_y = target_y.append(test_y)
                pred_y = pred_y.append(pd.DataFrame(model.predict(test_x), columns=df_y.columns))

            fold_loss, fold_auc = myModel.get_eval(pred_y, target_y)
            print("Fold " + str(i) + ": " + str(fold_loss) + " loss, " + str(fold_auc) + " auc")
        
        seed_losses.append(fold_loss)
        seed_aucs.append(fold_auc)
        print('-'*40)
        print("Seed " + str(seed+1)+ ":")
        print(fold_loss, fold_auc)

    print("Average performance: " + str(sum(seed_losses)/seeds) + ", " + str(sum(seed_aucs)/seeds))
    log_evaluation(params, sum(seed_losses)/seeds, sum(seed_aucs)/seeds)
    
def log_evaluation(params, loss, auc):
    print('log_evaluation: writing experiment to csv')
    df = pd.read_csv('../logs/experiment_results.csv')
    params['loss'] = loss
    params['auc'] = auc
    df = df.append(params, ignore_index=True)
    df = df.sort_values(by='loss')
    df.to_csv('../logs/experiment_resultsv2.csv', index=False)

params = {}
#Select data
# params['feature_csv'] = '../processing/gauss_pca2/v1.8g100c40.csv'
params['target_csv'] = '../processing/feature_eng_y.csv'
params['network'] = 'tolg_018_kerninit'

# Select hyperparameters
params['dropout'] = .7
params['learning_rate'] = 0.00472
params['batch_size'] = 100
params['label_smoothing'] = 0
params['layers'] = -1
params['neurons'] = -1

#Info for logging
params['extra_inf'] = ''

def evaluate(vars, gpcas, cpcas):
    for var in vars:
        for gpca in gpcas:
            for cpca in cpcas:
                path = '../processing/all_gauss_pca/v'+str(var)+'g'+str(gpca)+'c'+str(cpca)+'.csv'
                params['feature_csv'] = path
                
                full_test(params)

params['network'] = 'fat_lrelu'
params['label_smoothing'] = 0.000
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.001
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.002
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.003
evaluate([.9], [300], [40])

params['network'] = 'tolg_018_kerninit'
params['learning_rate'] = 0.0009904
params['label_smoothing'] = 0.000
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.001
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.002
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.003
evaluate([.9], [300], [40])

