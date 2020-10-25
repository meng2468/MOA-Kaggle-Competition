import numpy as np 
import pandas as pd 

#TODO PCA

#Label correcting and normalisation
def initial_data_parse():
    df_train_x = pd.read_csv('../input/train_features.csv')
    df_train_x['cp_dose'] = df_train_x['cp_dose'].replace({'D1' : 1, 'D2': 2})

    df_train_y = pd.read_csv('../input/train_targets_scored.csv')

    df_train_x.to_csv('./processed-input/clean_train_features.csv', encoding='utf-8', index=False)
    df_train_y.to_csv('./processed-input/clean_train_targets.csv', encoding='utf-8', index=False)

#Data preperation for training
def generate_train_csv():
    print("Parsing Data")
    df_train_x = pd.read_csv('./processed-input/clean_train_features.csv')
    df_train_y = pd.read_csv('./processed-input/clean_train_targets.csv')
    
    df_train_x = df_train_x.drop('sig_id', axis=1)
    df_train_y = df_train_y.drop('sig_id', axis=1)

    #Purge ctl_vehicle = 0 for the training set
    df_train = df_train_x
    for column in df_train_y.columns:
        df_train[column] = df_train_y[column]
    
    df_train = df_train[df_train['cp_type'] != 'ctl_vehicle']

    df_train_x = pd.DataFrame(columns=df_train_x.columns)
    df_train_y = pd.DataFrame(columns=df_train_y.columns)

    for column in df_train_y.columns:
        df_train_y[column] = df_train[column]
    for column in df_train_x.columns[1:]:
        df_train_x[column] = df_train[column]

    df_train_x= df_train_x.drop('cp_type', axis=1)
    df_train_x.to_csv('./processed-input/proc_train_features.csv', encoding='utf-8', index=False)
    df_train_y.to_csv('./processed-input/proc_train_targets.csv', encoding='utf-8', index=False)

#Return training datasets
def get_training_data():
    df_train_x = pd.read_csv('./processed-input/proc_train_features.csv')
    df_train_y = pd.read_csv('./processed-input/proc_train_targets.csv')
    return df_train_x, df_train_y