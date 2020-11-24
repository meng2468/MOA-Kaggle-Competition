import numpy as np 
import pandas as pd 

#TODO PCA

#Label correcting and normalisation
def initial_data_parse():
    df_train_x = pd.read_csv('./input/train_features.csv')
    df_train_x['cp_dose'] = df_train_x['cp_dose'].replace({'D1' : 1, 'D2': 2})

    df_train_y = pd.read_csv('./input/train_targets_scored.csv')

    df_train_x.to_csv('./processed-input/clean_train_features.csv', encoding='utf-8', index=False)
    df_train_y.to_csv('./processed-input/clean_train_targets.csv', encoding='utf-8', index=False)

#Data preperation for training
def generate_train_csv():
    print("Parsing Data")
    df_train_x = pd.read_csv('./processed-input/clean_train_features.csv')
    df_train_y = pd.read_csv('./processed-input/clean_train_targets.csv')
    
    df_train_x = df_train_x.drop('sig_id', axis=1)
    df_train_y = df_train_y.drop('sig_id', axis=1)
    features = list(df_train_x.columns)
    targets = list(df_train_y.columns)
    #Purge ctl_vehicle = 0 for the training set
    df_train = df_train_x
    for column in targets:
        df_train[column] = df_train_y[column]
    
    df_train = df_train[df_train['cp_type'] != 'ctl_vehicle']

    df_x = pd.DataFrame(columns=features)
    df_y = pd.DataFrame(columns=targets)

    print("Splitting merged dataframe")
    df_x = df_train.loc[:,features[1:]]
    df_y = df_train.loc[:, targets]

    df_train_x= df_train_x.drop('cp_type', axis=1)
    print(df_train_x.head())
    print(df_train_y.head())
    df_train_x.to_csv('./processed-input/proc_train_features.csv', encoding='utf-8', index=False)
    df_train_y.to_csv('./processed-input/proc_train_targets.csv', encoding='utf-8', index=False)

#Return training datasets
def get_training_data(kh):
    if kh:
        df_train_x = pd.read_csv('./processed-input/kh_train_features.csv')
        df_train_y = pd.read_csv('./processed-input/kh_train_targets.csv')
    else:
        df_train_x = pd.read_csv('./processed-input/proc_train_features.csv')
        df_train_y = pd.read_csv('./processed-input/proc_train_targets.csv')
    return df_train_x, df_train_y