import pandas as pd
import processing.preprocessing as pp
import processing.crossvalidation as cv
import processing.data_processing as khdp
import models.neural_net as nn

def save_cleaned_data():
    pp.initial_data_parse()
    pp.generate_train_csv()

def save_augmented_data():
    df_train_x = pd.read_csv('./input/train_features.csv')
    df_train_y = pd.read_csv('./input/train_targets_scored.csv')
    df_test_x = pd.read_csv('./input/test_features.csv')

    df_train_x, df_train_y, _ = khdp.preprocessing_pipeline(df_train_x, df_train_y, df_test_x)

    df_train_x.loc[:, df_train_x.columns[1:]].to_csv('./processed-input/kh_train_features.csv', index=False)
    df_train_y.loc[:, df_train_y.columns[1:]].to_csv('./processed-input/kh_train_targets.csv', index=False)

#Tweak actual net in neural_net
def run_experiment(learning_rate):
    df_train_x, df_train_y = pp.get_training_data(kh=True)
    datasets = cv.get_folds(df_train_x, df_train_y, 5)

    avg_loss = []
    avg_auc = []
    seeds = 3

    for seed in range(seeds):
        losses = []
        aucs = []
        i = 0
        for fold in datasets:
            i += 1
            print("Running for Seed " + str(seed+1) + ", Fold " + str(i))
            train_x, train_y = fold['train']
            test_x, test_y = fold['test']

            model = nn.train_model(train_x, train_y, test_x, test_y, learning_rate)
            loss, auc = nn.evaluate_model(test_x, test_y, model)
            losses.append(loss)
            aucs.append(auc)
            print("AUC: " + str(auc))

        avg_loss.append(sum(losses)/len(losses))
        avg_auc.append(sum(aucs)/len(aucs))

    for seed in range(seeds):
        print("Seed " + str(seed+1) + ": " + str(avg_loss[seed]) + " loss, " + str(avg_auc[seed]) + " auc")
    return [learning_rate, sum(avg_loss)/len(avg_loss), sum(avg_auc)/len(avg_auc)]

def train_and_save():
    df_train_x, df_train_y = pp.get_training_data()
    model = nn.train_model(df_train_x, df_train_y, save=True, name="nn.007.515")


results = []
for lr in [.008,.01,.03,.05]:
    results.append(run_experiment(lr))

print(results)