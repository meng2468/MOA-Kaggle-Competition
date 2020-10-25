import processing.preprocessing as pp
import processing.crossvalidation as cv
import models.neural_net as nn

def load_data():
    pp.initial_data_parse()
    pp.generate_train_csv

#Tweak actual net in neural_net
def run_experiment():
    df_train_x, df_train_y = pp.get_training_data()
    datasets = cv.get_folds(df_train_x, df_train_y, 4)

    losses = []
    accurracies = []
    i = 0
    for fold in datasets:
        i += 1
        print("Running for Fold: " + str(i))
        train_x, train_y = fold['train']
        test_x, test_y = fold['test']

        model = nn.train_model(train_x, train_y)
        loss, accurracy = nn.evaluate_model(test_x, test_y, model)
        losses.append(loss)
        accurracies.append(accurracy)

    for i in range(len(accurracies)):
        print("Fold " + str(i) + ": " + str(losses[i]) + " loss, " + str(accurracies[i]) + " accuracy")
    
    print("Average Loss: " + str(sum(losses)/len(losses)))
    print("Average Accuracy: " + str(sum(accurracies)/len(accurracies)))

def train_and_save():
    df_train_x, df_train_y = pp.get_training_data()
    model = nn.train_model(df_train_x, df_train_y, save=True, name="trialv1")

train_and_save()