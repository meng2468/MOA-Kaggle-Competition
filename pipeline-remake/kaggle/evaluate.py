import pandas as pd
import numpy as np
np.random.seed(1001)
import tensorflow as tf
tf.random.set_seed(221)
from sklearn import metrics
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import optuna

def get_strat_folds(df_features, df_targets, folds, state):
    #https://www.kaggle.com/tolgadincer/iter-strat-some-seeds-are-better-than-others
    # good_seeds = [14, 16, 77, 76, 34]
    # state = good_seeds[state]
    features = df_features.to_numpy()
    targets = df_targets.to_numpy()
    mlskf = MultilabelStratifiedKFold(n_splits = folds, shuffle=True, random_state=state)

    datasets = []
    for train_index, test_index in mlskf.split(features, targets):
        features_train, features_test = pd.DataFrame(features[train_index], columns=df_features.columns), pd.DataFrame(features[test_index], columns=df_features.columns)
        targets_train, targets_test = pd.DataFrame(targets[train_index], columns=df_targets.columns), pd.DataFrame(targets[test_index], columns=df_targets.columns)
        datasets.append({'train': (features_train, targets_train), 'test': (features_test, targets_test)})
    
    print(len(datasets))
    return datasets

class Model:
    def __init__(self, df_x, df_y, params):
        features = len(df_x.columns)
        targets = len(df_y.columns)
        bias = keras.initializers.Constant(df_y.mean().values)

        self.dropout = params['dropout']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']
        self.label_smoothing = params['label_smoothing']
        self.layers = params['layers']
        self.neurons = params['neurons']
        self.network = params['network']

        def base():
            inputs = keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = tfa.layers.WeightNormalization(layers.Dense(self.neurons, activation="relu"))(x)

            for _ in range(self.layers):
                x = layers.BatchNormalization()(x)
                x = layers.AlphaDropout(self.dropout)(x)
                x = tfa.layers.WeightNormalization(layers.Dense(self.neurons, activation="relu"))(x)

            x = layers.BatchNormalization()(x)
            x = layers.AlphaDropout(self.dropout)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation="sigmoid"))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="base")
            return model

        def starter_01901():
            inputs = keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = layers.Dropout(0.2)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(2048, activation="relu"))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(2048, activation="relu"))(x)
            x = layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets, activation="sigmoid"))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="starter_01901")
            return model

        def tl_01874():
            inputs = keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = layers.Dropout(0.3)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(480,kernel_initializer="he_normal"))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(tf.nn.leaky_relu)(x)
            x = layers.Dropout(0.4)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(256,kernel_initializer="he_normal"))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(tf.nn.leaky_relu)(x)
            x = layers.Dropout(0.2)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation="sigmoid",kernel_initializer="he_normal"))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="tl_01874")
            return model

        def tolg_018():
            inputs = tf.keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = tfa.layers.WeightNormalization(layers.Dense(800, activation='swish'))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(400, activation='swish'))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation='sigmoid',bias_initializer=bias))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="tolg_018")
            return model

        def tolg_018_kerninit():
            inputs = tf.keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = tfa.layers.WeightNormalization(layers.Dense(800, activation='swish', kernel_initializer="he_normal"))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(400, activation='swish', kernel_initializer="he_normal"))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation='sigmoid',kernel_initializer="he_normal", bias_initializer=bias))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="tolg_018")
            return model
        
        def fat_lrelu():
            inputs = tf.keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = tfa.layers.WeightNormalization(layers.Dense(1500, kernel_initializer="he_normal"))(x)
            x = layers.LeakyReLU()(x)

            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(1500, kernel_initializer="he_normal"))(x)
            x = layers.LeakyReLU()(x)

            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation='sigmoid',kernel_initializer="he_normal", bias_initializer=bias))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="fat_lrelu")
            return model
        
        def pytorch_cp():
            inputs = tf.keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = tfa.layers.WeightNormalization(layers.Dense(self.neurons, kernel_initializer="he_normal"))(x)
            x = layers.LeakyReLU()(x)

            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(self.neurons, kernel_initializer="he_normal"))(x)
            x = layers.LeakyReLU()(x)

            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation='sigmoid',kernel_initializer="he_normal", bias_initializer=bias))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="pytorch_cp")
            return model

        if self.network == 'base':
            self.model = base()
        if self.network == 'tl_01874':
            self.model = tl_01874()
        if self.network == 'tolg_018':
            self.model = tolg_018()
        if self.network == 'tolg_018_kerninit':
            self.model = tolg_018_kerninit()
        if self.network == 'fat_lrelu':
            self.model = fat_lrelu()
        if self.network == 'pytorch_cp':
            self.model = pytorch_cp()

    
    def run_training(self, df_train_x, df_train_y, df_test_x, df_test_y):
        def ls(y_true,y_pred):
            return tf.keras.losses.binary_crossentropy(y_true,y_pred,label_smoothing=self.label_smoothing)

        self.model.compile(
                loss=ls,
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
                metrics = [keras.losses.BinaryCrossentropy()]
            )

        early_stop = keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=3, min_delta=1E-5, restore_best_weights=True, baseline=None)
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_binary_crossentropy', factor=0.1, patience=2, mode='min', min_lr=1E-5)
            
        history = self.model.fit(
        df_train_x.to_numpy(),
        df_train_y.to_numpy(),
        batch_size=self.batch_size,
        epochs=30,
        validation_data=(df_test_x, df_test_y),
        callbacks=[early_stop,reduce_lr])

        return self.model

    def get_eval(self, df_pred_y, df_test_y):
        if len(df_test_y) >= 20000:
            tf.keras.backend.clear_session()
            m = tf.keras.metrics.BinaryCrossentropy()
            m.update_state(df_test_y.iloc[:10974,:].to_numpy(), df_pred_y.iloc[:10974,:].to_numpy())
            loss = m.result().numpy()
            m.update_state(df_test_y.iloc[10974:,:].to_numpy(), df_pred_y.iloc[10974:,:].to_numpy())
            loss = (loss + m.result().numpy())/2

            m = tf.keras.metrics.AUC()
            m.update_state(df_test_y.iloc[:10974,:].to_numpy(), df_pred_y.iloc[:10974,:].to_numpy())
            auc = m.result().numpy()
            m.update_state(df_test_y.iloc[10974:,:].to_numpy(), df_pred_y.iloc[10974:,:].to_numpy())
            auc = m.result().numpy()
            return loss, auc
        elif len(df_test_y) >= 15000:
            tf.keras.backend.clear_session()
            m = tf.keras.metrics.BinaryCrossentropy()
            m.update_state(df_test_y.iloc[:8779,:].to_numpy(), df_pred_y.iloc[:8779,:].to_numpy())
            loss = m.result().numpy()
            m.update_state(df_test_y.iloc[8779:,:].to_numpy(), df_pred_y.iloc[8779:,:].to_numpy())
            loss = (loss + m.result().numpy())/2

            m = tf.keras.metrics.AUC()
            m.update_state(df_test_y.iloc[:8779,:].to_numpy(), df_pred_y.iloc[:8779,:].to_numpy())
            auc = m.result().numpy()
            m.update_state(df_test_y.iloc[8779:,:].to_numpy(), df_pred_y.iloc[8779:,:].to_numpy())
            auc = m.result().numpy()
            return loss, auc


        tf.keras.backend.clear_session()
        m = tf.keras.metrics.BinaryCrossentropy()
        m.update_state(df_test_y.to_numpy(), df_pred_y.to_numpy())
        loss = m.result().numpy()


        m = tf.keras.metrics.AUC()
        m.update_state(df_test_y.to_numpy(), df_pred_y.to_numpy())
        auc = m.result().numpy()
        m.update_state(df_test_y.to_numpy(), df_pred_y.to_numpy())
        auc = m.result().numpy()
        return loss, auc

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
    df = pd.read_csv('experiment_resultsv3.csv')
    params['loss'] = loss
    params['auc'] = auc
    df = df.append(params, ignore_index=True)
    df = df.sort_values(by='loss')
    df.to_csv('experiment_resultsv3.csv', index=False)

params = {}
#Select data
# params['feature_csv'] = '../processing/gauss_pca2/v1.8g100c40.csv'
params['target_csv'] = '/content/drive/My Drive/moa_data/feature_eng_y.csv'
params['network'] = 'tolg_018_kerninit'

# Select hyperparameters
params['dropout'] = 0.5691029448658625
params['learning_rate'] = 0.00472
params['batch_size'] = 100
params['label_smoothing'] = 0
params['layers'] = -1
params['neurons'] = 640

#Info for logging
params['extra_inf'] = ''

def evaluate(vars, gpcas, cpcas):
    for var in vars:
        for gpca in gpcas:
            for cpca in cpcas:
                path = '/content/drive/My Drive/moa_data/v'+str(var)+'g'+str(gpca)+'c'+str(cpca)+'.csv'
                params['feature_csv'] = path
                
                full_test(params)

params['network'] = 'pytorch_cp'
params['learning_rate'] = 0.007937790536634029
params['dropout'] = 0.6

params['label_smoothing'] = 0
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.001
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.002
evaluate([.9], [300], [40])
params['label_smoothing'] = 0.003
evaluate([.9], [300], [40])

