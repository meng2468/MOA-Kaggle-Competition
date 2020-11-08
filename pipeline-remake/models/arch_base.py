import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow_addons as tfa
from sklearn import metrics


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
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation='sigmoid',bias_initializer=None))(x)
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
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation='sigmoid',bias_initializer=bias))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="tolg_018")
            return model
        
        def fat_lrelu():
            inputs = tf.keras.Input(shape=(features))
            x = layers.BatchNormalization()(inputs)
            x = tfa.layers.WeightNormalization(layers.Dense(1500, kernel_initializer="he_normal"))(x)
            x = layers.LeakyReLU()(x)

            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            x = tfa.layers.WeightNormalization(layers.Dense(1500, kernel_initializer="he_normal"))(x)
            x = layers.LeakyReLU()(x)

            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation='sigmoid',bias_initializer=bias))(x)
            model = keras.Model(inputs=inputs, outputs=outputs, name="fat_lrelu")
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

    
    def run_training(self, df_train_x, df_train_y, df_test_x, df_test_y):
        def ls(y_true,y_pred):
            return tf.keras.losses.binary_crossentropy(y_true,y_pred,label_smoothing=self.label_smoothing)

        self.model.compile(
                loss=ls,
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate)
            )

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=1E-5, restore_best_weights=True, baseline=None)
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min', min_lr=1E-5)
            
        history = self.model.fit(
        df_train_x.to_numpy(),
        df_train_y.to_numpy(),
        batch_size=self.batch_size,
        epochs=40,
        validation_data=(df_test_x, df_test_y),
        callbacks=[early_stop,reduce_lr])

        return self.model

    def get_eval(self, df_test_x, df_test_y):
        # loss = self.model.evaluate(df_test_x.to_numpy(), df_test_y.to_numpy(), verbose=1)#[0]
        m = tf.keras.metrics.BinaryCrossentropy()
        m.update_state(df_test_y.to_numpy(), self.model.predict(df_test_x))
        loss = m.result().numpy()
        m = tf.keras.metrics.AUC()
        m.update_state(df_test_y.to_numpy(), self.model.predict(df_test_x))
        auc = m.result().numpy()
        return loss, auc


