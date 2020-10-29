import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn import metrics

class Model:
    def __init__(self, features, targets, params):
        self.dropout = params['dropout']
        self.learning_rate = params['learning_rate']
        self.batch_size = params['batch_size']

        
        inputs = keras.Input(shape=(features))
        x = layers.BatchNormalization()(inputs)
        x = layers.Dropout(self.dropout)(x)

        x = tfa.layers.WeightNormalization(layers.Dense(1024, activation='elu'))(x)
        x = layers.BatchNormalization()(x)
        x = layers.AlphaDropout(self.dropout)(x)

        x = tfa.layers.WeightNormalization(layers.Dense(1024, activation='elu'))(x)
        x = layers.BatchNormalization()(x)
        outputs = tfa.layers.WeightNormalization(layers.Dense(targets,activation="sigmoid"))(x)

        self.model = keras.Model(inputs=inputs, outputs=outputs, name="base")
    
    def run_training(self, df_train_x, df_train_y, df_test_x, df_test_y):
        self.model.compile(
                loss=keras.losses.BinaryCrossentropy(),
                optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            )

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min', min_lr=1E-5)
            
        history = self.model.fit(
        df_train_x.to_numpy(),
        df_train_y.to_numpy(),
        batch_size=self.batch_size,
        epochs=100,
        validation_data=(df_test_x, df_test_y),
        callbacks=[early_stop, reduce_lr])

        return self.model

    def get_eval(self, df_test_x, df_test_y):
        loss = self.model.evaluate(df_test_x.to_numpy(), df_test_y.to_numpy(), verbose=1)#[0]
        
        m = tf.keras.metrics.AUC()
        m.update_state(df_test_y.to_numpy(), self.model.predict(df_test_x))
        auc = m.result().numpy()
        return loss, auc


