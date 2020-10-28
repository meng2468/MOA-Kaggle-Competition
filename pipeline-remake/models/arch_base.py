import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn import metrics

class Model:
    def __init__(self, features, targets):
        self.dropout = .2
        self.learning_rate = .001
        self.batch_size = 2000

        
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
                metrics=[keras.metrics.Precision(), keras.metrics.Recall()],
            )

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min', min_lr=1E-5)
            
        history = self.model.fit(
        df_train_x.to_numpy(),
        df_train_y.to_numpy(),
        batch_size=self.batch_size,
        epochs=120,
        validation_data=(df_test_x, df_test_y),
        callbacks=[early_stop, reduce_lr])

        return self.model

    def get_eval(self, df_test_x, df_test_y):
        loss = self.model.evaluate(df_test_x.to_numpy(), df_test_y.to_numpy(), verbose=1)[0]
        
        m = tf.keras.metrics.AUC()
        m.update_state(df_test_y.to_numpy(), self.model.predict(df_test_x))
        auc = m.result().numpy()
        return loss, auc


