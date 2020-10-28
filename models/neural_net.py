import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from sklearn import metrics

#TODO Think of proper way to store / tweak model settings

#Lower batch size, way lower learning rate 500 ~ e-5!
def train_model(df_train_x, df_train_y, df_test_x, df_test_y, learning_rate):
    #Architecture
    inputs = keras.Input(shape=(len(df_train_x.columns)))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dropout(.2)(x)
    x = tfa.layers.WeightNormalization(layers.Dense(1024, activation='elu'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.AlphaDropout(.2)(x)
    x = tfa.layers.WeightNormalization(layers.Dense(1024, activation='elu'))(x)
    x = layers.BatchNormalization()(x)
    outputs = tfa.layers.WeightNormalization(layers.Dense(len(df_train_y.columns),activation="sigmoid"))(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="moa-first-try")
    
    #Training parameters
    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=[keras.metrics.Precision(), keras.metrics.Recall()],
    )
    
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min', min_lr=1E-5)
    
    class_weight = {}
    for x in range(len(df_train_y.columns)):
        class_weight[x] = 1

    history = model.fit(
        df_train_x.to_numpy(),
        df_train_y.to_numpy(),
        batch_size=2000,
        epochs=200,
        validation_data=(df_test_x, df_test_y),
        callbacks=[early_stop, reduce_lr],
        class_weight=class_weight)     

    return model

def evaluate_model(df_test_x, df_test_y, model):
    loss = model.evaluate(df_test_x.to_numpy(), df_test_y.to_numpy(), verbose=1)[0]
    
    m = tf.keras.metrics.AUC()
    m.update_state(df_test_y.to_numpy(), model.predict(df_test_x))
    auc = m.result().numpy()
    return loss, auc


