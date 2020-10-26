import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa

#TODO Think of proper way to store / tweak model settings

#Lower batch size, way lower learning rate 500 ~ e-5!
def train_model(df_train_x, df_train_y, save=False, name="trialv1"):
    inputs = keras.Input(shape=(len(df_train_x.columns)))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dropout(.1)(x)
    x = tfa.layers.WeightNormalization(layers.Dense(1024, activation='elu'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.AlphaDropout(.1)(x)
    x = tfa.layers.WeightNormalization(layers.Dense(1024, activation='elu'))(x)
    x = layers.BatchNormalization()(x)
    outputs = tfa.layers.WeightNormalization(layers.Dense(len(df_train_y.columns),activation="tanh"))(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="moa-first-try")

    model.compile(
        loss=keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=.0001),
        metrics=["accuracy"],
    )
    
    early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min', min_lr=1E-5)
    
    class_weight = {}
    for x in range(len(df_train_y.columns)):
        class_weight[x] = 1

    history = model.fit(df_train_x.to_numpy(), df_train_y.to_numpy(), batch_size=500, epochs=200, validation_split=0.1, callbacks=[early_stop, reduce_lr], class_weight=class_weight)     
    
    if save:
        model.save('./models/pre-trained/'+name) 
    
    return model

def evaluate_model(df_test_x, df_test_y, model):
    test_scores = model.evaluate(df_test_x.to_numpy(), df_test_y.to_numpy(), verbose=1)
    return test_scores[0], test_scores[1]


