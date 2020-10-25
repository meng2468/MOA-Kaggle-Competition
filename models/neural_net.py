import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


#Lower batch size, way lower learning rate 500 ~ e-5!
def train_model(df_train_x, df_train_y, save=False):
    inputs = keras.Input(shape=(len(df_train_x.columns)))
    x = layers.BatchNormalization()(inputs)
    x = layers.Dropout(.2)(x)
    x = tfa.layers.WeightNormalization(layers.Dense(1024, activation='elu'))(x)
    x = layers.BatchNormalization()(x)
    x = layers.AlphaDropout(.2)(x)
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
    
    history = model.fit(df_train_x.to_numpy(), df_train_y.to_numpy(), batch_size=500, epochs=100, validation_split=0.1, callbacks=[early_stop, reduce_lr])     
    
    if save:
        name = 'l' + str(test_scores[0])[1:5] + '_a' + str(test_scores[1])[1:5]
        model.save('./nnets/great/'+name) 
    
    return model

def evaluate_model(df_test_x, df_test_y, model):
    test_scores = model.evaluate(df_test_x.to_numpy(), df_test_y.to_numpy(), verbose=1)
    return test_scores[0], test_scores[1]


