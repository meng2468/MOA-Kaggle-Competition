import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa


#Lower batch size, way lower learning rate 500 ~ e-5!

# tf.random.set_seed(12345)

df_train_x = pd.read_csv('../processed-input/proc_train_features.csv')
df_train_y = pd.read_csv('../processed-input/proc_train_targets.csv')

x_train = df_train_x[:20000].to_numpy()
y_train = df_train_y[:20000].to_numpy()

x_test = df_train_x[20000:].to_numpy()
y_test = df_train_y[20000:].to_numpy()

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
print(model.summary())

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=.001),
    metrics=["accuracy"],
)

early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=4)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, mode='min', min_lr=1E-5)

history = model.fit(x_train, y_train, batch_size=500, epochs=100, validation_split=0.2, callbacks=[early_stop, reduce_lr])

test_scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

name = 'l' + str(test_scores[0])[1:5] + '_a' + str(test_scores[1])[1:5]
model.save('./nnets/great/'+name)
