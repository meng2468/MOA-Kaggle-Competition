import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# tf.random.set_seed(12345)

df_train_x = pd.read_csv('../processed-input/prep_train_features.csv')
df_train_y = pd.read_csv('../processed-input/train_target_p.csv')

x_train = df_train_x[:20000].to_numpy()
y_train = df_train_y[:20000].to_numpy()

x_test = df_train_x[20000:].to_numpy()
y_test = df_train_y[20000:].to_numpy()

inputs = keras.Input(shape=(len(df_train_x.columns)))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(.1, input_shape=(512,))(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(.1, input_shape=(512,))(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(.1, input_shape=(512,))(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(.1, input_shape=(512,))(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(.1, input_shape=(512,))(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dropout(.1, input_shape=(512,))(x)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation='relu')(inputs)
x = layers.BatchNormalization()(x)

x = layers.Dense(512, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
outputs = layers.Dense(len(df_train_y.columns))(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="moa-first-try")
print(model.summary())

model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=.003),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=20000, epochs=2000, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])