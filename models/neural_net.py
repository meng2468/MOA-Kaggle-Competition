import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(1234)

df_train_x = pd.read_csv('../processed-input/prep_train_features.csv')
df_train_y = pd.read_csv('../processed-input/train_target_p.csv')

x_train = df_train_x[:20000].to_numpy()
y_train = df_train_y[:20000].to_numpy()

x_test = df_train_x[20000:].to_numpy()
y_test = df_train_y[20000:].to_numpy()

inputs = keras.Input(shape=(len(df_train_x.columns)))
dense = layers.Dense(100, activation='relu')
x = dense(inputs)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)
x = layers.Dense(100, activation='relu')(x)

outputs = layers.Dense(len(df_train_y.columns))(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="moa-first-try")
print(model.summary())

model.compile(
    loss=keras.losses.mean_squared_logarithmic_error,
    optimizer=keras.optimizers.Adam(learning_rate=.005),
    metrics=["accuracy"],
)

history = model.fit(x_train, y_train, batch_size=20000, epochs=40, validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])