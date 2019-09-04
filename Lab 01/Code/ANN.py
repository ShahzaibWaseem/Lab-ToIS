import tensorflow as tf
from tensorflow import keras
import numpy as np

(train_data, train_labels), (test_data, test_labels) = (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,1,1,0])), (np.array([[0,1], [1,0]]), np.array([1,1]))

model = keras.Sequential([
    keras.layers.Dense(2, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy'])
model.fit(train_data, train_labels, epochs=1000)

predictions = model.predict(test_data).round()
print(predictions)