import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = (np.array([[0,0], [0,1], [1,0], [1,1]]), np.array([0,1,1,0])), (np.array([[0,1], [1,0]]), np.array([1,1]))

X = tf.placeholder(tf.float32, shape=(None, 2), name="input")
Y = tf.placeholder(tf.float32, shape=(None), name="output")

def model(X):
	inputLayer_w = tf.Variable(tf.random_normal(shape=[2, 3]), dtype=tf.float32)
	inputLayer_b = tf.Variable(tf.zeros(3))
	inputLayer = tf.matmul(X, inputLayer_w) + inputLayer_b
	inputLayer = tf.nn.relu(inputLayer)

	hiddenLayer_w = tf.Variable(tf.random_normal(shape=[3, 2]), dtype=tf.float32)
	hiddenLayer_b = tf.Variable(tf.zeros(2))
	hiddenLayer = tf.matmul(inputLayer, hiddenLayer_w) + hiddenLayer_b
	hiddenLayer = tf.nn.relu(hiddenLayer)

	outputLayer_w = tf.Variable(tf.random_normal(shape=[2, 1]), dtype=tf.float32)
	outputLayer_b = tf.Variable(tf.zeros(1))
	outputLayer = tf.matmul(hiddenLayer, outputLayer_w) + outputLayer_b
	outputLayer = tf.nn.sigmoid(outputLayer)

	return outputLayer

LEARNING_RATE = 0.005

logits = model(X)
loss_op = tf.reduce_mean(tf.losses.mean_squared_error(Y, logits))

correct_prediction = tf.equal(tf.round(logits), tf.round(Y))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = optimizer.minimize(loss_op)

sess = tf.Session()
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())

loss_list = []
accuracy_list = []

for i in range(1, 100):
    loss, _, acc = sess.run([loss_op, train_op, accuracy_operation], feed_dict={X: train_data, Y: train_labels})
    print("Loop: ", i, "\tLoss:", loss, "\tAccuracy: ", acc)
    loss_list.append(loss)
    accuracy_list.append(acc * 100)

test_loss, _, test_acc = sess.run([loss_op, train_op, accuracy_operation], feed_dict={X:test_data, Y:test_labels})
print("TEST\nLoss: ", test_loss, "\tAccuracy:", test_acc)