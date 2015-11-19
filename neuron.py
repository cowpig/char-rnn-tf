import tensorflow as tf
import numpy as np

n_inputs = 4
n_nodes = 2
batch_size = 1

x = tf.placeholder(tf.float32, name="x", shape=(batch_size, n_inputs, 1))
w = tf.Variable(tf.random_normal([n_nodes, n_inputs+1, batch_size], name="w"))
ones_for_biases = tf.constant(np.ones([batch_size, 1, 1]), name="b", dtype=tf.float32)

with_bias = tf.concat(1, [ones_for_biases, x])

act = tf.tanh(tf.mul(w, with_bias))

with tf.Session() as sesh:
	data = np.ones([n_inputs, 1, batch_size])
	init = tf.initialize_all_variables()
	sesh.run(init)
	print sesh.run(act, feed_dict={x: data})
