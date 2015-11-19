import tensorflow as tf
import numpy as np

n_inputs = 4
n_nodes = 3
batch_size = 1

x = tf.placeholder(tf.float32, name="x", shape=(batch_size,n_inputs,1))
#w = tf.Variable(tf.random_normal([n_nodes,n_inputs+1,batch_size], name="w"))
# TODO: names ??
w_f, w_i, w_c, w_o = (tf.Variable(tf.random_normal([batch_size,n_nodes,n_inputs+1])) for _ in xrange(4))
h = tf.Variable(tf.random_normal([batch_size,n_inputs,1], name="h"))
old_c = tf.Variable(tf.random_normal([batch_size,2,n_nodes], name="old_c"))

ones_for_biases = tf.constant(np.ones([batch_size,1,1]), name="b", dtype=tf.float32)

x_with_bias = tf.concat(1, [ones_for_biases, x])
h_with_bias = tf.concat(1, [ones_for_biases, h])
x_h_concat = tf.concat(2, [h_with_bias, x_with_bias])

#act = tf.tanh(tf.batch_matmul(w, x_with_bias))

# input gate layer
f = tf.sigmoid(tf.batch_matmul(w_f, x_h_concat))

# candidate values
i = tf.sigmoid(tf.batch_matmul(w_i, x_h_concat))
c = tf.tanh(tf.batch_matmul(w_c, x_h_concat))

# new cell state
new_c = tf.add(tf.batch_matmul(f, old_c), tf.batch_matmul(i, c))

# new hidden state
o = tf.sigmoid(tf.batch_matmul(w_o, x_h_concat))
new_h = tf.batch_matmul(o, tf.tanh(new_c))

#def lstm_module(data):
with tf.Session() as sesh:
    data = np.ones((batch_size,n_inputs,1))
    init = tf.initialize_all_variables()
    sesh.run(init)
    #print sesh.run(act, feed_dict={x: data})
    print sesh.run(new_c, feed_dict={x: data})
    print sesh.run(new_h, feed_dict={x: data})
