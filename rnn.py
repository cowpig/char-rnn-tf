import tensorflow as tf
import numpy as np

# hyperparameters
n_inputs = 4
n_hidden = 3
batch_size = 1
cell_size = 14


# tensors and placeholders
x = tf.placeholder(tf.float32, name="x", shape=(batch_size,n_inputs,1))
#w = tf.Variable(tf.random_normal([n_hidden,n_inputs+1,batch_size], name="w"))
# TODO: names ??
w_f = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+n_inputs+2]), name="w_f")
w_i = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+n_inputs+2]), name="w_i")
w_c = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+n_inputs+2]), name="w_c")
w_o = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+n_inputs+2]), name="w_o")
h_ = tf.Variable(tf.random_normal([batch_size,n_hidden,1]), name="h")
c_ = tf.Variable(tf.random_normal([batch_size,cell_size,1]), name="c_")

ones_for_biases = tf.constant(np.ones([batch_size,1,1]), name="b", dtype=tf.float32)

x_with_bias = tf.concat(1, [ones_for_biases, x])
h_with_bias = tf.concat(1, [ones_for_biases, h_])
x_h_concat = tf.concat(1, [h_with_bias, x_with_bias])

#act = tf.tanh(tf.batch_matmul(w, x_with_bias))

# forget gate layer
f = tf.sigmoid(tf.batch_matmul(w_f, x_h_concat))

# candidate values
i = tf.sigmoid(tf.batch_matmul(w_i, x_h_concat))
candidate_c = tf.tanh(tf.batch_matmul(w_c, x_h_concat))

# new cell state
# forget old values of c
old_c_to_keep = tf.mul(f, c_)
# scaled candidate values of c
new_c_to_keep = tf.mul(i, candidate_c)
c = tf.add(old_c_to_keep, new_c_to_keep)

# new hidden state
o = tf.sigmoid(tf.batch_matmul(w_o, x_h_concat))
h = tf.mul(o, tf.tanh(c))


with tf.Session() as sesh:
    data = np.ones((batch_size,n_inputs,1))
    init = tf.initialize_all_variables()
    sesh.run(init)
    #print sesh.run(act, feed_dict={x: data})
    print sesh.run(c, feed_dict={x: data})
    print sesh.run(h, feed_dict={x: data})
