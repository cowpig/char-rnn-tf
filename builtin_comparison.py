import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell

def cross_entropy(observed, actual):
    return -tf.reduce_sum(actual*tf.log(observed))


if __name__ == '__main__':

    n_steps = 4
    batch_size = 1
    input_size = 3
    stack_size = 1
    learning_rate=0.7
    hyperparameters = [{
        "input_size": input_size,
        "output_size": input_size,
        "batch_size": batch_size,
    } for _ in xrange(stack_size)]

    x_data = np.array([[[1,0,0],[0,1,0],[0,0,1],[1,0,0]]])
    y_data = np.array([[[0,1,0],[0,0,1],[1,0,0],[0,1,0]]])

    x_in = tf.placeholder(tf.float32, name="x", shape=(batch_size,n_steps,input_size))
    y_in = tf.placeholder(tf.float32, name="y", shape=(batch_size,n_steps,input_size))

    lstm_cell = rnn_cell.BasicLSTMCell(input_size, forget_bias=0.0)
    stack = rnn_cell.MultiRNNCell([lstm_cell]*stack_size)

    cellstate = stack.zero_state(batch_size, tf.float32)
    y_outs = []

    for t in xrange(n_steps):
        if t > 0: tf.get_variable_scope().reuse_variables()
        x = tf.squeeze(tf.slice(x_in, [0, t, 0], [-1, 1, -1]), [1])
        cellstate, out = stack(x, cellstate)
        y_outs.append(out)

    y_out = tf.nn.softmax(tf.concat(1, y_outs))


    with tf.Session() as sesh:
        tf.initialize_all_variables().run()
        print sesh.run(y_out, {x_in:x_data})


    # cost = cross_entropy(y_out, y_in)
    