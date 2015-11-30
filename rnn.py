import tensorflow as tf
import numpy as np

class Cell(object):
    def __init__(self, x_in, c_in, h_in, input_size, n_hidden, batch_size, cell_size):
        # HYPERPARAMETERS
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size

        # tensors and placeholders
        self.x_in = x_in
        self.c_in = c_in
        self.h_in = h_in

        # TODO: names ??
        # TODO: initialize biases as zeros ? (see Karpathy)
        # weights
        self.w_f = w_f = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+2]), name="w_f")
        self.w_i = w_i = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+2]), name="w_i")
        self.w_c = w_c = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+2]), name="w_c")
        self.w_o = w_o = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+2]), name="w_o")

        self.ones_for_bias_wgts = ones_for_bias_wgts = tf.constant(np.ones([batch_size,1,1]), name="b", dtype=tf.float32)

        self.x_with_bias = x_with_bias = tf.concat(1, [ones_for_bias_wgts, x])
        self.h_with_bias = h_with_bias = tf.concat(1, [ones_for_bias_wgts, h_in])
        self.x_h_concat = x_h_concat = tf.concat(1, [h_with_bias, x_with_bias])

        # forget gate layer
        self.f = f = tf.sigmoid(tf.batch_matmul(w_f, x_h_concat))

        # candidate values
        self.i = i = tf.sigmoid(tf.batch_matmul(w_i, x_h_concat))
        self.candidate_c = candidate_c = tf.tanh(tf.batch_matmul(w_c, x_h_concat))

        # new cell state (hidden)
        # forget old values of c
        self.old_c_to_keep = old_c_to_keep = tf.mul(f, c_in)
        # scaled candidate values of c
        self.new_c_to_keep = new_c_to_keep = tf.mul(i, candidate_c)
        self.c = c = tf.add(old_c_to_keep, new_c_to_keep)

        # new scaled output
        self.o = o = tf.sigmoid(tf.batch_matmul(w_o, x_h_concat))
        self.h = h = tf.mul(o, tf.tanh(c))


        self.params = [w_f, w_i, w_c, w_o]


def build_stack(stack_size, x_in, h_in, c_in, hyperparameters):
    cells = []
    cells.append(Cell(x_in, h_in, c_in, **hyperparameters))
    for i` in xrange(stack_size-1):
        with tf.variable_scope("Cell_{}".format(i)):
            cells.append(Cell(cells[-1].h, , cells[-1].c, **hyperparameters))



def build_graph(TODO):
    x_in = tf.placeholder(tf.float32, name="x", shape=(batch_size,input_size,1))
    # output from LSTM at t-1
    h_in = tf.Variable(tf.random_normal([batch_size,n_hidden,1]), name="h_in")
    # cell state from LSTM at t-1
    c_in = tf.Variable(tf.zeros([batch_size,cell_size,1]), name="c_in")

    all_params = [param for cell in cells for param in cell.params]
    gradients = tf.GradientOptimizer(cost, all_params)

    y_out = tf.nn.softmax(h)

def add_cell_to(geriatric_cell, x, params):
    return Cell(x, geriatric_cell.c, geriatric_cell.h, **params)

def cross_entropy(observed, actual):
    return -tf.reduce_sum(actual*tf.log(observed))

def run_tf_sesh():
    with tf.Session() as sesh:
        data = np.ones((batch_size,input_size,1))
        init = tf.initialize_all_variables()
        sesh.run(init)
        print sesh.run(c, feed_dict={x: data})
        print sesh.run(h, feed_dict={x: data})


def run_epoch(session):
    # cross entropy
    cost_function = cross_entropy()

    with session as sesh:



# hyperparameters
input_size = 4
n_hidden = 3
batch_size = 1
cell_size = 14

if name == '__main__':
    initialize_graph(input_size, n_hidden, batch_size, cell_size)
    run_tf_sesh()
