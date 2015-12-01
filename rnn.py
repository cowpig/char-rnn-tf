import tensorflow as tf
import numpy as np

class Cell(object):
    def __init__(self, input_size, n_hidden, batch_size, cell_size):
        # HYPERPARAMETERS
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size

        # weights
        dim_weights = [batch_size, cell_size, n_hidden + input_size]
	zeros_for_biases = tf.zeros(dim_weights[:-1] + [1])

        self.w_f = w_f = tf.Variable(tf.concat(2, tf.random_normal(dim_weights), zeros_for_biases), name="w_f")
        self.w_i = w_i = tf.Variable(tf.concat(2, tf.random_normal(dim_weights), zeros_for_biases), name="w_i")
        self.w_c = w_c = tf.Variable(tf.concat(2, tf.random_normal(dim_weights), zeros_for_biases), name="w_c")
        self.w_o = w_o = tf.Variable(tf.concat(2, tf.random_normal(dim_weights), zeros_for_biases), name="w_o")

        self.ones_for_bias_wgts = ones_for_bias_wgts = tf.constant(np.ones([batch_size,1,1]), name="b", dtype=tf.float32)

        self.params = [w_f, w_i, w_c, w_o]


    def build_node(self, x_in, c_in, h_in, scope="lstm_cell"):
        with tf.variable_scope(scope):
            x_with_h = tf.concat(1, [x_in, h_in])
            x_h_concat = tf.concat(1, [ones_for_bias_wgts, h_with_x])

            # forget gate layer
            f = tf.sigmoid(tf.batch_matmul(w_f, x_h_concat))

            # candidate values
            i = tf.sigmoid(tf.batch_matmul(w_i, x_h_concat))
            candidate_c = tf.tanh(tf.batch_matmul(w_c, x_h_concat))

            # new cell state (hidden)
            # forget old values of c
            old_c_to_keep = tf.mul(f, c_in)
            # scaled candidate values of c
            new_c_to_keep = tf.mul(i, candidate_c)
            c = tf.add(old_c_to_keep, new_c_to_keep)

            # new scaled output
            o = tf.sigmoid(tf.batch_matmul(w_o, x_h_concat))
            h = tf.mul(o, tf.tanh(c))
            return (c, h)

def cross_entropy(observed, actual):
    return -tf.reduce_sum(actual*tf.log(observed))

def build_graph(hyperparameters, n_steps, batch_size):
    cells = []
    for i in xrange(stack_size):
        with tf.variable_scope("Cell_{}".format(i)):
            cells.append(Cell(**hyperparameters[i]))

    x_in = tf.placeholder(tf.float32, name="x", shape=(batch_size,input_size,n_steps))
    y_in = tf.placeholder(tf.float32, name="y", shape=(batch_size,input_size,n_steps))
    h_arr = [[tf.Variable(tf.zeros([batch_size,cell.n_hidden,1]), name="h_in") for cell in cells]]
    c_arr = [[tf.Variable(tf.zeros([batch_size,cell.n_hidden,1]), name="c_in") for cell in cells]]
    y_arr = []
    
    # list of lists of (x, c, h) tuples
    # dimensions: n_steps by stack size by 3
    for t in xrange(n_steps):
        next_h = []
        next_c = []
        x = tf.slice(x_in, [0, 0, t], [-1, -1, 1])

        for i, cell in enumerate(cells):
            c, x = cell.build_node(x_in=x, h_in=h_arr[i], c_in=c_arr[i], scope="Cell_{}_t_{}".format(i,t))
            next_c.append(c)
            next_h.append(x)

        h_arr = next_h
        c_arr = next_c
        y_arr.append(tf.nn.softmax(x))

    all_params = [param for cell in cells for param in cell.params]
    cost = cross_entropy(tf.concat(2, y_arr), y_in)

    gradients = tf.GradientOptimizer(cost, all_params)

    return (x_in, y_in, all_params, cost, gradients)


def run_tf_sesh():
    with tf.Session() as sesh:
        data = np.ones((batch_size,input_size,1))
        init = tf.initialize_all_variables()
        sesh.run(init)
        print sesh.run(c, feed_dict={x: data})
        print sesh.run(h, feed_dict={x: data})


# def run_epoch(session):
    
#     # cross entropy
#     cost_function = cross_entropy()

#     with session as sesh:


if __name__ == '__main__':

    n_steps = 2
    batch_size = 1
    input_size = 3
    stack_size = 500
    hyperparameters = [{
        "input_size": input_size,
        "n_hidden": input_size,
        "batch_size": batch_size,
        "cell_size": 14
    } for _ in xrange(stack_size)]

    x = np.array([[[0,1,0],[1,0,0]]])
    y = np.array([[[1,0,0],[0,1,0]]])

    x_in, y_in, params, costs, gradients = build_graph(hyperparameters, n_steps, batch_size)

    with tf.Session() as sesh:
        print sesh.run(costs, {x_in:x, y_in:y})
