import tensorflow as tf
import numpy as np

class Cell(object):
    def __init__(self, input_size, n_hidden, batch_size, cell_size):
        # HYPERPARAMETERS
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size


        # TODO: names ??
        # TODO: initialize biases as zeros ? (see Karpathy)
        # weights
        self.w_f = w_f = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+1]), name="w_f")
        self.w_i = w_i = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+1]), name="w_i")
        self.w_c = w_c = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+1]), name="w_c")
        self.w_o = w_o = tf.Variable(tf.random_normal([batch_size,cell_size,n_hidden+input_size+1]), name="w_o")

        self.ones_for_bias_wgts = ones_for_bias_wgts = tf.constant(np.ones([batch_size,1,1]), name="b", dtype=tf.float32)

        self.params = [w_f, w_i, w_c, w_o]


    def build_node(self, scope="lstm_cell", x_in, c_in, h_in):
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


def build_graph():
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
        x = x_arr.slice(TODO)

        for i, cell in enumerate(cells):
            c, x = cell.build(scope="Cell_{}_t_{}".format(i,t), x_in=x, h_in=h_arr[i], c_in=c_arr[i])
            next_c.append(c)
            next_h.append(x)

        h_arr = next_h
        c_arr = next_c
        y_arr.append(tf.nn.softmax(x))

    all_params = [param for cell in cells for param in cell.params]
    cost = cross_entropy(tf.concat(2, y_arr), y_in)

    gradients = tf.GradientOptimizer(cost, all_params)



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


if name == '__main__':


    hyperparameters = {
        "input_size": 4,
        "n_hidden": 3,
        "batch_size": 1,
        "cell_size": 14
    }
    # initialize_graph(input_size, n_hidden, batch_size, cell_size)
    cell = Cell()
    run_tf_sesh()
