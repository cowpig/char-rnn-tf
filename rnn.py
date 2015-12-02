import tensorflow as tf
import numpy as np

class Cell(object):
    def __init__(self, input_size, output_size, batch_size):
        # HYPERPARAMETERS
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size

        # weights
        weight_dims = [batch_size, input_size + output_size, output_size]
        zeros_for_biases = tf.zeros([batch_size, 1, output_size])

        # print tf.concat(2, [tf.random_normal(weight_dims), zeros_for_biases])

        self.w_f = w_f = tf.Variable(tf.concat(1, [zeros_for_biases, tf.random_normal(weight_dims)]), name="w_f")
        self.w_i = w_i = tf.Variable(tf.concat(1, [zeros_for_biases, tf.random_normal(weight_dims)]), name="w_i")
        self.w_c = w_c = tf.Variable(tf.concat(1, [zeros_for_biases, tf.random_normal(weight_dims)]), name="w_c")
        self.w_o = w_o = tf.Variable(tf.concat(1, [zeros_for_biases, tf.random_normal(weight_dims)]), name="w_o")

        self.params = [w_f, w_i, w_c, w_o]


    def build_node(self, x_in, c_in, h_in, scope="lstm_cell"):
        #print (x_in, c_in, h_in, scope)
        #print [type(thing) for thing in (x_in, c_in, h_in, scope)]
        # print [(item.name, item.dtype) for thing in (h_in, c_in) for item in thing]
        # print (x_in.name, x_in.dtype)

        with tf.variable_scope(scope):
            # print x.shape
            # print h_in.get_shape()
            x_with_h = tf.concat(2, [x_in, h_in])
            ones_for_bias_wgts = tf.constant(np.ones([batch_size,1,1]), name="b", dtype=tf.float32)
            x_h_concat = tf.concat(2, [ones_for_bias_wgts, x_with_h])

            # forget gate layer
            # print "w_f: ", self.w_f.get_shape()
            # print "x_h_concat: ", x_h_concat.get_shape()
            f = tf.sigmoid(tf.batch_matmul(x_h_concat, self.w_f))

            # candidate values
            i = tf.sigmoid(tf.batch_matmul(x_h_concat, self.w_i))
            candidate_c = tf.tanh(tf.batch_matmul(x_h_concat, self.w_c))

            # new cell state (hidden)
            # forget old values of c
            old_c_to_keep = tf.mul(f, c_in)
            # scaled candidate values of c
            new_c_to_keep = tf.mul(i, candidate_c)
            c = tf.add(old_c_to_keep, new_c_to_keep)

            # new scaled output
            o = tf.sigmoid(tf.batch_matmul(x_h_concat, self.w_o))
            h = tf.mul(o, tf.tanh(c))
            return (c, h)

def cross_entropy(observed, actual):
    return -tf.reduce_sum(actual*tf.log(observed))

def build_graph(hyperparameters, n_steps, batch_size):
    cells = []
    for i in xrange(stack_size):
        with tf.variable_scope("Cell_{}".format(i)):
            cells.append(Cell(**hyperparameters[i]))

    x_in = tf.placeholder(tf.float32, name="x", shape=(batch_size,n_steps,input_size))
    y_in = tf.placeholder(tf.float32, name="y", shape=(batch_size,n_steps,input_size))
    h_arr = [tf.Variable(tf.zeros([batch_size,1,cell.output_size]), name="h_in") for cell in cells]
    c_arr = [tf.Variable(tf.zeros([batch_size,1,cell.output_size]), name="c_in") for cell in cells]
    y_arr = []
    
    # list of lists of (x, c, h) tuples
    # dimensions: n_steps by stack size by 3
    for t in xrange(n_steps):
        next_h = []
        next_c = []
        x = tf.slice(x_in, [0, t, 0], [-1, 1, -1])

        for i, cell in enumerate(cells):
            # print 'x ', x.get_shape()
            # print 'h ', h_arr[i].get_shape()
            # print 'c ', c_arr[i].get_shape()
            c, x = cell.build_node(x_in=x, h_in=h_arr[i], c_in=c_arr[i], scope="Cell_{}_t_{}".format(i,t))
            next_c.append(c)
            next_h.append(x)

        h_arr = next_h
        c_arr = next_c
        vec = tf.nn.softmax(tf.squeeze(x, [0]))
        y_arr.append(tf.expand_dims(vec, 0))

    y_out = tf.concat(1, y_arr)
    cost = cross_entropy(y_out, y_in)

    return (x_in, y_in, cost, y_out)


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

    n_steps = 4
    batch_size = 1
    input_size = 3
    stack_size = 2
    learning_rate=0.7

    hyperparameters = [{
        "input_size": input_size,
        "output_size": 4,
        "batch_size": batch_size,
    },
    {
        "input_size": 4,
        "output_size": input_size,
        "batch_size": batch_size,

    }]

    x = np.array([[[1,0,0],[0,1,0],[0,0,1],[1,0,0]]])
    y = np.array([[[1,0,0],[0,1,0],[0,0,1],[1,0,0]]])


    x_in, y_in, costs, out = build_graph(hyperparameters, n_steps, batch_size)

    tvars = tf.trainable_variables()
    grads = tf.gradients(costs, tvars)
    optimus_prime = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimus_prime.apply_gradients(zip(grads, tvars))
    
    with tf.Session() as sesh:
        sesh.run(tf.initialize_all_variables())
        cost = np.inf
        i=0
        while i < 20000:
            cost, _ = sesh.run([costs, train], {x_in:x, y_in:y})
            if i % 100 == 0:
                print "cost at epoch {}: {}".format(i, cost)
            
            if i % 1000 == 0:
                print "predictions:\n{}".format(sesh.run([out], {x_in:x}))

            i+=1



