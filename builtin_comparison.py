import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn_cell

def cross_entropy(observed, actual):
    return -tf.reduce_sum(actual*tf.log(observed))


if __name__ == '__main__':
    n_steps = 4
    batch_size = 1
    input_size = 3
    stack_size = 2
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
    initial_state = stack.zero_state(batch_size, tf.float32)

    cellstate = initial_state
    y_arr = []

    for t in xrange(n_steps):
        if t > 0: tf.get_variable_scope().reuse_variables()
        x = tf.squeeze(tf.slice(x_in, [0, t, 0], [-1, 1, -1]), [1])
        out, cellstate = stack(x, cellstate)
        # print "out is {}".format(out.get_shape())
        # reshaped = tf.reshape(out,[1,1,input_size])
        # print "reshaped is {}".format(reshaped.get_shape())
        vec = tf.nn.softmax(out)
        # print "vec is ", vec.get_shape()
        y_arr.append(tf.reshape(vec,[1,1,input_size]))

    # print len(y_outs)
    # print "y_in", y_in.get_shape()
    # print "you_outs[0]: {}".format(y_outs[0].get_shape())
    # print "concated: {}".format(tf.concat(1, y_outs).get_shape())
    
    # y_arr.append(tf.expand_dims(vec, 0))


    # y_out = tf.nn.softmax(tf.concat(1, y_arr))
    y_out = tf.concat(1, y_arr)
    # print "y_out", y_out.get_shape()
    # print "y_in", y_in.get_shape()
    costs = cross_entropy(y_out, y_in)
    #print "cost", cost.get_shape()
    
    tvars = tf.trainable_variables()
    
    for t in tvars:
        print t.name
    
    grads = tf.gradients(costs, tvars)
    optimus_prime = tf.train.GradientDescentOptimizer(learning_rate)
    # print optimus_prime
    train = optimus_prime.apply_gradients(zip(grads, tvars))
    # print train

    # cost = np.inf
    with tf.Session() as sesh:
        state = initial_state.eval()
        sesh.run(tf.initialize_all_variables())
        # out = sesh.run([y_out], {x_in:x_data})#, y_in:y_data})
        # print "out", out
        # diff = sesh.run(-tf.reduce_sum(y_in * tf.log(y_out)), {x_in:x_data, y_in:y_data})
        # print "diff", diff
        # cost = sesh.run([cost], {x_in:x_data, y_in:y_data})
        # print "cost", cost

        i=0
        while i < 5000:
            cost, state, out, _ = sesh.run([costs, cellstate, y_out, train], 
                                    {x_in:x_data, y_in:y_data, initial_state:state})

            # print tf.Graph().get_operations()

            if i % 100 == 0:
                print "cost at epoch {}: {}".format(i, cost)

            if i % 1000 == 0:
                print "predictions:\n{}".format(out)

            i+=1
    print "end"
    # cost = cross_entropy(y_out, y_in)
    