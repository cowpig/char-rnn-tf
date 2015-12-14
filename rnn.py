import tensorflow as tf
import numpy as np
import layers

def build_graph(hyperparams, n_steps, batch_size, stack_size, seed=1):
    cells = []
    for i in xrange(stack_size):
        with tf.variable_scope("Cell_{}".format(i)):
            cells.append(LSTM(**hyperparams[i]))

    # states handle both h_in and c_in
    total_state_size = sum([param['output_size']*2 for param in hyperparams])
    states_in = tf.placeholder(tf.float32, name="states_in",
                                shape=(batch_size, 1, total_state_size))
    x_in = tf.placeholder(tf.float32, name="x",
                                shape=(batch_size,n_steps,cells[0].input_size))
    y_in = tf.placeholder(tf.float32, name="y",
                                shape=(batch_size,n_steps,cells[-1].output_size))


    y_arr = []
    states = []
    n=0
    for cell in cells:
        m = cell.output_size*2
        states.append(tf.slice(states_in, [0,0,n], [-1,-1,m]))
        n = m

    for t in xrange(n_steps):
        next_states = []
        # print "x_in", x_in.get_shape()
        x_at_t = tf.slice(x_in, [0, t, 0], [-1, 1, -1])
        out = x_at_t
        for i, cell in enumerate(cells):
            # print 'x ', x.get_shape
            # print 'h ', h_arr[i].get_shape()
            # print 'c ', c_arr[i].get_shape()

            c, out = cell.build_layer(x_in=out, state=states[i],
                                    scope="Cell_{}_t_{}".format(i,t))
            next_states.append(tf.concat(2, [c, out]))

        states = next_states

        # final_1 = fully_connected_layer(tf.squeeze())
        final = fully_connected_layer(tf.squeeze(out, [1]), activation=tf.identity, 
                                                scope="fully_conn_t{}".format(t))
        # print "out: ", out.get_shape()
        vec = tf.nn.softmax(final)
        y_arr.append(tf.expand_dims(vec, 1))

    y_out = tf.concat(1, y_arr)
    states_out = tf.concat(2, states)
    # print "y_out: ", y_out.get_shape()
    # print "y_in: ", y_in.get_shape()
    cost = cross_entropy(y_out, y_in)

    return {
        'x_in': x_in,
        'y_in': y_in,
        'states_in': states_in,
        'states_out': states_out,
        'y_out': y_out,
        'cost' : cost
    }


def initial_state(hyperparams):
    total_state_size = sum([param['output_size']*2 for param in hyperparams])
    return np.zeros((hyperparams[0]['batch_size'], 1, total_state_size))



if __name__ == '__main__':
    config = {
        "model" : {
            "layers" : [
                {
                    "type" : layers.FullyConnected,
                    "input_size" : input_size,
                    "output_size": input_size,
                    "activation" : tf.nn.sigmoid
                },
                {
                    "type" : layers.LSTM,
                    "input_size": input_size,
                    "output_size": 8,
                    "batch_size": batch_size,
                },
                {
                    "type" : layers.LSTM,
                    "input_size": 8,
                    "output_size": input_size,
                    "batch_size": batch_size,
                },
                {
                    "type" : layers.FullyConnected,
                    "input_size" : input_size,
                    "output_size": input_size,
                    "activation" : tf.nn.softmax
                }
            ]
        }
        "training" : {
            "learning_rate" : 0.1,
            "n_steps" : 4,
            "batch_size" : 1,
            "cost" : cross_entropy
        }
    }

    state_0 = initial_state(params)

    x_in, y_in, states_in, states_out, y_out, costs = build_graph(params, n_steps, batch_size, stack_size)

    tvars = tf.trainable_variables()
    grads = tf.gradients(costs, tvars)
    optimus_prime = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimus_prime.apply_gradients(zip(grads, tvars))

    def data_iterator():
        x = np.array([[ [1,0,0],[0,1,0],[0,0,1],[1,0,0]],
                      [ [1,0,0],[0,1,0],[0,0,1],[1,0,0]] ])

        y = np.array([[ [1,0,0],[0,1,0],[0,0,1],[1,0,0]],
                      [ [1,0,0],[0,1,0],[0,0,1],[1,0,0]] ])
        yield (x, y)

    train(data_iterator, state_0, x_in, y_in, states_in, states_out, y_out, costs, train_op)
