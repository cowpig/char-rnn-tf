from rnn import *
from tf_ptb import reader



if __name__ == '__main__':

    filepath = './data/ptb_examples/data'

    train_data, valid_data, test_data, _ = reader.ptb_raw_data(filepath)

    n_steps = 3
    batch_size = 1
    input_size = 10**4
    stack_size = 2
    learning_rate=0.01

    params = [{
        "input_size": input_size,
        "output_size": input_size,
        "batch_size": batch_size,
    },
    {
        "input_size": input_size,
        "output_size": input_size,
        "batch_size": batch_size,
    }]


    states_0 = initial_state(params)


    x_in, y_in, states_in, states_out, y_out, costs = build_graph(params, n_steps, batch_size, stack_size)

    tvars = tf.trainable_variables()
    grads = tf.gradients(costs, tvars)
    optimus_prime = tf.train.GradientDescentOptimizer(learning_rate)
    train = optimus_prime.apply_gradients(zip(grads, tvars))

    with tf.Session() as sesh:

        state = states_0
        sesh.run(tf.initialize_all_variables())

        cost = np.inf

        i=0
        while i < 5000:
            for (x_idxs, y_idxs) in reader.ptb_iterator(train_data, batch_size, n_steps):
                x = np.array([np.eye(input_size)[idx] for idx in x_idxs]).astype(np.float32)
                y = np.array([np.eye(input_size)[idx] for idx in y_idxs]).astype(np.float32)

                state, out, cost, _ = sesh.run([states_out, y_out, costs, train],
                                feed_dict={x_in:x, y_in:y, states_in:state})

                if i % 1 == 0:
                    print "cost at epoch {}: {}".format(i, cost)

                if i % 100 == 0:
                    print "predictions:\n{}".format(out)

                i+=1
