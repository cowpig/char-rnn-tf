import rnn
import data
import tensorflow as tf
import numpy as np

if __name__ == "__main__":

    dataset = data.DataSet('./data/letsgetmeta.txt')
    dataset.idx['train'] = (0,1024)

    n_steps = 32
    batch_size = 1 # data module has no support for batches yet
    input_size = dataset.n_chars
    stack_size = 2
    learning_rate=0.01

    params = [{
        "input_size": input_size,
        "output_size": 512,
        "batch_size": batch_size,
    },
    {
        "input_size": 512,
        "output_size": input_size,
        "batch_size": batch_size,
    }]


    states_0 = rnn.initial_state(params)

    x_in, y_in, states_in, states_out, y_out, costs = rnn.build_graph(params, n_steps, batch_size, stack_size)

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

            for (x,y) in dataset.yield_examples(steps=n_steps):
                
                state, out, cost, _ = sesh.run([states_out, y_out, costs, train],
                                feed_dict={x_in:np.array([x]), 
                                            y_in:np.array([y]), 
                                            states_in:state})

            if i % 1 == 0:
                print "cost at epoch {}: {}".format(i, cost)

            # if i % 10 == 0:
            #     readable_x = ''.join(dataset.convert(x))
            #     readable_out = []

            #     indexed_probs =sorted([idxed for idxed in enumerate(out)],
            #                                 key=lambda x: x[1], reverse=True) 
            #     top_5_idxs = [i[0] for i in indexed[:5]]
            #     top_5_chars = [np.eye(dataset.n_chars)[i] for i in top_5_idxs]
                
            #     readable_output = ''.join(dataset.convert(top_5_chars))
                
            #     print "input seq: {}".format(readable_x)
            #     print "predictions:\n{}".format(out)

            i+=1
    # we need a print function
    # stuff
