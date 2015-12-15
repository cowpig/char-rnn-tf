import tensorflow as tf
import numpy as np


def train(dataset, starting_state, x_in, y_in, states_in, states_out, 
                                                y_out, costs, train_op):
    with tf.Session() as sesh:
        # logger = tf.train.SummaryWriter('./log', sesh.graph_def)
        # pylogger = tf.python.training.summary_io.SummaryWriter('./pylog', sesh.graph_def)

        state = starting_state
        sesh.run(tf.initialize_all_variables())
        # tf.train.write_graph(sesh.graph_def, './graph', 'rnn_graph.pbtxt')

        cost = np.inf

        i=0
        while True:
            for step, (x, y) in enumerate(dataset()):
                state, out, cost, _ = sesh.run([states_out, y_out, costs, train_op], 
                                feed_dict={x_in:x, 
                                            y_in:y, 
                                            states_in:state})

                # print tf.Graph().get_operations()

            if i % 100 == 0:
              print "cost at epoch {}: {}".format(i, cost)

            if i % 1000 == 0:
              print "predictions:\n{}".format(out)

            i+=1
