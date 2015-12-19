import rnn
import data
import tensorflow as tf
import numpy as np
from layers import LSTM, FullyConnected
import sys
import os.path
import config

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        fn = None

    if fn:
        if not os.path.isfile(fn):
            raise Exception("File '{}' does not exist.".format(fn))


    dataset = data.DataSet('./data/dum.txt',decoding_fx=lambda x: x)
    config_name = "test_lstm_no_embedding"
    conf = config.get_config(dataset.n_chars, config_name)

    graph = rnn.build_graph(conf)

    with tf.Session() as sesh:
        sesh.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        if fn:
            saver.restore(sesh, fn)

        t = graph['train']
        test = graph['test']
        train_state = np.zeros([conf['training']['batch_size'], rnn.get_state_size(conf)])

        cost = np.inf
        itr=0

        testing_generator = dataset.yield_examples(dataset="test")


        while True:
            try:
                train_costs = []
                for step, (x,y) in enumerate(dataset.yield_examples(steps=conf['training']['n_steps'])):
                    train_state, cost, _ = sesh.run([t['states_out'], t['cost'], t['train_op']],
                                                  feed_dict={t['x_in']:np.array([x]),
                                                            t['y_in']:np.array([y]),
                                                            t['states_in']:train_state})
                    train_costs.append(cost)
                    # if step % 10000 == 0:
                    #     print "\ttraining cost at epoch {} step {}:\t{}".format(itr, step, cost)

                print "Avg cost for training epoch {}: {}".format(itr, np.mean(train_costs))

                if itr % 20 == 0:
                    rnn.print_score(sesh, conf, test, dataset, mode="valid")
                elif itr % 5 == 0:
                    rnn.print_score(sesh, conf, test, dataset, mode="valid", n_to_print=0)



            except KeyboardInterrupt:
                import datetime
                q = raw_input("Type 'Y' or 'y' to see save model & see test score:\n")
                if q.lower() == 'y':
                    fn = "models/{}-{}.ckpt".format(config_name, datetime.datetime.now().strftime("%m-%d_%Hh%M"))
                    saver.save(sesh, fn)
                    print "model saved at: {}".format(fn)

                    rnn.print_score(sesh, conf, test, dataset, mode="test")
                sys.exit(0)
                
            itr+=1
