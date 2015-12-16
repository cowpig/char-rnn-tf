import rnn
import data
import tensorflow as tf
import numpy as np
from layers import LSTM, FullyConnected
import sys
import os.path

def score(sesh, config, graph, dataset, mode="valid", n_examples=50):
    costs = []
    outs = []
    state = np.zeros([1, rnn.get_state_size(config)])

    for i, (x, y) in enumerate(dataset.yield_examples(dataset=mode)):
        state, out, cost = sesh.run([graph['states_out'], graph['y_out'], graph['cost']],
                                    feed_dict={ graph['x_in']:x, 
                                                graph['states_in']:state,
                                                graph['y_in']:y })
        costs.append(cost)



def generate_text(sesh, config, graph, dataset, mode="valid", n_examples=50):
    test_cost = 0
    test_state = np.zeros([1, rnn.get_state_size(config)])
    print "testing..."
    inputs = []
    outputs = []
    for _ in xrange(30):
        x, y = testing_generator.next()
        inputs.append(x[:])
        test_state, x, cost = sesh.run([test['states_out'],
                                          test['y_out'],
                                          test['cost']],
                                        feed_dict={
                                            test['x_in']:x,
                                            test['states_in']:test_state,
                                            test['y_in']:y})
        test_cost += cost
        outputs.append(x[0][:])

        print "Test cost: ", test_cost

        readable_x = u''.join(dataset.convert(
              dataset.data_to_ords(inputs)))
        print "input: ", readable_x

        indexed_probs = [sorted([idxed for idxed in enumerate(letter)],
                                  key=lambda x: x[1], reverse=True)
                          for letter in outputs]
        top_5_idxs = [i[:5] for i in indexed_probs]

        for char, top5 in zip(readable_x, top_5_idxs):
            print "'",char,"'"
            top_out = [u"{0}: {1:.3f}".format(dataset.convert(l[0]), l[1])
                        for l in top5]
            print u"\t", u" | ".join(top_out).replace("\n", "\\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        fn = sys.argv[1]
    else:
        fn = None

    if fn:
        if not os.path.isfile(fn):
            raise Exception("File '{}' does not exist.".format(fn))


    dataset = data.DataSet('./data/edgar.txt',decoding_fx=data.decode_encoding)
    # dataset.idx['train'] = (0,320)

    input_size = dataset.n_chars
    n_steps = 16
    batch_size = 1
    n_hidden = 100
    config = {
        "layers" : [
            {
                "type" : FullyConnected,
                "input_size" : input_size,
                "output_size": n_hidden,
                "activation" : tf.nn.sigmoid,
                "name" : 'embedding'
            },
            {
                "type" : LSTM,
                "input_size": n_hidden,
                "output_size": n_hidden,
                "name": "LSTM_1"
            },
            {
                "type" : LSTM,
                "input_size": n_hidden,
                "output_size": n_hidden,
                "name" : "LSTM_2"
            },
            {
                "type" : FullyConnected,
                "input_size" : n_hidden,
                "output_size": input_size,
                "activation" : tf.nn.softmax,
                "name" : "softmax"
            }
        ],

        "training" : {
            "learning_rate" : 0.01,
            "n_steps" : n_steps,
            "batch_size" : batch_size,
            "seed" : 1,
            "dropout" : 0.3
        }
    }

    graph = rnn.build_graph(config)

    with tf.Session() as sesh:
        sesh.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        if fn:
            saver.restore(sesh, fn)
        
        t = graph['train']
        test = graph['test']
        train_state = np.zeros([batch_size, rnn.get_state_size(config)])

        cost = np.inf
        itr=0

        testing_generator = dataset.yield_examples(dataset="test")


        while True:
            try:
                for t, (x,y) in enumerate(dataset.yield_examples(steps=n_steps)):
                    train_costs = 
                    train_state, cost, _ = sesh.run([t['states_out'], t['cost'], t['train_op']],
                                                  feed_dict={t['x_in']:np.array([x]),
                                                            t['y_in']:np.array([y]),
                                                            t['states_in']:train_state})
                    if t % 100 == 0:
                        print "\ttraining cost at epoch {} step {}:\t{}".format(itr, t, cost)

                if itr % 10 == 0:


                if itr % 50 == 0:


                      itr+=1

          except KeyboardInterrupt:
            import datetime
            fn = "models/metamodel_{}.ckpt".format(datetime.now().strftime("%m-%d_%H:%M"))
            saver.save(sesh, fn)
            print "model saved at: {}".format(fn)
