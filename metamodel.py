import rnn
import data
import tensorflow as tf
import numpy as np
from layers import LSTM, FullyConnected




if __name__ == "__main__":
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
        # logger = tf.train.SummaryWriter('./log', sesh.graph_def)
        # pylogger = tf.python.training.summary_io.SummaryWriter('./pylog', sesh.graph_def)

        sesh.run(tf.initialize_all_variables())
        t = graph['train']
        test = graph['test']
        train_state = np.zeros([batch_size, rnn.get_state_size(config)])

        # tf.train.write_graph(sesh.graph_def, './graph', 'rnn_graph.pbtxt')

        # training = {
        #     'x_in': x_in,
        #     'y_in': y_in,
        #     'states_in': states_in,
        #     'states_out': states_out,
        #     'cost' : cost,
        #     'train_op': train_op
        # }

        cost = np.inf
        itr=0

        testing_generator = dataset.yield_examples(dataset="test")

        try:
          for (x,y) in dataset.yield_examples(steps=n_steps):
              train_state, cost, _ = sesh.run([t['states_out'], t['cost'], t['train_op']],
                                              feed_dict={t['x_in']:np.array([x]),
                                                        t['y_in']:np.array([y]),
                                                        t['states_in']:train_state})


              if itr % 100 == 0:
                  print "cost at epoch {}: {}".format(itr, cost)

              if itr % 1000 == 0:
                  test_cost = 0
                  test_state = np.zeros([1, rnn.get_state_size(config)])
                  print "testing..."
                  inputs = []
                  outputs = []
                  for _ in xrange(30):
                      x, y = testing_generator.next()
                      inputs.append(x[:])
                      test_state, x, cost = sesh.run([test['states_out'], test['y_out'],
                                                      test['cost']],
                                                feed_dict={test['x_in']:x,
                                                          test['states_in']:test_state,
                                                          test['y_in']:y})
                      test_cost += cost
                      outputs.append(x[0][:])

                  print "Test cost: ", test_cost

                  readable_x = u''.join(dataset.convert(dataset.data_to_ords(inputs)))

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

                  itr+=1

        except KeyboardInterrupt:
            print dataset.plaintext_dataset()
            import sys;sys.exit(0)
