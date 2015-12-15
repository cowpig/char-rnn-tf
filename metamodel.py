import rnn
import data
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
    dataset = data.DataSet('./data/edgar.txt')
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
                "batch_size": batch_size,
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

        cost = np.inf

        itr=0
        try:
            while True:
                for (x,y) in dataset.yield_examples(steps=n_steps):
                    
                    state, out, cost, _ = sesh.run([states_out, y_out, costs, train],
                                    feed_dict={x_in:np.array([x]), 
                                                y_in:np.array([y]), 
                                                states_in:state})

                if itr % 1 == 0:
                    print "cost at epoch {}: {}".format(itr, cost)

                if itr % 5 == 0:
                    readable_x = u''.join(unichr(char) for char in dataset.convert(dataset.data_to_ords(x)))
                    #print readable_x
                    readable_out = []


                    example = out[0]
                    #print "example", example
                    indexed_probs = [sorted([idxed for idxed in enumerate(letter)],
                                                key=lambda x: x[1], reverse=True) for letter in example]
                    #print "idxd probs", indexed_probs
                    top_5_idxs = [i[:5] for i in indexed_probs]
                    
                    #print "probs:",indexed_probs
                    # print "top idxs",top_5_idxs

                    print u"input seq: {}".format(readable_x)
                    for char, top5 in zip(readable_x, top_5_idxs):
                        print "'",char,"'"
                        # print [l for l in top5]
                        top_out = [u"{0}: {1:.3f}".format(unichr(dataset.convert(l[0])), l[1]) for l in top5]
                        # print top_out
                        print u"\t", u" | ".join(top_out).replace("\n", "\\n")
                        # readable_output = ''.join(chr(char) for char in dataset.convert(letter[0]))
                        # print "predictions:\n{}".format(readable_output)
                itr+=1
        except KeyboardInterrupt:
            print dataset.plaintext_dataset()
            import sys;sys.exit(0)

    # we need a print function
    # stuff
