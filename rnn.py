import tensorflow as tf
import numpy as np
from layers import LSTM, FullyConnected

def get_state_size(config):
    size = 0
    for layer in config['layers']:
        if layer['type'] is LSTM:
            size += layer['output_size'] * 2

    return size

def cross_entropy(observed, actual):
    return -tf.reduce_sum(actual*tf.log(observed))

def build_graph(config):
     ######################################
    # PREPARE VARIABLES & HYPERPARAMETERS
    total_state_size = get_state_size(config)

    print 'total_state_size', total_state_size

    batch_size = config['training']['batch_size']
    n_steps = config['training']['n_steps']
    learning_rate = config['training']['learning_rate']
    seed = config['training']['seed']
    x_size = config['layers'][0]['input_size']
    y_size = config['layers'][-1]['output_size']

    ######################################
    # PREPARE LAYERS
    n=0
    layers = []
    next_size = x_size
    for layer in config['layers']:
        assert(next_size == layer['input_size'])
        w = layer['w'] if 'w' in layer else None
        b = layer['b'] if 'b' in layer else None
        next_size = layer['output_size']
        scope = layer['name']
        with tf.variable_scope(scope):
            layers.append({
                'scope': scope,
                'object': layer['type'](layer['input_size'], next_size, w, b, seed),
            })
        if layer['type'] is LSTM:
            m = layer['output_size']*2
            layers[-1]['state_idx'] = (n, m)
            n = m
        elif layer['type'] is FullyConnected:
            layers[-1]['act'] = layer['activation']

    #######################################
    # TESTING

    with tf.variable_scope("test"):
        states_in = tf.placeholder(tf.float32, name="states_in", shape=(1, total_state_size))
        x_in = tf.placeholder(tf.float32, name="x", shape=(1, x_size))
        y_in = tf.placeholder(tf.float32, name="y", shape=(1, y_size))

        for layer in layers:
            if isinstance(layer['object'], LSTM):
                n, m = layer['state_idx']
                layer['state'] = tf.slice(states_in, [0, n], [-1, m])

        h = x_in
        for i, layer in enumerate(layers):
            # print 'h', h.get_shape()
            # print layer
            scope = layer['scope']
            if isinstance(layer['object'], LSTM):
                state = layer['state']
                c, h = layer['object'].build_layer(x_in=h, state=state, scope=scope)
                state = tf.concat(1, [c, h])
            else:
                h = layer['object'].build_layer(x_in=h, activation=layer['act'], 
                                                                        scope=scope)

        states_out = tf.concat(1, [layer['state'] for layer in layers \
                                if isinstance(layer['object'], LSTM)])
        cost = cross_entropy(h, y_in)

    testing = {
        "x_in" : x_in,
        "y_in" : y_in,
        "states_in" : states_in,
        "states_out" : states_out,
        "y_out" : h,
        "cost" : cost
    }


    #######################################
    # TRAINING

    with tf.variable_scope("train"):
        y_arr = []
        states_in = tf.placeholder(tf.float32, name="states_in",
                                    shape=(batch_size,total_state_size))
        x_in = tf.placeholder(tf.float32, name="x",
                                    shape=(batch_size,n_steps,x_size))
        y_in = tf.placeholder(tf.float32, name="y",
                                    shape=(batch_size,n_steps,y_size))

        for layer in layers:
            if isinstance(layer['object'], LSTM):
                n, m = layer['state_idx']
                layer['state'] = tf.slice(states_in, [0,n], [-1,m])

        for t in xrange(n_steps):
            if t>0:
                tf.get_variable_scope().reuse_variables()
            x_at_t = tf.slice(x_in, [0, t, 0], [-1, 1, -1])
            h = tf.squeeze(x_at_t, [1])
            # print 't', t
            for i, layer in enumerate(layers):
                # print 'h', h.get_shape()
                # print layer
                if i > len(layers):
                    d = 0.0
                else:
                    d = config['training']['dropout']

                scope = layer['scope']
                if isinstance(layer['object'], LSTM):
                    state = layer['state']
                    c, h = layer['object'].build_layer(x_in=h, state=state, 
                                                        scope=scope, dropout=d)
                    state = tf.concat(1, [c, h])
                else:

                    h = layer['object'].build_layer(x_in=h, scope=scope, dropout=d,
                                                            activation=layer['act'])
            y_arr.append(tf.expand_dims(h, 1))

        states_out = tf.concat(1, [layer['state'] for layer in layers \
                                if isinstance(layer['object'], LSTM)])
        y_out = tf.concat(1, y_arr)

        cost = cross_entropy(y_out, y_in)
        tvars = tf.trainable_variables()
        grads = tf.gradients(cost, tvars)
        optimus_prime = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimus_prime.apply_gradients(zip(grads, tvars))

    training = {
        'x_in': x_in,
        'y_in': y_in,
        'states_in': states_in,
        'states_out': states_out,
        'y_out' : y_out,
        'cost' : cost,
        'train_op': train_op
    }

    return {
        'train' : training,
        'test' : testing
    }


if __name__ == '__main__':
    def data_iterator():
        x = np.array([[ [1,0,0],[0,1,0],[0,0,1],[1,0,0]],
                      [ [1,0,0],[0,1,0],[0,0,1],[1,0,0]] ])

        y = np.array([[ [1,0,0],[0,1,0],[0,0,1],[1,0,0]],
                      [ [1,0,0],[0,1,0],[0,0,1],[1,0,0]] ])
        yield (x, y)

    input_size = 3
    n_steps = 4
    batch_size = 2
    config = {
        "layers" : [
            {
                "type" : FullyConnected,
                "input_size" : input_size,
                "output_size": 2,
                "activation" : tf.nn.sigmoid,
                "name" : 'embedding'
            },
            {
                "type" : LSTM,
                "input_size": 2,
                "output_size": 3,
                "batch_size": batch_size,
                "name" : "LSTM_1"
            },
            {
                "type" : LSTM,
                "input_size": 3,
                "output_size": input_size,
                "batch_size": batch_size,
                "name" : "LSTM_2"
            },
            {
                "type" : FullyConnected,
                "input_size" : input_size,
                "output_size": input_size,
                "activation" : tf.nn.softmax,
                "name" : "softmax"
            }
        ],
        "training" : {
            "learning_rate" : 0.1,
            "n_steps" : 4,
            "batch_size" : 2,
            "seed" : 1,
            "dropout" : 0.5
        }
    }
    graph = build_graph(config)

    with tf.Session() as sesh:
        # logger = tf.train.SummaryWriter('./log', sesh.graph_def)
        # pylogger = tf.python.training.summary_io.SummaryWriter('./pylog', sesh.graph_def)

        sesh.run(tf.initialize_all_variables())
        t = graph['train']
        test = graph['test']
        train_state = np.zeros([batch_size, get_state_size(config)])

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
        i=0
        while True:
            x, y  = data_iterator().next()
            # feed_dict={t['x_in']:x, 
            #             t['y_in']:y, 
            #             t['states_in']:train_state}
            # for k, v in feed_dict.iteritems():
            #     print k.name, v
            train_state, cost, _ = sesh.run([t['states_out'], t['cost'], t['train_op']], 
                                                feed_dict={t['x_in']:x, 
                                                            t['y_in']:y, 
                                                            t['states_in']:train_state})

            # print tf.Graph().get_operations()

            if i % 100 == 0:
              print "cost at epoch {}: {}".format(i, cost)

            if i % 1000 == 0:
                x = [1,0,0]
                test_state = np.zeros([batch_size, get_state_size(config)])
                print "testing... starting with {}".format(x)
                for _ in xrange(5):
                    test_state, x = sesh.run([test['states_out'], test['y_out']], 
                                            feed_dict={test['x_in']:x, 
                                                        test['states_in']:state})
                    print "...", x

            i+=1
