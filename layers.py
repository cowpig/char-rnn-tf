import tensorflow as tf
import numpy as np

class LSTM(object):
    def __init__(self, input_size, output_size, w=None, b=None, seed=1):
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        # print "input_size: ", input_size
        # print "output_size: ", output_size
        # print "weight_dims: ", weight_dims

        if w:
            self.w = w
        else:
            weight_dims = [input_size + output_size, output_size*4]
            random_weights = tf.random_normal(weight_dims, seed=self.seed)
            self.w = tf.Variable(random_weights, trainable=True, name='w')

        if b:
            self.b = b
        else:
            bias_dims = [output_size*4]
            bias_start = tf.zeros(bias_dims)
            self.b = tf.Variable(bias_start, trainable=True, name='b')


    def get_params(self):
        w_f, w_i, w_c, w_o = tf.split(1, 4, self.w)
        b_f, b_i, b_c, b_o = tf.split(1, 4, self.b)

    def build_layer(self, x_in, state, scope="lstm_cell", dropout=0.0):
        # print (x_in, c_in, h_in, scope)
        # print [type(thing) for thing in (x_in, c_in, h_in, scope)]
        # print [(item.name, item.dtype) for thing in (h_in, c_in) for item in thing]
        # print (x_in.name, x_in.dtype)

        with tf.variable_scope(scope):
            # print "scope: ", scope
            # print "state: ", state.get_shape()
            c_in, h_in = tf.split(2, 2, state)
            # print "x: ", x.shape
            # print "h_in: ", h_in.get_shape()
            # print "c_in: ", c_in.get_shape()
            x_with_h = tf.concat(2, [x_in, h_in])

            ones_for_bias = tf.constant(np.ones([self.batch_size,1,1]), name="b", 
                                                                    dtype=tf.float32)
            x_h_concat = tf.concat(2, [ones_for_bias, x_with_h])

            # forget gate layer
            # print "w_f: ", self.w_f.get_shape()
            # print "x_h_concat: ", x_h_concat.get_shape()
            f = tf.sigmoid(tf.batch_matmul(x_h_concat, self.w_f))

            # candidate values
            i = tf.sigmoid(tf.batch_matmul(x_h_concat, self.w_i))
            candidate_c = tf.tanh(tf.batch_matmul(x_h_concat, self.w_c))

            # new cell state (hidden)
            # forget old values of c
            old_c_to_keep = tf.mul(f, c_in)
            # scaled candidate values of c
            new_c_to_keep = tf.mul(i, candidate_c)
            c = tf.add(old_c_to_keep, new_c_to_keep)

            # new scaled output
            o = tf.sigmoid(tf.batch_matmul(x_h_concat, self.w_o))
            h = tf.mul(o, tf.tanh(c))

            if self.dropout:
                h = tf.nn.dropout(h, self.dropout, seed=self.seed)

            return (c, h)

class FullyConnected(object):
    def __init__(self, input_size, output_size, w=None, b=None, seed=1):
        w = tf.Variable(tf.random_normal([shape[1].value]*2, seed=seed), 
                            name="w", trainable=True)
        b = tf.Variable(tf.zeros([output_size]), name="b", trainable=True)

    def build_layer(x_in, scope="fully_connected_layer"):
        with tf.variable_scope(scope):
            return activation(tf.nn.xw_plus_b(var_in, w, b))
