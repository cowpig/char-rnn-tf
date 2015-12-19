import tensorflow as tf
from layers import FullyConnected, LSTM

def get_config(input_size, config_name):
    if config_name == "double_lstm_flat":
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
                "learning_rate" : 0.1,
                "n_steps" : n_steps,
                "batch_size" : batch_size,
                "seed" : 1,
                "dropout" : 0.5
            }
        }
    if config_name == "single_lstm_flat":
        n_steps = 32
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
                    "type" : FullyConnected,
                    "input_size" : n_hidden,
                    "output_size": input_size,
                    "activation" : tf.nn.softmax,
                    "name" : "softmax"
                }
            ],

            "training" : {
                "learning_rate" : 0.1,
                "n_steps" : n_steps,
                "batch_size" : batch_size,
                "seed" : 1,
                "dropout" : 0.3
            }
        }

    if config_name == "test_lstm":
        n_steps = 16
        batch_size = 1
        n_hidden = 5
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
                    "type" : FullyConnected,
                    "input_size" : n_hidden,
                    "output_size": input_size,
                    "activation" : tf.nn.softmax,
                    "name" : "softmax"
                }
            ],

            "training" : {
                "learning_rate" : 0.1,
                "n_steps" : n_steps,
                "batch_size" : batch_size,
                "seed" : 1,
                "dropout" : 0.1
            }
        }
    if config_name == "test_lstm_no_embedding":
        n_steps = 16
        batch_size = 1
        n_hidden = 8
        config = {
            "layers" : [
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
                    "name": "LSTM_2"
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
                "learning_rate" : 0.1,
                "n_steps" : n_steps,
                "batch_size" : batch_size,
                "seed" : 1,
                "dropout" : 0.9
            }
        }

    return config

