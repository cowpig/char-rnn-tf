import rnn
import data
import os.path
import config
import sys

if __name__ == "__main__":
    fn = sys.argv[1]
    if not os.path.isfile(fn):
        raise Exception("File '{}' does not exist.".format(fn))


    dataset = data.DataSet('./data/edgar.txt',decoding_fx=data.decode_encoding)
    conf = config.get_config(dataset.n_chars)
    # dataset.idx['train'] = (0,320)

    rnn.riff(fn, conf, dataset)
