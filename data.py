import numpy as np

def decode_encoding(str_in, encoding='utf8'):
    return str_in.decode(encoding)


class DataSet(object):
    def __init__(self, filename, max_chars=-1, decoding_fx=decode_encoding):
        self.text = decoding_fx( open(filename).read() )[:max_chars]
        chars = [char for char in self.text]
        self.all_chars = list(set(chars))
        #print "chars", self.all_chars
        self.char_idx_map = {char : i for i, char in enumerate(self.all_chars)}
        #print "mapped", self.char_idx_map
        self.idx_to_char_map = {v: k for k,v in self.char_idx_map.iteritems()}
        self.data = np.array([self.char_idx_map[c] for c in chars])

        valid_idx = int(len(self.data) * 0.6)
        test_idx = int(len(self.data) * 0.8)
        #print 'valid', valid_idx
        #print 'test', test_idx
        #print 'end',len(self.data)

        self.idx = {
            "train" : (0, len(self.text)),
            # "train" : (0, valid_idx),
            "valid" : (valid_idx, test_idx),
            "test" : (test_idx, len(self.data))
        }

        self.n_chars = len(self.all_chars)

    def plaintext_dataset(self, dataset="train"):
        return self.text[self.idx[dataset][0]:self.idx[dataset][1]]

    def yield_examples(self, dataset="train", steps=None):
        if dataset not in ("train", "valid", "test"):
            raise Exception("Invalid dataset")

        dataset_len = self.idx[dataset][1] - self.idx[dataset][0]

        if steps is None:
            steps = 1

        n_steps = (dataset_len-1) / steps

        for i in xrange(n_steps):
            idx = self.idx[dataset][0] + i*steps
            x = np.eye(self.n_chars)[self.data[idx:idx+steps]]
            y = np.eye(self.n_chars)[self.data[idx+1:idx+steps+1]]
            yield (x, y)

    def convert(self, idx):
        if type(idx) is int:
            return self.idx_to_char_map[idx]
        return [self.idx_to_char_map[i] for i in idx]

    def data_to_ords(self, data):
        'takes a list of one-hot vectors and returns them as a list of indices'
        if not hasattr(self, 'reverser'):
            self.reverser = np.array(range(self.n_chars))
        if (type(data[0]) is list) or (type(data[0] is np.array)):
            return np.array([np.sum(self.reverser * d) for d in data])
        return sum(self.reverser * data)

if __name__ == "__main__":
    d = DataSet(filename='todo.txt', decoding_fx=str)
    txt = d.plaintext_dataset()
    #print 'plaintext', txt
    #for (x,y) in d.yield_examples():
        #print 'x ',x,' and y ',y
