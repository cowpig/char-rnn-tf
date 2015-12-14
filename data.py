import numpy as np

class DataSet(object):
    def __init__(self, filename, max_chars=-1):
        self.text = open(filename).read().decode('utf8')[:max_chars]
        ords = [ord(char) for char in self.text]
        self.all_chars = list(set(ords))
        self.char_idx_map = {ordinal : i for i, ordinal in enumerate(self.all_chars)}
        self.idx_to_char_map = {v: k for k,v in self.char_idx_map.iteritems()}
        self.data = np.array([self.char_idx_map[o] for o in ords])

        valid_idx = int(len(self.data) * 0.6)
        test_idx = int(len(self.data) * 0.8)

        self.idx = {
            "train" : (0, len(self.text)),
            # "train" : (0, valid_idx),
            "valid" : (valid_idx, test_idx),
            "test" : (test_idx, -1)
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
            yield(x, y)

    def convert(self, ords):
        if type(ords) is int:
            return self.idx_to_char_map[ords]
        return [self.idx_to_char_map[o] for o in ords]

    def data_to_ords(self, data):
        'takes a list of one-hot vectors and returns them as a list of indices'
        if not hasattr(self, 'reverser'):
            self.reverser = np.array(range(self.n_chars))
        if (type(data[0]) is list) or (type(data[0] is np.array)):
            return np.array([np.sum(self.reverser * data[i]) for i in xrange(len(data))])
        return sum(self.reverser * data)
