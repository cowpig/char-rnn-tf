import numpy as np


class DataSet(object):
    def __init__(self, filename, max_chars=-1):
        text = [ord(char) for char in open(filename).read()[:max_chars]]
        all_chars = list(set(text))
        self.char_idx_map = {ordinal : i for i, ordinal in enumerate(all_chars)}
        self.data = np.array([self.char_idx_map[o] for o in text])

        valid_idx = int(len(self.data) * 0.6)
        test_idx = int(len(self.data) * 0.8)

        self.idx = {
            "train" : (0, valid_idx),
            "valid" : (valid_idx, test_idx),
            "test" : (test_idx, -1)
        }

        self.n_chars = len(all_chars)


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
            return self.char_idx_map[ords]
        return [self.char_idx_map[o] for o in ords]
