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


    def yield_examples(self, dataset="train", batch_size=None):
        if dataset not in ("train", "valid", "test"):
            raise Exception("Invalid dataset")
        
        dataset_len = self.idx[dataset][1] - self.idx[dataset][0]

        if batch_size is None:
            batch_size = dataset_len-1

        n_batches = (dataset_len-1) / batch_size

        for i in xrange(n_batches):
            idx = self.idx[dataset][0] + i*batch_size
            x = np.eye(self.n_chars)[self.data[idx:idx+batch_size]]
            y = np.eye(self.n_chars)[self.data[idx+1:idx+batch_size+1]]
            yield(x, y)

    def convert(self, ords):
        if type(ords) is int:
            return self.char_idx_map[ords]
        return [self.char_idx_map[o] for o in ords]
