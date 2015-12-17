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

        valid_idx = int(len(self.data) * 0.7)
        test_idx = int(len(self.data) * 0.85)

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

    def idx_to_one_hot(self, idx):
        """Takes in int or list of ints corresponding to indices in character map and returns corresponding one-hot vector or list of vectors"""
        if type(idx) is int or np.int64 or np.int32:
            return np.eye(self.n_chars)[self.data[idx]]
        return [np.eye(self.n_chars)[i] for i in idx]

    def convert(self, idx):
        """Takes in int or list of ints corresponding to indices in character map and returns corresponding char or list of chars"""
        if type(idx) is int or np.int64 or np.int32:
            return self.idx_to_char_map[idx]
        return [self.idx_to_char_map[i] for i in idx]

    def data_to_ords(self, data):
        """takes a list of one-hot vectors and returns them as a list of indices"""
        if not hasattr(self, 'reverser'):
            self.reverser = np.array(range(self.n_chars))
        if (type(data[0]) is list) or (type(data[0] is np.array)):
            return np.array([np.sum(self.reverser * d) for d in data])
        return sum(self.reverser * data)

    def print_model_output(self, inputs, outputs):
        readable_x = u''.join(self.convert(
              self.data_to_ords(inputs)))

        def prettify(pair):
            prob_str = "{:0.3f}".format(np.round(pair[1],3))
            return (self.convert(pair[0]), unicode(prob_str))

        indexed_probs = [[prettify(l) for l in enumerate(ltrs)] for ltrs in outputs]

        indexed_probs = [sorted(l, key=lambda x: x[1], reverse=True) for l in indexed_probs]

        # import pdb; pdb.set_trace()

        top_5_idxs = [l[:5] for l in indexed_probs]

        for char, top5 in zip(readable_x, top_5_idxs):
            print "'",char,"'"
            top_out = [u"{}: {}".format(*l) for l in top5]
            print u"\t", u" | ".join(top_out).replace("\n", "\\n")


if __name__ == "__main__":
    d = DataSet(filename='todo.txt', decoding_fx=str)
    txt = d.plaintext_dataset()
    #print 'plaintext', txt
    #for (x,y) in d.yield_examples():
        #print 'x ',x,' and y ',y
