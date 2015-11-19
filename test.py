import data
import numpy as np

d = data.DataSet("data/data.txt", 100)
x, y = d.yield_examples("train").next()

assert(np.array_equal(x[1:11], y[:10]))

print "tests pass w00t"
