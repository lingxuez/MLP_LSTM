## check the test accuracy of mlp
import numpy as np
from utils import *

dataset = "../data/smaller"
max_len = 10
output_file = "../output/predict_smaller.npy"

## predicted results
test_predict = np.load(output_file)

## true results
dp = DataPreprocessor()
data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
       len(data.chardict), len(data.labeldict), shuffle=False)
(idxs,e,l) = mb_test.next()
test_true = np.argmax(l, axis=1)

## accuracy
compare = np.concatenate((test_predict[:, np.newaxis], test_true[:, np.newaxis]), axis=1)
accuracy = np.sum(test_predict== test_true) / float(len(test_true))
print "accuracy=", accuracy
