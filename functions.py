# some useful functions
import numpy as np
from scipy.special import expit
from xman import *


# some useful functions
# declare all operations here first

class f(XManFunctions):
    @staticmethod
    def square(a):
        return XManFunctions.registerDefinedByOperator('square',a)
    
    @staticmethod
    def relu(a):
        return XManFunctions.registerDefinedByOperator('relu', a)

    @staticmethod
    def crossEnt(p, y):
        return XManFunctions.registerDefinedByOperator('crossEnt', p, y)

    @staticmethod
    def softMax(a):
        return XManFunctions.registerDefinedByOperator('softMax', a)

    @staticmethod
    def predict(a):
        return XManFunctions.registerDefinedByOperator('predict', a)

    @staticmethod
    def sigmoid(a):
        return XManFunctions.registerDefinedByOperator('sigmoid', a)

    @staticmethod
    def tanh(a):
        return XManFunctions.registerDefinedByOperator('tanh', a)

    @staticmethod
    def ele_mul(x1, x2):
        return XManFunctions.registerDefinedByOperator('ele_mul', x1, x2)

    @staticmethod
    def tri_add(x1, x2, x3):
        return XManFunctions.registerDefinedByOperator('tri_add', x1, x2, x3)

    ## when add a new operator here, also need to add to EVAL_FUNS and BP_FUNS


# the functions that autograd.eval will use to evaluate each function,
# to be called with the functions actual inputs as arguments
P_MIN = 1e-10
P_MAX = 1

EVAL_FUNS = {
    ## note that x1, x2 can be arrays with same shape
    'add':      lambda x1,x2: x1+x2,
    'tri_add': lambda x1,x2,x3: x1+x2+x3, 
    'subtract': lambda x1,x2: x1-x2,
    'mul': lambda x1,x2: x1.dot(x2),
    ## element-wise squre 
    'square':  lambda x: np.square(x),
    ## p, y are arrays with same shape; rows are samples
    ## y is one-hot representation
    'crossEnt': lambda p,y: -np.sum(y * np.log(p)) / y.shape[0],
                    # -np.sum(np.multiply(y, np.log(np.clip(p, P_MIN, P_MAX)))),
    ## x: rows are samples; soft-max is applied row-wise
    'softMax': lambda x: np.exp(x-np.max(x)) / np.sum(np.exp(x-np.max(x)), axis=1)[:, np.newaxis], 
                    # np.clip(np.exp(x-np.max(x)) / \
                    # np.sum(np.exp(x-np.max(x)), axis=1)[:, np.newaxis],
                    # P_MIN, P_MAX),
    ## relu is applied element wise 
    'relu': lambda x: np.maximum(x, 0), #np.multiply(x, np.greater(x, 0)),
    ## for each row of p (one sample), find the index of maxima
    'predict': lambda p: np.argmax(p, axis=1),
    ## element-wise sigmoid
    'sigmoid': lambda x: expit(x),
    ## element-wise tanh
    'tanh': lambda x: np.tanh(x),
    ## element-wise multiply
    'ele_mul': lambda x1,x2: x1 * x2
    }

# the functions that autograd.bprop will use in reverse mode
# differentiation.  BP_FUNS[f] is a list of functions df1,....,dfk
# where dfi is used in propagating errors to the i-th input xi of f.
# Specifically, dfi is called with the ordinary inputs to f, with two
# additions: the incoming error, and the output of the function, which
# was computed by autograd.eval in the eval stage.  dfi will return
# delta * df/dxi [f(x1,...,xk)]
# 
# NOTE: Autograd has an optimization where if it finds a softMax op
# followed by crossEnt op, it combines the backward pass for both. So
# you only need to implement the BP_FUNS for the combined operation 
# crossEnt-softMax below.

def _derivAdd(delta,x):
    # This is for XW + b, where the output is N-by-d after broadcasting
    # but when computing the gradient of b, we only want 1-by-d
    if delta.shape!=x.shape:
        # broadcast, sum along axis=0
        if delta.shape[1]!=x.shape[1]:
            raise ValueError("Dimension Mismatch: delta.shape[1]="+
                    str(delta.shape[1]) + ", x.shape[1]=" + str(x.shape[1]))
        return delta.sum(axis=0)[np.newaxis, :] #we sum the gradients over the batch
    else: return delta

BP_FUNS = {
    'add': [lambda delta,out,x1,x2: _derivAdd(delta,x1), lambda delta,out,x1,x2: _derivAdd(delta,x2)],
    'tri_add': [lambda delta,out,x1,x2,x3: _derivAdd(delta,x1),
                lambda delta,out,x1,x2,x3: _derivAdd(delta,x2),
                lambda delta,out,x1,x2,x3: _derivAdd(delta,x3)],
    'subtract': [lambda delta,out,x1,x2: _derivAdd(delta,x1), lambda delta,out,x1,x2: -_derivAdd(delta,x2)],
    'square': [lambda delta,out,x : delta * 2.0 * x],
    'mul': [lambda delta,out,x1,x2: np.dot(delta, x2.transpose()), 
                        lambda delta,out,x1,x2: np.dot(x1.transpose(), delta)],
    ## crossEnt(softMax(o), y)
    'crossEnt-softMax':  [lambda delta,out,o,y: -(y-EVAL_FUNS["softMax"](o))/y.shape[0], 
                        lambda delta,out,o,y: y ## this won't be used!
                        ],
    ## element-wise operations
    'relu': [lambda delta,out,x: delta * (x > 0) 
            ],
    'sigmoid': [lambda delta,out,x: delta * out * (1-out)],
    'tanh': [lambda delta,out,x: delta * (1 - np.square(out))],
    'ele_mul': [lambda delta,out,x1,x2: delta * x2, 
                lambda delta,out,x1,x2: delta * x1]
    }
