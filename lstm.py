"""
Long Short Term Memory for character level entity classification
"""
import sys, math, time
import argparse
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

class LSTM(object):
    """
    Long Short Term Memory + Feedforward layer
    Accepts maximum length of sequence, input size, number of hidden units and output size
    """
    def __init__(self, max_len, in_size, num_hid, out_size, nsample=1):
        ## Store the output of xman.setup() in this variable
        self.my_xman= self._build(max_len, in_size, num_hid, out_size, nsample)

    def _build(self, max_len, in_size, num_hid, out_size, nsample):
        """
        Define the model of MLP, and initialize all values.
        """
        xm = XMan()
        bound_w, bound_u= 6.0/math.sqrt(in_size+num_hid), 6.0/math.sqrt(2*num_hid)
        bound_b, bound_w2 = 0.1, 6.0 / math.sqrt(num_hid+out_size)

        ## data x: a list of registers, one for each character, one-hot
        xm.x = [f.input(name="x"+str(i), default=np.random.rand(nsample, in_size))
                    for i in xrange(max_len)]
        # xm.y = f.input(name="y", default=np.random.rand(nsample, out_size))
        # xm.x = []
        # for i in xrange(max_len):
        #     default_x = np.zeros((nsample, in_size))
        #     for n in xrange(nsample):
        #         default_x[n, np.random.choice(in_size)] = 1
        #     xm.x += [f.input(name="x"+str(i), default=default_x)]
        ## labels y (one-hot representation)
        default_y = np.zeros((nsample, out_size))
        for n in xrange(nsample):
            default_y[n, np.random.choice(out_size)] = 1
        xm.y = f.input(name="y", default=default_y)

        ## parameters with initial values
        xm.Wf = f.param(name="Wf", 
            default=np.random.uniform(low=-bound_w, high=bound_w,
                        size=(in_size, num_hid)))
        xm.Wi = f.param(name="Wi", 
            default=np.random.uniform(low=-bound_w, high=bound_w,
                        size=(in_size, num_hid)))
        xm.Wc = f.param(name="Wc", 
            default=np.random.uniform(low=-bound_w, high=bound_w,
                        size=(in_size, num_hid)))
        xm.Wo = f.param(name="Wo", 
            default=np.random.uniform(low=-bound_w, high=bound_w,
                        size=(in_size, num_hid)))
        xm.Uf = f.param(name="Uf", 
            default=np.random.uniform(low=-bound_u, high=bound_u,
                        size=(num_hid, num_hid)))
        xm.Ui = f.param(name="Ui", 
            default=np.random.uniform(low=-bound_u, high=bound_u,
                        size=(num_hid, num_hid)))
        xm.Uc = f.param(name="Uc", 
            default=np.random.uniform(low=-bound_u, high=bound_u,
                        size=(num_hid, num_hid)))
        xm.Uo = f.param(name="Uo", 
            default=np.random.uniform(low=-bound_u, high=bound_u,
                        size=(num_hid, num_hid)))
        xm.bf = f.param(name="bf", default=bound_b*np.ones((1, num_hid)))
        xm.bi = f.param(name="bi", default=bound_b*np.ones((1, num_hid)))
        xm.bc = f.param(name="bc", default=bound_b*np.ones((1, num_hid)))
        xm.bo = f.param(name="bo", default=bound_b*np.ones((1, num_hid)))

        xm.W2 = f.param(name="W2", 
            default=np.random.uniform(low=-bound_w2, high=bound_w2, 
                        size=(num_hid, out_size)))
        xm.b2 = f.param(name="b2", default=bound_b*np.ones((1, out_size)))

        ## hidden states and gates; we have one hidden state per sample
        xm.f, xm.i, xm.cnew, xm.o = [], [], [], [] 
        ## initialize the first hidden state and cell state
        xm.h = [f.input(name="h0", default=0.1*np.ones((nsample, num_hid)))]
        xm.c = [f.input(name="c0", default=0.1*np.ones((nsample, num_hid)))]
        ## LSTM
        for t in xrange(max_len):
            xm.f += [f.sigmoid(f.tri_add(xm.bf, 
                        f.mul(xm.x[t], xm.Wf), f.mul(xm.h[t], xm.Uf)))]
            xm.i += [f.sigmoid(f.tri_add(xm.bi,
                        f.mul(xm.x[t], xm.Wi), f.mul(xm.h[t], xm.Ui)))]
            xm.cnew += [f.tanh(f.tri_add(xm.bc,
                        f.mul(xm.x[t], xm.Wc), f.mul(xm.h[t], xm.Uc)))]
            xm.o += [f.sigmoid(f.tri_add(xm.bo,
                        f.mul(xm.x[t], xm.Wo), f.mul(xm.h[t], xm.Uo)))]
            xm.c += [f.ele_mul(xm.f[t], xm.c[t]) + f.ele_mul(xm.i[t], xm.cnew[t])]
            xm.h += [f.ele_mul(xm.o[t], f.tanh(xm.c[t+1]))]

        ## output
        xm.o2 = f.relu(f.mul(xm.h[max_len], xm.W2) + xm.b2)
        xm.output = f.softMax(xm.o2)
        xm.loss = f.crossEnt(xm.output, xm.y)
        # xm.output = f.predict(xm.p)
        return xm.setup()
 
    def checkGradient(self, eps=1e-4, tol=1e-4):
        """
        Check gradient implementation with numeric gradient.
        THIS HAS BUG!!!
        """
        wengart_list = self.my_xman.operationSequence(self.my_xman.loss)
        value_dict = self.my_xman.inputDict()
        ad = Autograd(self.my_xman)

        ## gradient calculated by model
        value_dict = ad.eval(wengart_list, value_dict)
        loss = value_dict["loss"]
        gradients = ad.bprop(wengart_list, value_dict, loss=1.0)

        ## compare to numeric results
        print "checking gradients..."
        opt_wengart = ad.optimizeForBProp(wengart_list)
        #for rname in gradients:
        for (rname,funName,inputNames) in opt_wengart:
            if (rname=="loss"): #or (not self.my_xman.isParam(rname))
                print "skipped " + rname
                continue

            print "checking gradient for " + rname
            grad = gradients[rname]
            value = np.copy(value_dict[rname])

            ## change one coordinate at a time, evaluate new loss
            for row in xrange(value.shape[0]):
                for col in xrange(value.shape[1]):
                    ## new value
                    new_value = np.copy(value)
                    new_value[row][col] += eps
                    ## new loss
                    value_dict[rname] = new_value
                    ## BUGGY!!!
                    new_loss = ad.eval(wengart_list, value_dict)["loss"]
                    ## numerical gradient
                    num_grad = (new_loss - loss) / eps
                    if abs(num_grad - grad[row][col]) > tol:
                        raise ValueError("Gradient check fails for "+rname+
                            " at row="+str(row)+", col="+str(col)+
                            ", num="+str(num_grad)+", cal="+str(grad[row][col]))
                    ## set back to original value
                    value_dict[rname] = value

def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']

    # load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    # build
    print "building lstm..."
    lstm = LSTM(max_len, mb_train.num_chars, num_hid, mb_train.num_labels)
    ## check gradient
    # lstm.checkGradient()

    # train
    print "training..."
    wengart_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    value_dict = lstm.my_xman.inputDict()
    ad = Autograd(lstm.my_xman)
    # track loss for debugging
    (val_losses, training_time) = ([], [])
    lr, min_val_loss, best_value_dict = init_lr, None, None 
    for i in range(epochs):
        mb_train.reset()
        mb_valid.reset()
        (start_time, num_train) = (time.time(), 0)
        ## mini-batch stochastic gradient descent
        for (idxs,e,l) in mb_train:
            ## idxs: ids of examples in minibatch, 
            ##      where len(idxs) == batch_size
            ## e: entities in one-hot format,
            ##    e.shape == (batch_size, max_len, num_chars)
            ##    e[i] for the i-th example, 
            ## l: corresponding output labels also in one-hot format
            ##    l[i] for the i-th example
            ##    l.shape == (batch_size, num_labels)

            ## prepare the input, resize e to be (max_len, batch_size, num_chars)
            cur_batch_size = e.shape[0]
            num_train += cur_batch_size
            value_dict["x"] = np.swapaxes(e, 0, 1)
            value_dict["y"] = l
            value_dict["h0"] = 0.1*np.ones((cur_batch_size, num_hid))
            value_dict["c0"] = 0.1*np.ones((cur_batch_size, num_hid))

            ## feed-forward and back-propogation
            value_dict = ad.eval(wengart_list, value_dict)
            gradients = ad.bprop(wengart_list, value_dict, loss=1.0/cur_batch_size)

            # update parameters
            for rname in gradients:
                if lstm.my_xman.isParam(rname):
                    value_dict[rname] -= lr * gradients[rname] 

        training_time += [time.time() - start_time]

        ## calculate the validating loss
        val_num = len(data.validation)
        (idxs,e,l) = mb_valid.next()
        ## data
        value_dict["x"] = np.swapaxes(e, 0, 1)
        value_dict["y"] = l
        value_dict["h0"] = 0.1*np.ones((val_num, num_hid))
        value_dict["c0"] = 0.1*np.ones((val_num, num_hid))
        ## loss
        val_loss = ad.eval(wengart_list, value_dict)["loss"] / val_num
        val_losses += [val_loss]
        # print ("calculated loss on %d validating samples" % val_num)

        ## save the best model
        if (min_val_loss is None) or val_loss < min_val_loss:
            best_value_dict = value_dict
            min_val_loss = val_loss
    print "done"

    ## predict on the test set using the best parameters
    # output_wengart_list = lstm.my_xman.operationSequence(lstm.my_xman.output)
    test_num = len(data.test)
    ## data
    (idxs,e,l) = mb_test.next()
    best_value_dict["x"] = np.swapaxes(e, 0, 1)
    best_value_dict["y"] = l
    best_value_dict["h0"] = 0.1*np.ones((test_num, num_hid))
    best_value_dict["c0"] = 0.1*np.ones((test_num, num_hid))
    ## loss
    best_value_dict = ad.eval(wengart_list, best_value_dict)
    test_loss = best_value_dict["loss"] / test_num
    test_output = best_value_dict["output"]
    # print ("predicted labels on %d testing samples" % test_num)

    ## save predicted probabilities on test set
    np.save(output_file, test_output)
    # return (training_time, val_losses, test_loss)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', dest='max_len', type=int, default=10)
    parser.add_argument('--num_hid', dest='num_hid', type=int, default=50)
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=64)
    parser.add_argument('--dataset', dest='dataset', type=str, default='tiny')
    parser.add_argument('--epochs', dest='epochs', type=int, default=15)
    parser.add_argument('--init_lr', dest='init_lr', type=float, default=0.5)
    parser.add_argument('--output_file', dest='output_file', type=str, default='output')
    params = vars(parser.parse_args())
    main(params)

    ## for debugging
    # params = {"max_len":3, "num_hid":10, "batch_size":16,
    #             "dataset":"../data/tiny", "epochs":15, "init_lr":0.5,
    #             "output_file":"../lstm_out/predict_tiny.npy"}

    # (training_time, val_losses, test_loss) = main(params)
    # ## save validating loss and training time
    # basename = params["output_file"].split(".npy")[0]
    # np.savetxt(basename+"_val_loss.txt", val_losses)
    # np.savetxt(basename+"_ trainig_time.txt", training_time)

    #####################
    # for testing and debugging
    ######################
    # max_len, num_chars, num_hid, num_labels, nsample = 3, 10, 8, 5, 20
    # lstm = LSTM(max_len, num_chars, num_hid, num_labels, nsample) 

    # # ## try ffwd and bprop, check dimensions
    # wengart_list = lstm.my_xman.operationSequence(lstm.my_xman.loss)
    # value_dict = lstm.my_xman.inputDict()
    # ad = Autograd(lstm.my_xman) 
    # value_dict = ad.eval(wengart_list, value_dict)

    # value_dict["x"] = np.random.uniform(max_len, nsample, num_chars)
    # value_dict = ad.eval(wengart_list, value_dict)
    # print value_dict["p"].shape

    # output_wengart_list = lstm.my_xman.operationSequence(lstm.my_xman.output)
    # predict = list(ad.eval(output_wengart_list, value_dict)["output"])
#     lstm.checkGradient()
#     gradients = ad.bprop(wengart_list, value_dict, loss=1.0)

#     # for rname in gradients:
#     #     if lstm.my_xman.isParam(rname):
#     #         print rname, value_dict[rname].shape == gradients[rname].shape 

