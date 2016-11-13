"""
Multilayer Perceptron for character level entity classification
"""
import argparse, math, time, copy
import numpy as np
from xman import *
from utils import *
from autograd import *

np.random.seed(0)

class MLP(object):
    """
    Multilayer Perceptron
    Accepts list of layer sizes [in_size, hid_size1, hid_size2, ..., out_size]
    """
    def __init__(self, layer_sizes):
        # Store the output of xman.setup() in this variable
        self.my_xman = self._build(layer_sizes) 

    def _build(self, layer_sizes):
        """
        Define the model of MLP, and initialize all values.
        """
        xm = XMan()
        (dim_input, dim_hidden, dim_class) = layer_sizes
        init_bias = 0.1

        ## data: features x and labels y (one-hot representation)
        xm.x = f.input(name="x", default=np.random.rand(1, dim_input))
        default_y = np.zeros((1, dim_class))
        default_y[0, np.random.choice(dim_class)] = 1
        xm.y = f.input(name="y", default=default_y)

        ## initialize parameters using uniform (-bound, bound)
        ## first layer
        init_bound = math.sqrt(6.0 / (dim_input + dim_hidden))
        xm.W1 = f.param(name="W1", 
            default=np.random.uniform(low=-init_bound, high=init_bound,
                        size=(dim_input, dim_hidden)))
        xm.b1 = f.param(name="b1", 
            default=np.random.uniform(low=-init_bias, high=init_bias,
            size=(1, dim_hidden)))
        xm.o1 = f.relu(f.mul(xm.x, xm.W1) + xm.b1)

        ## second layer
        init_bound = math.sqrt(6.0 / (dim_hidden + dim_class))
        xm.W2 = f.param(name="W2", 
            default=np.random.uniform(low=-init_bound, high=init_bound, 
                        size=(dim_hidden, dim_class)))
        xm.b2 = f.param(name="b2", 
            default=np.random.uniform(low=-init_bias, high=init_bias,
                size=(1, dim_class)))
        xm.o2 = f.relu(f.mul(xm.o1, xm.W2) + xm.b2)

        ## cross entropy loss
        xm.output = f.softMax(xm.o2)
        xm.loss = f.crossEnt(xm.output, xm.y)
        return xm.setup()

    def checkGradient(self, eps=1e-4, tol=1e-4):
        """
        Check gradient implementation with numeric gradient.
        """
        wengart_list = self.my_xman.operationSequence(self.my_xman.loss)
        value_dict = self.my_xman.inputDict()
        ad = Autograd(self.my_xman)

        ## gradient calculated by model
        value_dict = ad.eval(wengart_list, value_dict)
        old_loss = value_dict["loss"]
        gradients = ad.bprop(wengart_list, value_dict, loss=1.0)

        ## compare to numeric results
        print "checking gradients..."
        for rname in gradients:
            if (not self.my_xman.isParam(rname)) or (rname=="loss"):
                print "skipped " + rname
                continue

            print "checking gradient for " + rname
            grad = gradients[rname]
            old_value = np.copy(value_dict[rname])

            ## change one coordinate at a time, evaluate new loss
            for row in xrange(value.shape[0]):
                for col in xrange(value.shape[1]):
                    ## new value
                    new_value = np.copy(old_value)
                    new_value[row][col] += eps
                    ## new loss
                    value_dict[rname] = new_value
                    new_loss = ad.eval(wengart_list, value_dict)["loss"]
                    ## numerical gradient
                    num_grad = (new_loss - old_loss) / eps

                    if abs(num_grad - grad[row][col]) > tol:
                        raise ValueError("Gradient check fails for "+rname)
                    ## set back to original value
                    value_dict[rname] = old_value


def main(params):
    epochs = params['epochs']
    max_len = params['max_len']
    num_hid = params['num_hid']
    batch_size = params['batch_size']
    dataset = params['dataset']
    init_lr = params['init_lr']
    output_file = params['output_file']

    ## load data and preprocess
    dp = DataPreprocessor()
    data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    ## minibatches
    mb_train = MinibatchLoader(data.training, batch_size, max_len, 
           len(data.chardict), len(data.labeldict))
    mb_valid = MinibatchLoader(data.validation, len(data.validation), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)
    mb_test = MinibatchLoader(data.test, len(data.test), max_len, 
           len(data.chardict), len(data.labeldict), shuffle=False)

    num_chars = mb_train.num_chars

    ## build
    print "building mlp..."
    mlp = MLP([max_len*mb_train.num_chars, num_hid, mb_train.num_labels])
    ## check gradient
    # mlp.checkGradient()

    ## train
    print "training..."
    min_val_loss, best_value_dict = None, None  
    wengart_list = mlp.my_xman.operationSequence(mlp.my_xman.loss)
    value_dict = mlp.my_xman.inputDict()
    ad = Autograd(mlp.my_xman)
    # track loss for debugging
    # (val_losses, training_time) = ([], [])

    lr = init_lr
    for i in range(epochs):
        # print "epoch_", i+1, 
        mb_train.reset()
        mb_valid.reset()
        # lr = init_lr / (i+1)
        # start_time = time.time()
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

            ## prepare the input, flatten e to be (batch_size, max_len*num_chars)
            cur_batch_size = e.shape[0]
            value_dict["x"] = e.reshape(cur_batch_size, max_len*num_chars)
            value_dict["y"] = l

            ## feed-forward and back-propogation
            value_dict = ad.eval(wengart_list, value_dict)
            gradients = ad.bprop(wengart_list, value_dict, loss=1.0)

            # update parameters
            # lr = init_lr / (cur_batch_size*((i+1) * (i+1))) ## learning rate
            for rname in gradients:
                if mlp.my_xman.isParam(rname):
                    value_dict[rname] -= lr * gradients[rname] 

        # training_time += [time.time() - start_time]
        # print i, "train_loss=", value_dict["loss"]

        #####################################
        ## validating loss
        #####################################
        val_num = len(data.validation)
        (idxs,e,l) = mb_valid.next()
        ## data
        value_dict["x"] = e.reshape(val_num, max_len*num_chars)
        value_dict["y"] = l
        ## loss
        val_loss = ad.eval(wengart_list, value_dict)["loss"]
        # val_losses += [val_loss]
        # print i, "val_loss=", val_loss

        ## save the best model
        if (min_val_loss is None) or val_loss < min_val_loss:
            best_value_dict = copy.deepcopy(value_dict)
            min_val_loss = val_loss
    print "done"

    ## predict on the test set using the best parameters
    test_num = len(data.test)
    ## data
    (idxs,e,l) = mb_test.next()
    best_value_dict["x"] = e.reshape(test_num, max_len*num_chars)
    best_value_dict["y"] = l
    ## loss
    best_value_dict = ad.eval(wengart_list, best_value_dict)
    test_loss = best_value_dict["loss"]
    test_output = best_value_dict["output"]

    ## save predicted probabilities on test set
    np.save(output_file, test_output)

    # return (training_time, val_losses, test_loss)
    # return test_loss


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

    ## training
    # (training_time, val_losses, test_loss) = main(params)
    # test_loss = main(params)
    # print "test_loss=", test_loss

    # ## save validating loss and training time
    # basename = params["output_file"].split(".npy")[0]
    # np.savetxt(basename+"_val_loss.txt", val_losses)
    # np.savetxt(basename+"_ trainig_time.txt", training_time)

    #####################
    # for testing and debugging
    ######################

    # ## load data and preprocess
    # dataset = "../data/tiny"
    # batch_size=6
    # max_len=20

    # dp = DataPreprocessor()
    # data = dp.preprocess('%s.train'%dataset, '%s.valid'%dataset, '%s.test'%dataset)
    # ## minibatches
    # mb_train = MinibatchLoader(data.training, batch_size, max_len, 
    #        len(data.chardict), len(data.labeldict))

    # (idxs,e,l) = mb_train.next() 
    # mlp = MLP([max_len*mb_train.num_chars, num_hid, mb_train.num_labels])
