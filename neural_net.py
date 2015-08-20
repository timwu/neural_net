import numpy as np
import theano.tensor as T
import theano

import gzip, pickle


class Layer(object):
    def __init__(self, input, n_in, n_out, activation=None):
        self.input = input
        self.activation = activation

        self.W = None

        self.b = theano.shared(np.zeros((n_in,), dtype=theano.config.floatX),
                               name='b', borrow=True)

    def _lin_output(self):
        return T.dot(self.input, self.W) + self.b

    @property
    def params(self):
        return [self.W, self.b]


class HiddenLayer(Layer):
    def __init__(self, input, n_in, n_out):
        Layer.__init__(self, input, n_in, n_out)

        rng = np.random.RandomState(123123)

        self.W = theano.shared(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                           high=np.sqrt(6. / (n_in + n_out)),
                                           size=(n_in, n_out)).astype(theano.config.floatX),
                               name='W',
                               borrow=True)

    @property
    def output(self):
        return T.tanh(self._lin_output())


class LogisticLayer(Layer):
    def __init__(self, input, n_in, n_out):
        Layer.__init__(self, input, n_in, n_out, T.nnet.softmax)

        self.W = theano.shared(np.zeros((n_in, n_out), dtype=theano.config.floatX),
                               name='W', borrow=True)

    @property
    def output(self):
        return T.nnet.softmax(self._lin_output())


def main():
    with gzip.open("data/mnist.pkl.gz") as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

    x = T.matrix('x')


if __name__=="__main__":
    main()