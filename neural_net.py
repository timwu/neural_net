import gzip
import pickle
import sys

import numpy as np
import theano.tensor as T
import theano
import sklearn.metrics
import seaborn as sns
import matplotlib.pyplot as plt

plt.ioff()


class HiddenLayer(object):
    def __init__(self, n_in, n_out):
        rng = np.random.RandomState()

        self.shape = (n_in, n_out)

        self.W = theano.shared(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                           high=np.sqrt(6. / (n_in + n_out)),
                                           size=self.shape).astype(theano.config.floatX),
                               name='Wh',
                               borrow=True)

        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='bh', borrow=True)

        self.params = [self.W, self.b]

    def output(self, input):
        x = T.dot(input, self.W) + self.b
        return T.maximum(0, x)

    def connect(self, next_layer):
        assert self.shape[1] == next_layer.shape[0]

        return MLP(self).connect(next_layer)

    @property
    def l2(self):
        return T.sum(T.power(self.W, 2))

    @property
    def l1(self):
        return T.sum(abs(self.W))


class LogisticLayer(object):
    def __init__(self, n_in, n_out):
        self.shape = (n_in, n_out)
        self.W = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                               name='Wl', borrow=True)

        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='bl', borrow=True)

        self.params = [self.W, self.b]

    def output(self, input):
        return T.nnet.softmax(T.dot(input, self.W) + self.b)

    def connect(self, next_layer):
        assert self.shape[1] == next_layer.shape[0]
        return MLP(self).connect(next_layer)

    @property
    def l2(self):
        return T.sum(T.power(self.W, 2))

    @property
    def l1(self):
        return T.sum(abs(self.W))


class MLP(object):
    def __init__(self, in_layer):
        self.layers = [in_layer]

    def output(self, input):
        output = input
        for layer in self.layers:
            output = layer.output(output)
        return output

    @property
    def shape(self):
        return self.in_layer.shape[0], self.out_layer.shape[1]

    @property
    def in_layer(self):
        return self.layers[0]

    @property
    def out_layer(self):
        return self.layers[-1]

    @property
    def params(self):
        l = []
        for layer in self.layers:
            l += layer.params
        return l

    @property
    def l2(self):
        l2 = 0
        for layer in self.layers:
            l2 += layer.l2
        return l2

    @property
    def l1(self):
        l1 = 0
        for layer in self.layers:
            l1 += layer.l1
        return l1

    def connect(self, next_layer):
        assert next_layer.shape[0] == self.out_layer.shape[1]
        self.layers.append(next_layer)
        return self


def main():
    train_set, valid_set, test_set = load_data()

    shared_inputs, shared_labels = share_dataset(*train_set)

    x = T.matrix('x')

    mlp = HiddenLayer(784, 128).connect(HiddenLayer(128, 64)).connect(HiddenLayer(64, 32)).connect(LogisticLayer(32, 10))
    mlp_output = mlp.output(shared_inputs)

    lambda_1 = 0
    lambda_2 = 0.001

    NLL = -T.mean(T.log(mlp_output)[T.arange(shared_labels.shape[0]), shared_labels])
    cost = NLL + lambda_1 * mlp.l1 + lambda_2 * mlp.l2

    alpha = 0.1
    updates = [(param, param - alpha * T.grad(cost, param)) for param in mlp.params]

    train_mlp = theano.function(inputs=[], outputs=cost, updates=updates)

    prev_cost = np.inf
    costs = []
    for i in range(5000):
        cur_cost = train_mlp()
        print(cur_cost)
        # if i > 50 and abs((cur_cost - prev_cost) / prev_cost) < 0.00001:
        #     print "Gave up after %i iterations" % i
        #     break
        prev_cost = cur_cost
        costs.append(cur_cost)

    mlp_predictions = T.argmax(mlp.output(x), axis=1)
    predict = theano.function(inputs=[x], outputs=mlp_predictions)
    predictions = predict(test_set[0])

    print(sklearn.metrics.confusion_matrix(test_set[1], predictions))
    print(sklearn.metrics.classification_report(test_set[1], predictions))

    # sns.tsplot(costs, value="cost")
    # plt.show()


def load_data():
    with gzip.open("data/mnist.pkl.gz") as f:
        if sys.version_info.major == 3:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        else:
            train_set, valid_set, test_set = pickle.load(f)

    return train_set, valid_set, test_set


def share_dataset(inputs, labels):
    shared_input = theano.shared(inputs.astype(theano.config.floatX))
    shared_labels = theano.shared(labels.astype('int32'))

    return shared_input, shared_labels


if __name__ == "__main__":
    main()
