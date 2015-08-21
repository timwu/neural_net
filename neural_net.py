import gzip
import pickle

import numpy as np
import theano.tensor as T
import theano
import sklearn.metrics


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
        return T.maximum(0, T.dot(input, self.W) + self.b)

    def connect(self, next_layer):
        assert self.shape[1] == next_layer.shape[0]

        return MLP(self).connect(next_layer)

    @property
    def l2(self):
        return T.mean(T.power(self.W, 2))


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
        return T.mean(T.power(self.W, 2))


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
        return reduce(lambda x, y: x + y, [layer.l2 for layer in self.layers]) / len(self.layers)

    def connect(self, next_layer):
        assert next_layer.shape[0] == self.out_layer.shape[1]
        self.layers.append(next_layer)
        return self


def main():
    with gzip.open("data/mnist.pkl.gz") as f:
        train_set, valid_set, test_set = pickle.load(f)

    x = T.matrix('x')
    y = T.ivector('y')

    mlp = HiddenLayer(784, 64).connect(HiddenLayer(64, 64)).connect(HiddenLayer(64, 32)).connect(LogisticLayer(32, 10))
    mlp_output = mlp.output(x)
    mlp_predictions = T.argmax(mlp_output, axis=1)

    NLL = -T.mean(T.log(mlp_output)[T.arange(y.shape[0]), y % 10])
    cost = NLL + 0.1 * mlp.l2

    alpha = 0.1
    updates = [(param, param - alpha * T.grad(cost, param)) for param in mlp.params]

    train_mlp = theano.function(inputs=[x, y], outputs=cost, updates=updates)
    predict = theano.function(inputs=[x], outputs=mlp_predictions)

    prev_cost = np.inf
    for i in range(1000):
        cur_cost = train_mlp(train_set[0], train_set[1].astype('int32'))
        print cur_cost
        if i > 50 and abs((cur_cost - prev_cost) / prev_cost) < 0.001:
            print "Gave up after %i iterations" % i
            break
        prev_cost = cur_cost

    predictions = predict(test_set[0])

    print(sklearn.metrics.confusion_matrix(test_set[1], predictions))
    print(sklearn.metrics.classification_report(test_set[1], predictions))

if __name__ == "__main__":
    main()
