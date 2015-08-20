import numpy as np
import theano.tensor as T
import theano
import sklearn.metrics

import gzip, pickle


class HiddenLayer(object):
    def __init__(self, n_in, n_out):
        rng = np.random.RandomState(123123)

        self.shape = (n_in, n_out)

        self.W = theano.shared(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                           high=np.sqrt(6. / (n_in + n_out)),
                                           size=self.shape).astype(theano.config.floatX),
                               name='W',
                               borrow=True)

        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b', borrow=True)

        self.params = [self.W, self.b]

    def output(self, input):
        return T.tanh(T.dot(input, self.W) + self.b)

    def connect(self, next_layer):
        assert self.shape[1] == next_layer.shape[0]

        return MLP(self, next_layer)


class LogisticLayer(object):
    def __init__(self, n_in, n_out):
        self.shape = (n_in, n_out)
        self.W = theano.shared(np.zeros(self.shape, dtype=theano.config.floatX),
                               name='W', borrow=True)

        self.b = theano.shared(np.zeros((n_out,), dtype=theano.config.floatX),
                               name='b', borrow=True)

        self.params = [self.W, self.b]

    def output(self, input):
        return T.nnet.softmax(T.dot(input, self.W) + self.b)

    def connect(self, next_layer):
        assert self.shape[1] == next_layer.shape[0]
        return MLP(self, next_layer)


class MLP(object):
    def __init__(self, in_layer, out_layer):
        self.in_layer = in_layer
        self.out_layer = out_layer

        self.params = self.in_layer.params + self.out_layer.params

    def output(self, input):
        return self.out_layer.output(self.in_layer.output(input))


def main():
    with gzip.open("data/mnist.pkl.gz") as f:
        train_set, valid_set, test_set = pickle.load(f)

    x = T.matrix('x')
    y = T.ivector('y')

    l1 = HiddenLayer(784, 32)
    l2 = LogisticLayer(32, 10)

    mlp = l1.connect(l2)
    mlp_output = mlp.output(x)
    mlp_predictions = T.argmax(mlp_output, axis=1)

    NLL = -T.mean(T.log(mlp_output)[T.arange(y.shape[0]), y % 10])
    # TODO: add regularization somehow
    cost = NLL

    alpha = 0.1
    updates = [(param, param - alpha * T.grad(cost, param)) for param in mlp.params]

    train_mlp = theano.function(inputs=[x, y], outputs=cost, updates=updates)
    predict = theano.function(inputs=[x], outputs=mlp_predictions)

    for i in range(10):
        print(train_mlp(train_set[0], train_set[1].astype('int32')))

    predictions = predict(test_set[0])

    print(sklearn.metrics.f1_score(test_set[1], predictions, average='weighted'))

if __name__ == "__main__":
    main()
