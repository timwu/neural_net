#!/bin/sh

BASEDIR=`dirname "${BASH_SOURCE[0]}"`

mkdir -p $BASEDIR/data
cd $BASEDIR/data

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

wget http://deeplearning.net/data/mnist/mnist.pkl.gz