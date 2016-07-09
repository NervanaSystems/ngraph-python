#!/usr/bin/env python
# ----------------------------------------------------------------------------
# Copyright 2015-2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Train a small multi-layer perceptron with fully connected layers on MNIST data.

This example has some command line arguments that enable different neon features.

Examples:

    python examples/mnist_mlp.py -b gpu -e 10

        Run the example for 10 epochs using the NervanaGPU backend

    python examples/mnist_mlp.py --eval_freq 1

        After each training epoch, process the validation/test data
        set through the model and display the cost.

    python examples/mnist_mlp.py --serialize 1 -s checkpoint.pkl

        After every iteration of training, dump the model to a pickle
        file named "checkpoint.pkl".  Changing the serialize parameter
        changes the frequency at which the model is saved.

    python examples/mnist_mlp.py --model_file checkpoint.pkl

        Before starting to train the model, set the model state to
        the values stored in the checkpoint file named checkpoint.pkl.

"""
from geon.backends.graph.graphneon import *
from neon.data import ArrayIterator, load_mnist
from neon.initializers import Gaussian
from neon.util.argparser import NeonArgparser
from neon import logger as neon_logger


# parse the command line arguments
parser = NeonArgparser(__doc__)
parser.set_defaults(backend='dataloader')

args = parser.parse_args()

# load up the mnist data set
# split into train and tests sets
(X_train, y_train), (X_test, y_test), nclass = load_mnist(path=args.data_dir)

# setup a training set iterator
train_set = ArrayIterator(X_train, y_train, nclass=nclass, lshape=(1, 28, 28))
# setup a validation data set iterator
valid_set = ArrayIterator(X_test, y_test, nclass=nclass, lshape=(1, 28, 28))

# setup weight initialization function
#init_norm = Gaussian(loc=0.0, scale=0.01)
init_norm = Uniform(low=-.01, high=.01)

# setup model layers
layers = [Affine(nout=100, init=init_norm, activation=Rectlin()),
          Affine(nout=10, init=init_norm, activation=Softmax())]

# setup cost function as CrossEntropy
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# setup optimizer
optimizer = GradientDescentMomentum(
    0.1, momentum_coef=0.9, stochastic_round=args.rounding)

# initialize model object
mlp = Model(layers=layers)

# configure callbacks
callbacks = Callbacks(mlp, eval_set=valid_set, **args.callback_args)

# run fit
mlp.fit(train_set, input_axes=(ax.C, ax.H, ax.W), target_axes=(ax.Y,), optimizer=optimizer,
        num_epochs=args.epochs, cost=cost, callbacks=callbacks)
error_rate = mlp.eval(valid_set, metric=Misclassification())
neon_logger.display('Misclassification error = %.1f%%' % (error_rate * 100))
