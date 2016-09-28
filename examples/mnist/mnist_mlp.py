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
import numpy as np
import ngraph as ng
from ngraph.frontends.neon import Affine, Rectlin, Logistic, GeneralizedCost,\
    CrossEntropyBinary, Misclassification, GradientDescentMomentum, Model, Callbacks
from data import make_train_loader, make_validation_loader
from aeon import DataLoader, LoaderRuntimeError, gen_backend


# parse the command line arguments
# parser = NeonArgparser(__doc__)
# parser.set_defaults(backend='dataloader')

train_manifest = '/usr/local/data/mnist_ingested/manifest_train.csv'
valid_manifest = '/usr/local/data/mnist_ingested/manifest_valid.csv'

# args = parser.parse_args()
image_height = 28
image_width = 28
batch_size = 128
output_size = 10

backend = gen_backend('cpu')

N = ng.Axis(name='N', length=batch_size, batch=True)
hidden1 = ng.Axis(name='hidden1', length=100)
hidden2 = ng.Axis(name='hidden2')
inputX = ng.Axis(name='inputX', length=image_height*image_width)
outputY = ng.Axis(name='outputY', length=output_size)

# load up the mnist data set
train_set = make_train_loader(train_manifest, backend, N.length)
valid_set = make_validation_loader(valid_manifest, backend, N.length)

# # setup weight initialization function
# # init_norm = Gaussian(loc=0.0, scale=0.01)
# init_norm = Constant(0)
#
# # setup model layers
# layers = [Affine(nout=100, init=init_norm, activation=Rectlin()),
#           Affine(nout=10, init=init_norm, activation=Logistic(shortcut=True), axes=[ax.Y])]
#
# # setup cost function as CrossEntropy
# cost = GeneralizedCost(costfunc=CrossEntropyBinary())
#
# # cost = GeneralizedCost(costfunc=CrossEntropyBinary())
# # setup optimizer
# optimizer = GradientDescentMomentum(
#     0.1, momentum_coef=0.9, stochastic_round=args.rounding)

X = ng.placeholder(axes=[inputX, N])
Y = ng.placeholder(axes=[outputY, N])
W1init = np.random.normal(0, 0.01, (inputX.length, hidden1.length))
W2init = np.random.normal(0, 0.01, (hidden1.length, outputY.length))

W1 = ng.Variable(axes=[hidden1, inputX], initial_value=W1init)
# W2 = ng.Variable(axes=[hidden2, hidden1], initial_value=0)
Woutput = ng.Variable(axes=[outputY, hidden1], initial_value=W2init)

H1out = ng.maximum(ng.dot(W1, X), 0.0)
# H2out = ng.relu(ng.dot(W2, H1out))
Y_hat = ng.sigmoid(ng.dot(Woutput, H1out))

L = ng.cross_entropy_binary(Y_hat, Y) / ng.tensor_size(Y_hat)

variables = [W1, Woutput]
grads = [ng.deriv(L, variable) for variable in variables]
velocities = [ng.temporary(axes=variable.axes, initial_value=0) for variable in variables]
velocity_updates = [
    ng.assign(
        lvalue=velocity,
        rvalue=velocity * 0.9 - 0.1 * grad)
    for variable, grad, velocity in zip(variables, grads, velocities)]

param_updates = [
    ng.assign(lvalue=variable, rvalue=variable + velocity)
    for variable, velocity in zip(variables, velocities)
    ]

# grad2 = ng.deriv(L, W2)
# gradoutput = ng.deriv(L, Woutput)
update =  ng.doall(velocity_updates + param_updates)

transformer = ng.NumPyTransformer()
update_fun = transformer.computation([L, update], X, Y)

for i, data in enumerate(train_set):
    loss_val, _ = update_fun(data[0], data[1])
    print("loop {0}: loss {1}").format(i, np.mean(loss_val))

