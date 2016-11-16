.. ---------------------------------------------------------------------------
.. Copyright 2016 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------


Installation
************

To install |geon|, you must first install our neon in a virtual environment. For neon install instructions, see: http://neon.nervanasys.com/.

Activate the neon virtualenv with ``. .venv/bin/activate``, then run::

    git clone git@github.com:NervanaSystems/ngraph.git
    cd ngraph
    make install

Examples
========

Several jupyter notebook walk-throughs demonstrate how to use Nervana Graph:

* ``ngraph/examples/walk_through/`` guides developers through writing logistic regression with ngraph
* ``ngraph/examples/mnist/MNIST_Direct.ipynb`` demonstrates building a deep learning model using ngraph directly.

The neon frontend can also be used to define and train deep learning models:

* ``ngraph/examples/mnist/mnist_mlp.py``: multi-layer perceptron network on MNIST dataset
* ``ngraph/examples/cifar10/cifar10_conv.py``: convolutional neural network on CIFAR-10
* ``ngraph/examples/cifar10/cifar10_mlp.py``: multi-layer perceptron on CIFAR-10 dataset
* ``ngraph/examples/ptb/char_rnn.py`` character-level RNN model on Penn Treebank data.

We also include examples for using tensorflow to define graphs that are then passed to ngraph for execution:

* ``ngraph/frontends/tensorflow/examples/minimal.py``
* ``ngraph/frontends/tensorflow/examples/logistic_regression.py``
* ``ngraph/frontends/tensorflow/examples/mnist_mlp.py``


Developer Guide
===============

Before checking in code, run the unit tests and check for style errors::

    make test
    make style

Documentation can be generated via::

    make doc

And viewed at ``doc/build/html/index.html``.
