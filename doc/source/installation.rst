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

To install |geon|, you must first install neon in a virtual environment.
Activate the neon virtualenv with ``. .venv/bin/activate``, then run::

    git clone git@github.com:NervanaSystems/ngraph.git
    cd ngraph
    make install

Examples
========

Several useful example scripts demonstrate how to use Nervana Graph:

* ``ngraph/examples/walk_through/`` contains several code walk throughs.
* ``ngraph/examples/mnist/mnist_mlp.py`` uses the neon front-end to define and train a MLP model on MNIST data.
* ``ngraph/examples/cifar10/cifar10_conv.py`` uses the neon front-end to define and train a CNN model on CIFAR10 data.
* ``ngraph/examples/cifar10/cifar10_mlp.py`` uses the neon front-end to define and train a MLP model on CIFAR10 data.
* ``ngraph/examples/ptb/char_rnn.py`` uses the neon front-end to define and train a character-level RNN model on Penn Treebank data.

Developer Guide
===============

Before checking in code, run the unit tests and check for style errors::

    make test
    make style

Documentation can be generated via::

    make doc

