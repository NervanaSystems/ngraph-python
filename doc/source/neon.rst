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

neon
****

The neon frontend to ngraph wraps together common deep learning primitives, such as activation functions. optimizers, layers, and more. We include in this release several examples to illustrate how to use the neon frontend to construct your models:
- ``examples/minst/mnist_mlp.py``: multi-layer perceptron on the MNIST digits dataset.
- ``examples/cifar10/cifar10_mlp.py``: multi-layer perceptron on the CIFAR10 dataset.
- ``examples/cifar10/cifar10_conv.py``: convolutional neural networks applied to the CIFAR10 dataset.
- ``examples/ptb/char_rnn.py``: character-level RNN language model on the Penn Treebank dataset.


