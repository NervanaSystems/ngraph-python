# Nervana Graph

Nervana Graph is Nervana's computational graph.

## Installation

First ensure you have [neon](https://github.com/NervanaSystems/neon) checked out and built.

To install Nervana Graph into your neon virtual env:

```
cd neon
make PY=2 # or "make PY=3" to instead build a Python 3 virtual environment.
. .venv/bin/activate
cd ../ngraph/
make install
```

To uninstall Nervana Graph from your virtual env:
```
make uninstall
```

To run the unit tests:
```
make test
```

Before checking in code, ensure no "make style" errors
```
make style
```

To fix style errors
```
make fixstyle
```

To generate the documentation as html files
```
make doc
```

## Examples

* ``ngraph/examples/walk_through/`` contains several code walk throughs.
* ``ngraph/examples/mnist/mnist__mlp.py`` uses the neon front-end to define and train a MLP model on MNIST data.
* ``ngraph/examples/cifar10/cifar10_conv.py`` uses the neon front-end to define and train a CNN model on CIFAR10 data.
* ``ngraph/examples/cifar10/cifar10_mlp.py`` uses the neon front-end to define and train a MLP model on CIFAR10 data.
* ``ngraph/examples/ptb/char_rnn.py`` uses the neon front-end to define and train a character-level RNN model on Penn Treebank data.