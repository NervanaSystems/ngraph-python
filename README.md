# Graphiti: Computation Graphs

graphiti/ununoctium is Nervana's neon graph backend.

## Installation

First ensure you have [private-neon](https://github.com/NervanaSystems/private-neon) checked out and built.

To install Graphiti into your private-neon virtual env:

```
cd private-neon
make PY=2 # or "make PY=3" to instead build a Python 3 virtual environment.
. .venv/bin/activate
cd ../graphiti/ununoctium
make install
```

To uninstall graphiti from your virtual env:
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

## Examples

ununoctium/examples/models.py is a cifar_mlp example that runs.

```
python examples/models.py -vvv -w /usr/local/data/I1K/macrobatches
```

ununoctium/examples/dtest.py is cifar_mlp for the new front end.

ununoctium/examples/rnn.py is a simple rnn example using the new front end that matches a sequence of inputs and outputs.
