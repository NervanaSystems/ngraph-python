# Graphiti: Computation Graphs

graphiti/ununoctium is Nervana's neon graph backend.

## Installation

First ensure you have [private-neon](https://github.com/NervanaSystems/private-neon) installed,
along with the other python packages defined in the `requirements.txt`.

If you run neon in a virtualenv, activate it then do:
```
cd ununoctium
pip install -r requirements.txt
pip install -e .
```

If you'd prefer to install system-wide do:
```
cd unonoctium
pip install -r requirements.txt
pip install .
```

To uninstall:
```
pip uninstall Graphiti
```

To run the unit tests:
```
make
```

## Examples

ununoctium/examples/models.py is a cifar_mlp example that runs.

```
python examples/models.py -vvv -w /usr/local/data/I1K/macrobatches
```

ununoctium/examples/dtest.py is cifar_mlp for the new front end.

ununoctium/examples/rnn.py is a simple rnn example using the new front end that matches a sequence of inputs and outputs.
