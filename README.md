# Nervana Graph

Nervana Graph is Nervana's computational graph.

## Installation

First ensure you have [private-neon](https://github.com/NervanaSystems/private-neon) checked out and built.

To install Nervana Graph into your private-neon virtual env:

```
cd private-neon
make PY=2 # or "make PY=3" to instead build a Python 3 virtual environment.
. .venv/bin/activate
cd ../graphiti/
make install
```

To uninstall NervanaGraph from your virtual env:
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

The latest html documentation is also built by Jenkins and can be viewed
[here](http://jenkins.localdomain:8080/job/NEON_Graphiti_Integration_Test/lastSuccessfulBuild/artifact/ununoctium/doc/build/html/index.html)


## Examples

`examples/walk_through/log_res.py` is a simple example of using graph operations.
`examples/mnist_mlp.py` uses the neon front-end to define and train the model.
`examples/cifar10_mlp.py` uses the neon front-end to define and train the model.