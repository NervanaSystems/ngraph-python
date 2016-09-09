# Nervana Graph

Nervana Graph is Nervana's computational graph.

## Temporary Notice

Ths repository changed names on 9/10/2016 from "graphiti" to "ngraph"

Although github.com supports the old name, it is recommended to stop using graphiti by
either cloning anew or changing your checked out origin as follows:

```
cd ~/code
mv graphiti ngraph
cd ngraph
git remote set-url origin https://github.com/NervanaSystems/ngraph
```

## Installation

First ensure you have [private-neon](https://github.com/NervanaSystems/private-neon) checked out and built.

To install Nervana Graph into your private-neon virtual env:

```
cd private-neon
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

The latest html documentation is also built by Jenkins and can be viewed
[here](http://jenkins.localdomain:8080/job/NEON_NGRAPH_Integration_Test/lastSuccessfulBuild/artifact/doc/build/html/index.html)


## Examples

`examples/walk_through/log_res.py` is a simple example of using graph operations.
`examples/mnist_mlp.py` uses the neon front-end to define and train the model.
`examples/cifar10_mlp.py` uses the neon front-end to define and train the model.