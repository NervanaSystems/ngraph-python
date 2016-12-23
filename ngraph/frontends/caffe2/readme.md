# Caffe2 importer for ngraph

## Minimal example

```python
from __future__ import print_function
from ngraph.frontends.caffe2.c2_importer.importer import C2Importer
from caffe2.python import core, workspace
import ngraph.transformers as ngt


# Caffe2 - network creation
net = core.Net("net")
X = net.ConstantFill([], ["X"], shape=[2,2], value=2.0, run_once=0, name="X")
W = net.ConstantFill([], ["W"], shape=[2,2], value=3.0, run_once=0, name="W")
b = net.ConstantFill([], ["b"], shape=[2,], value=1.0, run_once=0, name="b")
Y = X.FC([W, b], ["Y"], name="Y")

# Import caffe2 network into ngraph
importer = C2Importer()
importer.parse_net_def(net.Proto(), False)

# Get handle
f_ng = importer.get_op_handle("Y")

# Execute
f_result = ngt.make_transformer().computation(f_ng)()
print(f_result)

```

## Example models

- MNIST MLP - TODO
- Logistic regression - TODO
