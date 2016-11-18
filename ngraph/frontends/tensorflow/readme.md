# TensorFlow importer for ngraph

## Minimal example

```python
from __future__ import print_function
from tf_importer.tf_importer.importer import TFImporter
import tensorflow as tf
import ngraph as ng

# tensorflow ops
x = tf.constant(1.)
y = tf.constant(2.)
f = x + y

# import
importer = TFImporter()
importer.parse_graph_def(tf.get_default_graph().as_graph_def())

# get handle
f_ng = importer.get_op_handle(f)

# execute
f_result = ng.NumPyTransformer().computation(f_ng)()
print(f_result)
```

## Example models

- MNIST MLP
- Logistic regression
