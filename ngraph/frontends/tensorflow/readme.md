# TensorFlow importer for ngraph

## Minimal example

```python
from __future__ import print_function
from ngraph.frontends.tensorflow.tf_importer.importer import TFImporter
import tensorflow as tf
import ngraph.transformers as ngt

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
f_result = ngt.make_transformer().computation(f_ng)()
print(f_result)
```

## Example models

- MNIST MLP
- Logistic regression
