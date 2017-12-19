## ONNX importer for ngraph

This module allows users to import and execute models
serialized in [ONNX](http://onnx.ai/) using ngraph.

Current support is limited to the operations listed below.

#### Usage example

If you have a simple model `y = a + b` stored in an ONNX file named `y_equals_a_plus_b.onnx.pb`, you can import it using the following code.
The `transformer.computation` line creates an executable version of the model.

```python
    >>> from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_file

    >>> onnx_protobuf = onnx.load()
    >>> import_onnx_file('y_equals_a_plus_b.onnx.pb')
    [{
        'name': 'Y',
        'inputs': [<AssignableTensorOp(placeholder):4552991464>,
                   <AssignableTensorOp(placeholder):4510192360>],
        'output': <Add(Add_0):4552894504>
    }]

    >>> ng_model = import_onnx_model(model)[0]
    >>> transformer = ng.transformers.make_transformer()
    >>> computation = transformer.computation(ng_model['output'], *ng_model['inputs'])
    >>> computation(4, 6)
    array([ 10.], dtype=float32)
```

#### Supported ONNX operations

* Abs
* Add
* ArgMax
* ArgMin
* AveragePool
* BatchNormalization
* Ceil
* Concat
* Constant
* Conv
* ConvTranspose
* Div
* Elu
* Exp
* Flatten
* Floor
* Gemm
* GlobalAveragePool
* GlobalMaxPool
* LeakyRelu
* Log
* MatMul
* Max
* MaxPool
* Mean
* Min
* Mul
* Neg
* PRelu
* Pad
* Reciprocal
* ReduceLogSumExp
* ReduceMax
* ReduceMean
* ReduceMin
* ReduceProd
* ReduceSum
* Relu
* Reshape
* Selu
* Sigmoid
* Slice
* Split
* Sqrt
* Squeeze
* Sub
* Sum
* Tanh
* Transpose


Refer to ONNX docs for the complete
[operator list](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
