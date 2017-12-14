## ONNX importer for ngraph

This module will allow users to import and execute models
serialized using [ONNX](https://github.com/onnx/onnx/) in ngraph.

Current support is limited and should be considered a **proof of concept**.
As more ONNX operations are supported this solution will become viable.

#### Minimal example

```python
    >>> import onnx
    >>> from ngraph.frontends.onnx.onnx_importer.importer import import_onnx_model

    >>> onnx_protobuf = onnx.load('y_equals_a_plus_b.onnx.pb')
    >>> import_onnx_model(onnx_protobuf)
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
* AveragePool
* BatchNormalization
* Concat
* Constant
* Conv
* ConvTranspose
* Div
* Elu
* Flatten
* Gemm
* GlobalAveragePool
* GlobalMaxPool
* LeakyRelu
* MatMul
* MaxPool
* Mul
* PRelu
* Pad
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
* Squeeze
* Sub
* Tanh
* Transpose

Refer to ONNX docs for the complete
[operator list](https://github.com/onnx/onnx/blob/master/docs/Operators.md).
