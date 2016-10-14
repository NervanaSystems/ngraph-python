from __future__ import division
import math

from ngraph.op_graph import op_graph
from ngraph.op_graph.axes import FunctionAxis, PaddedAxis, Axis, Axes


def _output_dim(X, S, padding, stride, pooling=False):
    """
    Compute along 1 dimension, with these sizes, what will be the output dimension.

    Arguments:
        X (int): input data dimension
        S (int): filter dimension
        padding (int): padding on each side
        stride (int): striding
        pooling (bool): flag for setting pooling layer size
    """

    # caffe compat disabled for now
    if False and pooling:
        size = int(math.ceil((float(X - S + 2 * padding) / stride))) + 1
        if padding > 0 and (size - 1) * stride >= X + padding:
            # decrement size if last pooling op is completely in padding
            size -= 1
    else:
        # normal neon output size determination
        size = ((X - S + 2 * padding) // stride) + 1

    if pooling and padding >= S:
        raise ValueError("Padding dim %d incompatible with filter size %d" % (padding, S))

    return size


class ConvolutionAxis(FunctionAxis):
    """
    An axis created by taking a convolution of another axis

    The length is computed dynamically from the length of the parent, along
    with the width and stride of the convolution.

    Arguments:
        parent: The axis being sliced.
        width: the width of the convolution filter
        stride: the convolution's stride
        kwargs: Arguments for related classes.
    """
    def __new__(cls, parent, width, stride, **kwargs):
        """
        This method acts like a factory for ConvolutionAxis.

        Detect cases when this ConvolutionAxis is a nop and return the parent
        or great grandparent instead
        """
        if width == 1 and stride == 1:
            return parent

        # if this is a Conv(Pad(Conv(a))) whose length will be the same as a,
        # then just return axis a.
        if isinstance(parent, PaddedAxis):
            if isinstance(parent.parent, ConvolutionAxis):
                length = _output_dim(parent.length, width, 0, stride)
                if length == parent.parent.parent.length:
                    return parent.parent.parent

        return super(ConvolutionAxis, cls).__new__(cls)

    def __init__(self, parent, width, stride, **kwargs):
        self.parent = parent
        self.width = width
        self.stride = stride

        super(ConvolutionAxis, self).__init__(
            parent=parent,
            length_fun=lambda: _output_dim(parent.length, width, 0, stride),
            **kwargs
        )

    def __repr__(self):
        return (
            'ConvolutionAxis({name}: {length}; parent: {parent}; width: {width}; stride: {stride})'
        ).format(
            name=self.name,
            length=self.length,
            parent=self.parent,
            width=self.width,
            stride=self.stride,
        )


class conv_fprop(op_graph.TensorOp):
    _index = 0

    def __init__(self, inputs, filters, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.

        Return:
        """
        if len(inputs.shape) != 5:
            raise ValueError((
                'convolution input shape must be length 5, found {}'
            ).format(len(inputs.shape)))

        if len(filters.shape) != 5:
            raise ValueError((
                'convolution filter shape must be length 5, found {}'
            ).format(len(filters.shape)))

        if 'axes' in kwargs:
            raise ValueError(
                "convolution does not currently support the 'axes' argument.  The "
                "output axes are entirely determined by the shape of the "
                "input and filter Ops."
            )

        if inputs.axes[0].length != filters.axes[0].length:
            raise ValueError((
                'the first axis in input and filter must be the same.  The '
                'first axis in input is {inputs} and in filter is {filters}.'
            ).format(
                inputs=inputs.axes[0],
                filters=filters.axes[0],
            ))

        batch_axes = inputs.axes.batch_axes()
        if len(batch_axes) != 1:
            raise ValueError((
                "Input must have one batch axis.  Found {n_batch_axes} batch "
                "axes: {batch_axes} and {n_sample_axes} sample axes: "
                "{sample_axes}."
            ).format(
                n_batch_axes=len(batch_axes),
                batch_axes=batch_axes,
                n_sample_axes=len(inputs.axes.sample_axes()),
                sample_axes=inputs.axes.sample_axes(),
            ))
        self.batch_axis = batch_axes[0]
        input_dims = [shape.length for shape in inputs.shape]
        filter_dims = [shape.length for shape in filters.shape]
        # TODO: account for padding and stride
        output_dims = [_output_dim(input_dims[i], filter_dims[i], 0, 1) for i in range(1, 4)]
        output_dims = [filter_dims[-1]] + output_dims
        axes = Axes([Axis(dim) for dim in output_dims] + [batch_axes[0]])
        axes[0].name = 'C'
        axes[1].name = 'D'
        axes[2].name = 'H'
        axes[3].name = 'W'

        self._input_shape = inputs.shape
        self._filter_shape = filters.shape
        conv_fprop._index += 1
        self.index = conv_fprop._index

        super(conv_fprop, self).__init__(
            args=(inputs, filters), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs, filters):
        """
        TODO
        """

        # TODO: call generate_add_delta() instead
        adjoints[filters] = conv_update(delta, inputs, filters, self)
        adjoints[inputs] = conv_bprop(delta, inputs, filters, self)


class conv_update(op_graph.TensorOp):
    def __init__(self, delta, inputs, filters, conv, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.
        """
        filter_dims = [shape.length for shape in filters.shape]
        axes = Axes([Axis(dim) for dim in filter_dims])
        self._input_shape = conv._input_shape
        self._filter_shape = conv._filter_shape
        self.index = conv.index

        super(conv_update, self).__init__(
            args=(delta, inputs, filters), *args, axes=axes, **kwargs
        )


class conv_bprop(op_graph.TensorOp):
    def __init__(self, delta, inputs, filters, conv, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.
        """
        input_dims = [shape.length for shape in inputs.shape]
        axes = Axes([Axis(dim) for dim in input_dims])
        self._input_shape = conv._input_shape
        self._filter_shape = conv._filter_shape
        self.index = conv.index

        super(conv_bprop, self).__init__(
            args=(delta, inputs, filters), *args, axes=axes, **kwargs
        )


class pool_fprop(op_graph.TensorOp):
    _index = 0

    def __init__(self, inputs, pool_params, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
            filters : filter/kernel tensor.

        Return:
        """
        if len(inputs.shape) != 5:
            raise ValueError((
                'pooling input shape must be length 5, found {}'
            ).format(len(inputs.shape)))

        if 'axes' in kwargs:
            raise ValueError(
                "pooling does not currently support the 'axes' argument.  The "
                "output axes are entirely determined by the shape of the "
                "input and filter Ops."
            )

        batch_axes = inputs.axes.batch_axes()
        if len(batch_axes) != 1:
            raise ValueError((
                "Input must have one batch axis.  Found {n_batch_axes} batch "
                "axes: {batch_axes} and {n_sample_axes} sample axes: "
                "{sample_axes}."
            ).format(
                n_batch_axes=len(batch_axes),
                batch_axes=batch_axes,
                n_sample_axes=len(inputs.axes.sample_axes()),
                sample_axes=inputs.axes.sample_axes(),
            ))
        self.batch_axis = batch_axes[0]
        input_dims = [shape.length for shape in inputs.shape]
        pool_dims = [pool_params[name] for name in ['C', 'T', 'R', 'S']]
        # TODO: account for padding and stride
        output_dims = [_output_dim(input_dims[i], pool_dims[i], 0, 1, pooling=True) for i in range(1, 4)]
        output_dims = [pool_dims[0]] + output_dims
        axes = Axes([Axis(dim) for dim in output_dims] + [batch_axes[0]])
        axes[0].name = 'C'
        axes[1].name = 'D'
        axes[2].name = 'H'
        axes[3].name = 'W'

        self._input_shape = inputs.shape
        pool_fprop._index += 1
        self.index = pool_fprop._index

        super(pool_fprop, self).__init__(
            args=(inputs, pool_params), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs):
        """
        TODO
        """

        # TODO: call generate_add_delta() instead
        adjoints[inputs] = pool_bprop(delta, inputs, self)


class pool_bprop(op_graph.TensorOp):
    def __init__(self, delta, inputs, pooling, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
        """
        input_dims = [shape.length for shape in inputs.shape]
        axes = Axes([Axis(dim) for dim in input_dims])
        self._input_shape = pooling._input_shape
        self._filter_shape = pooling._filter_shape
        self.index = pooling.index

        super(pool_bprop, self).__init__(
            args=(delta, inputs), *args, axes=axes, **kwargs
        )
