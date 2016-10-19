from __future__ import division
import math

from ngraph.op_graph import op_graph
from ngraph.op_graph.axes import FunctionAxis, PaddedAxis, Axis, Axes
import ngraph as ng


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


class fprop_conv(op_graph.TensorOp):
    _index = 0

    def __init__(self, dims, inputs, filters, *args, **kwargs):
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
        axes = Axes([Axis(dim) for dim in dims.dimO[:-1]]) + self.batch_axis
        axes[0].name = 'C'
        axes[1].name = 'D'
        axes[2].name = 'H'
        axes[3].name = 'W'

        self._input_shape = inputs.shape
        self._filter_shape = filters.shape
        fprop_conv._index += 1
        self.index = fprop_conv._index
        self.dims = dims

        super(fprop_conv, self).__init__(
            args=(inputs, filters), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs, filters):
        """
        TODO
        """

        # TODO: call generate_add_delta() instead
        adjoints[filters] = update_conv(delta, inputs, filters, self)
        adjoints[inputs] = bprop_conv(delta, inputs, filters, self)


class update_conv(op_graph.TensorOp):
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
        self.dims = conv.dims

        super(update_conv, self).__init__(
            args=(delta, inputs, filters), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs, filters):
        pass


class bprop_conv(op_graph.TensorOp):
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
        self.dims = conv.dims

        super(bprop_conv, self).__init__(
            args=(delta, inputs, filters), *args, axes=axes, **kwargs
        )


    def generate_adjoints(self, adjoints, delta, inputs, filters):
        pass


class fprop_pool(op_graph.TensorOp):
    _index = 0

    def __init__(self, dims, inputs, argmax, *args, **kwargs):
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
        axes = Axes([Axis(dim) for dim in dims.dimO[:-1]]) + self.batch_axis
        axes[0].name = 'C'
        axes[1].name = 'D'
        axes[2].name = 'H'
        axes[3].name = 'W'

        self._input_shape = inputs.shape
        fprop_pool._index += 1
        self.index = fprop_pool._index
        self.dims = dims
        self.argmax = argmax

        super(fprop_pool, self).__init__(
            args=(inputs, argmax), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs, filters):
        # TODO: call generate_add_delta() instead
        adjoints[inputs] = bprop_pool(delta, inputs, self)


class bprop_pool(op_graph.TensorOp):
    def __init__(self, delta, inputs, pooling, *args, **kwargs):
        """
        Arguments:
            inputs  : input tensor.
        """
        input_dims = [shape.length for shape in inputs.shape]
        axes = Axes([Axis(dim) for dim in input_dims])
        self.index = pooling.index
        self.dims = pooling.dims
        self.argmax = pooling.argmax

        super(bprop_pool, self).__init__(
            args=(delta, inputs, self.argmax), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, inputs, filters):
        pass
