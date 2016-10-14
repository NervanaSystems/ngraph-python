from __future__ import division
import math

from ngraph.op_graph import op_graph
from ngraph.op_graph.axes import FunctionAxis, PaddedAxis, Axes


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


class convolution1d(op_graph.TensorOp):
    def __init__(self, input, filter,
                 *args, **kwargs):
        """
        Arguments:
            input: input tensor.  axes should be (channels, length, batch_size)
            filter: filter/kernel tensor.  axes should be (input_channels,
                length, output_channels)

        Return:
            shape will be (filter[2], f(input[1], filter[1]), input[2])
        """
        if len(input.shape) != 3:
            raise ValueError((
                'convolution1d input shape must be length 3 (channels, '
                'length, batch_size), found {}'
            ).format(len(input.shape)))

        if len(filter.shape) != 3:
            raise ValueError((
                'convolution1d filter shape must be length 3 '
                '(input_channels, length, output_channels), found {}'
            ).format(len(filter.shape)))

        if 'axes' in kwargs:
            raise ValueError(
                "convolution1d does not currently support the 'axes' argument.  The "
                "output axes are entirely determined by the shape of the "
                "input and filter Ops."
            )

        if input.axes[0] != filter.axes[0]:
            raise ValueError((
                'the first axis in input and filter must be the same.  The '
                'first axis in input is {input} and in filter is {filter}.'
            ).format(
                input=input.axes[0],
                filter=filter.axes[0],
            ))

        batch_axes = input.axes.batch_axes()
        if len(batch_axes) != 1:
            raise ValueError((
                "Input must have one batch axis.  Found {n_batch_axes} batch "
                "axes: {batch_axes} and {n_sample_axes} sample axes: "
                "{sample_axes}."
            ).format(
                n_batch_axes=len(batch_axes),
                batch_axes=batch_axes,
                n_sample_axes=len(input.axes.sample_axes()),
                sample_axes=input.axes.sample_axes(),
            ))
        self.batch_axis = batch_axes[0]

        # TODO: support int arguments to Axes?
        # TODO: make a ConvAxis instead of creating an Axis with computed
        # values.
        axes = Axes([
            filter.shape[2],
            ConvolutionAxis(input.shape[1], filter.shape[1].length, 1),
            self.batch_axis,
        ])

        self._input_shape = input.shape
        self._filter_shape = filter.shape

        # NOTE: calling constructor without axes because we need args set
        # before computing axes, and this constructor sets args.
        super(convolution1d, self).__init__(
            args=(input, filter), *args, axes=axes, **kwargs
        )

    def generate_adjoints(self, adjoints, delta, input, filter):
        """
        warning: no adjoints computed for filter for now.
        """

        # TODO: delta has N in the axes, but convolution doesn't allow an N in
        # the filter's axes.  Should there be a reduce before the convolution
        # or after?  If after, how to get convolution to run inspite of N.

        # filter.generate_add_delta(adjoints, convolution1d(input, delta))

        # TODO: add flip Op
        # reverse the order of spatial axes in filter
        flipped_filter = op_graph.Slice(filter, [
            slice(None, None, None),
            slice(None, None, -1),
            slice(None, None, None),
        ])

        flipped_filter = op_graph.Dimshuffle(flipped_filter, axes=Axes(
            (flipped_filter.axes[2], flipped_filter.axes[1], flipped_filter.axes[0])
        ))

        # TODO: pad operator that acts on just one axis: .pad(x, x.axes[1], 3)
        if filter.axes[1].length == 1:
            pad_delta = delta
        else:
            pad_delta = op_graph.pad(
                delta, [0, filter.axes[1].length - 1, 0]
            )

        conv = convolution1d(pad_delta, flipped_filter)

        # if this fails, there is something wrong with generate_adjoints
        assert conv.axes == input.axes

        input.generate_add_delta(adjoints, conv)
