from __future__ import division
import math

from ngraph.op_graph import op_graph
from ngraph.op_graph import arrayaxes


def _output_dim(X, S, padding, strides, pooling=False):
    """
    Compute along 1 dimension, with these sizes, what will be the output dimension.

    Arguments:
        X (int): input data dimension
        S (int): filter dimension
        padding (int): padding on each side
        strides (int): striding
        pooling (bool): flag for setting pooling layer size
    """

    # caffe compat disabled for now
    if False and pooling:
        size = int(math.ceil((float(X - S + 2 * padding) / strides))) + 1
        if padding > 0 and (size - 1) * strides >= X + padding:
            # decrement size if last pooling op is completely in padding
            size -= 1
    else:
        # normal neon output size determination
        size = ((X - S + 2 * padding) // strides) + 1

    if pooling and padding >= S:
        raise ValueError("Padding dim %d incompatible with filter size %d" % (padding, S))

    return size


class convolution(op_graph.TensorOp):
    """
    conv op

    TODO: rename
    """

    def __init__(self, input, filter,
                 padding=None,
                 strides=None,
                 *args, **kwargs):
        """
        Arguments:
          input: input tensor.  The axes should be in order channels,
            height, width, batch_size.  In the case of video or other 4d data
            it should be in order channels, depth, height, width, batch_size.
          filter: filter/kernel tensor.  axes order should be the same as
            input tensor.
          padding: amount of zero-padding around the given edge
          strides: factor to step the filters by in a given direction
        """
        convolution._check_filter_axes(filter)

        # fill default values of padding if any are missing
        if padding is None:
            padding = [0] * len(filter.axes)
        padding = convolution._reshape_4d(padding, 0)
        self._padding = padding

        # fill default values of strides if any are missing
        if strides is None:
            strides = [1] * len(filter.axes)
        strides = convolution._reshape_4d(strides, 1)
        self._strides = strides

        if 'axes' in kwargs:
            raise ValueError(
                "conv does not currently support the 'axes' argument.  The "
                "output axes are entirely determined by the shape of the "
                "input and filter Ops."
            )

        input_shape = convolution._input_reshape(input.axes)
        filter_shape = convolution._filter_reshape(filter.axes)
        batch_axes = input.axes.batch_axes()

        if len(batch_axes) != 1:
            raise ValueError("Input must have one batch axis")
        self.batch_axis = batch_axes[0]

        # TODO: support int arguments to Axes?
        axes = arrayaxes.Axes([
            arrayaxes.Axis(filter_shape[0]),
            arrayaxes.Axis(_output_dim(input_shape[1], filter_shape[1], padding[0], strides[0])),
            arrayaxes.Axis(_output_dim(input_shape[2], filter_shape[2], padding[1], strides[1])),
            arrayaxes.Axis(_output_dim(input_shape[3], filter_shape[3], padding[2], strides[2])),
            self.batch_axis,
        ])

        self._input_shape = input_shape
        self._filter_shape = filter_shape

        # NOTE: calling constructor without axes because we need args set
        # before computing axes, and this constructor sets args.
        super(convolution, self).__init__(
            args=(input, filter), *args, axes=axes, **kwargs
        )

    @property
    def filter(self):
        """ Returns Op for filter argument to conv """
        return self.args[1]

    @property
    def input(self):
        """ Returns Op for input argument to conv """
        return self.args[0]

    @staticmethod
    def _filter_reshape(filter_axes):
        """ take a filter shape as input and return a 5d reshape necessary for
        ConvLayer """
        filter_shape = filter_axes.sample_axes().lengths

        if len(filter_shape) == 4:
            # 4d interpreted as Cin H W Cout
            return filter_shape[:1] + (1,) + filter_shape[1:]
        elif len(filter_shape) == 5:
            # 5d interpreted as Cin D H W Cout
            return filter_shape
        else:
            raise ValueError((
                'filter shape must be ..., but found {}'
            ).format(len(filter_shape)))

    @staticmethod
    def _reshape_4d(shape, default):
        """ take an input tensor shape and return a 4d tensor shape padded
        with a default value to make a 4d shape for ConvLayer.

        Returns:
            a shape tuple with axes in order: Channels Depth Height Width
        """

        if len(shape) < 2:
            raise ValueError((
                'input shape of sample axes must be at least 2 dimensions, '
                'was {}'
            ).format(len(shape)))
        elif len(shape) == 2:
            # 2d interpreted as H W
            return [default, default, shape[0], shape[1]]
        elif len(shape) == 3:
            # 3d interpreted as C H W
            return [shape[0], default, shape[1], shape[2]]
        elif len(shape) == 4:
            # 4d interpreted as C D H W
            return shape
        else:
            raise ValueError((
                'input shape of sample axes must be at most 4 dimensions, '
                'was {}'
            ).format(len(shape)))

    @staticmethod
    def _input_reshape(axes):
        """ given an axes object, return the shape of the 4d tensor ConvLayer
        wants.

        Returns:
            a shape tuple with axes in order: Channels Depth Height Width
        """
        return convolution._reshape_4d(
            axes.sample_axes().lengths, 1
        )

    @staticmethod
    def _check_filter_axes(filter):
        """ ensure filter_axes have no batch_axes """

        filter_batch_axes = filter.axes.batch_axes()
        if filter_batch_axes:
            raise ValueError((
                "filter's axes should not contain any batch_axes.  Found "
                "batch axes: {filter_batch_axes}."
            ).format(
                filter_batch_axes=', '.join(map(str, filter_batch_axes))
            ))

    def transform(self, transformer, out, input, filter):
        """
        send axes information as well as padding/strides info along to the
        transformer
        """

        transformer.conv(
            input, filter, out,
            self._input_shape, self._filter_shape,
            self._padding, self._strides,
        )

    def generate_adjoints(self, adjoints, delta, input, filter):
        """
        warning: no adjoints computed for filter for now.
        """

        if any(stride != 1 for stride in self._strides):
            raise ValueError((
                'conv adjoints doesnt currently support non-one strides. '
                'found: {}'
            ).format(self._strides))

        # TODO: delta has N in the axes, but convolution doesn't allow an N in
        # the filter's axes.  Should there be a reduce before the convolution
        # or after?  If after, how to get convolution to run inspite of N.
        # filter.generate_add_delta(adjoints, convolution(input, delta))
        input.generate_add_delta(
            adjoints, convolution(
                delta,
                op_graph.Slice(filter, [
                    slice(None, None, -1)
                    for _ in range(len(filter.shape))
                ]),
                padding=[
                    axis.length - 1 + padding
                    for axis, padding in zip(filter.axes, self._padding)
                ],
            )
        )
