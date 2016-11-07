# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

from tf_importer.tf_importer.ops_base import OpsBase
from tf_importer.tf_importer.utils import shape_to_axes
import collections
import ngraph as ng
import numpy as np


def _flatten(x):
    """
    https://goo.gl/yPP8hh
    """
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in _flatten(i)]
    else:
        return [x]


class OpsTransform(OpsBase):
    """
    Mix-in class for tensor transformation ops
    `<https://www.tensorflow.org/versions/r0.11/api_docs/python/array_ops.html>`_
    """

    def Rank(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Returns the rank of a tensor.

        This operation returns an integer representing the rank of `input`.

        For example:

        ```python
        # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
        # shape of tensor 't' is [2, 2, 3]
        rank(t) ==> 3
        ```

        **Note**: The rank of a tensor is not the same as the rank of a matrix. The
        rank of a tensor is the number of indices required to uniquely select each
        element of the tensor. Rank is also known as "order", "degree", or "ndims."

        Args:
            input: A `Tensor` or `SparseTensor`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor` of type `int32`.
        """
        # get inputs
        left = inputs[0]

        # get rank
        try:
            rank = len(left.axes.lengths)
        except:
            raise NotImplementedError("[NON-NATIVE] `Rank` op's axes must be "
                                      "pre-determined before execution.")
        # return
        return ng.constant(rank, ng.make_axes([]), name=tf_node.name)

    def Range(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Creates a sequence of integers.

        Creates a sequence of integers that begins at `start` and extends by
        increments of `delta` up to but not including `limit`.

        Like the Python builtin `range`, `start` defaults to 0, so that
        `range(n) = range(0, n)`.

        For example:

        ```
        # 'start' is 3
        # 'limit' is 18
        # 'delta' is 3
        tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

        # 'limit' is 5
        tf.range(limit) ==> [0, 1, 2, 3, 4]
        ```

        Args:
            start: A 0-D (scalar) of type `int32`. First entry in sequence.
                   Defaults to 0.
            limit: A 0-D (scalar) of type `int32`. Upper limit of sequence,
                   exclusive.
            delta: A 0-D `Tensor` (scalar) of type `int32`. Optional. Default is 1.
                   Number that increments `start`.
            name: A name for the operation (optional).

        Returns:
            An 1-D `int32` `Tensor`.
        """
        # get inputs
        start, limit, delta = inputs

        # get range
        try:
            range_val = np.arange(start.const, limit.const, delta.const)
        except:
            raise NotImplementedError("[NON-NATIVE] Input to `Range` must all "
                                      "be integer, dynamic allocation is not "
                                      "supported.")

        # return
        return ng.constant(range_val, shape_to_axes(range_val.shape),
                           name=tf_node.name)

    def Size(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Returns the size of a tensor.

        This operation returns an integer representing the number of elements in
        `input`.

        For example:

        ```prettyprint
        # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
        size(t) ==> 12
        ```

        Args:
            input: A `Tensor`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor` of type `int32`.

        TODO: Test coverage does not reach here yet
        """
        # get inputs
        left = inputs[0]

        # get rank
        try:
            size = np.prod(left.axes.lengths)
        except:
            raise NotImplementedError("[NON-NATIVE] `Size` op's axes must be "
                                      "pre-determined before execution.")
        # return
        return ng.constant(size, ng.make_axes([]), name=tf_node.name)

    def Cast(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Casts a tensor to a new type.

        The operation casts `x` (in case of `Tensor`) or `x.values`
        (in case of `SparseTensor`) to `dtype`.

        For example:

        ```python
        # tensor `a` is [1.8, 2.2], dtype=tf.float
        tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
        ```

        Args:
            x: A `Tensor` or `SparseTensor`.
            dtype: The destination type.
            name: A name for the operation (optional).

        Returns:
            A `Tensor` or `SparseTensor` with same shape as `x`.

        Raises:
            TypeError: If `x` cannot be cast to the `dtype`.
        """
        # TODO: now only a pass through
        # get src and dst datatypes
        # dst_type = tf_node.attr['DstT']
        # src_type = tf_node.attr['SrcT']
        return inputs[0]

    def Shape(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Returns the shape of a tensor.

        This operation returns a 1-D integer tensor representing the shape of `input`.

        For example:

        ```prettyprint
        # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
        shape(t) ==> [2, 2, 3]
        ```

        Args:
            input: A `Tensor`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor` of type `int32`.
        """

        # get inputs
        left = inputs[0]

        # get shape
        try:
            shape = left.axes.lengths
        except:
            raise NotImplementedError("[NON-NATIVE] `Size` op's axes must be "
                                      "pre-determined before execution.")
        axes = ng.make_axes([ng.make_axis(len(left.axes.lengths)), ])

        # return
        return ng.constant(shape, axes, name=tf_node.name)

    def Reshape(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Reshapes a tensor.

        Given `tensor`, this operation returns a tensor that has the same values
        as `tensor` with shape `shape`.

        If one component of `shape` is the special value -1, the size of that dimension
        is computed so that the total size remains constant. In particular, a `shape`
        of `[-1]` flattens into 1-D. At most one component of `shape` can be -1.

        If `shape` is 1-D or higher, then the operation returns a tensor with shape
        `shape` filled with the values of `tensor`. In this case, the number of elements
        implied by `shape` must be the same as the number of elements in `tensor`.

        For example:

        ```prettyprint
        # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
        # tensor 't' has shape [9]
        reshape(t, [3, 3]) ==> [[1, 2, 3],
                              [4, 5, 6],
                              [7, 8, 9]]

        # tensor 't' is [[[1, 1], [2, 2]],
        #                [[3, 3], [4, 4]]]
        # tensor 't' has shape [2, 2, 2]
        reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                              [3, 3, 4, 4]]

        # tensor 't' is [[[1, 1, 1],
        #                 [2, 2, 2]],
        #                [[3, 3, 3],
        #                 [4, 4, 4]],
        #                [[5, 5, 5],
        #                 [6, 6, 6]]]
        # tensor 't' has shape [3, 2, 3]
        # pass '[-1]' to flatten 't'
        reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

        # -1 can also be used to infer the shape

        # -1 is inferred to be 9:
        reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [4, 4, 4, 5, 5, 5, 6, 6, 6]]
        # -1 is inferred to be 2:
        reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                               [4, 4, 4, 5, 5, 5, 6, 6, 6]]
        # -1 is inferred to be 3:
        reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                    [2, 2, 2],
                                    [3, 3, 3]],
                                   [[4, 4, 4],
                                    [5, 5, 5],
                                    [6, 6, 6]]]

        # tensor 't' is [7]
        # shape `[]` reshapes to a scalar
        reshape(t, []) ==> 7
        ```

        Args:
            tensor: A `Tensor`.
            shape: A `Tensor` of type `int32`. Defines the shape of the output tensor.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `tensor`.
        """
        # TODO: currently only support constants, reshape is not in ngraph
        # get inputs
        tensor, shape = inputs

        try:
            # new tensor
            np_val = np.reshape(tensor.const, shape.const.astype(int))
            return ng.constant(np_val, shape_to_axes(np_val.shape),
                               name=tf_node.name)
        except:
            raise NotImplementedError("Reshape not supported in ngraph, "
                                      "currently only const tensor is supported.")

    def Tile(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Constructs a tensor by tiling a given tensor.

        This operation creates a new tensor by replicating `input` `multiples` times.
        The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
        and the values of `input` are replicated `multiples[i]` times along the 'i'th
        dimension. For example, tiling `[a b c d]` by `[2]` produces
        `[a b c d a b c d]`.

        Args:
            input: A `Tensor`. 1-D or higher.
            multiples: A `Tensor` of type `int32`.
                       1-D. Length must be the same as the number of dimensions in `input`
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `input`.
        """
        tensor, multiples = inputs

        # get inputs
        try:
            input_val = tensor.const
            multiples_val = multiples.const
        except:
            raise NotImplementedError("Tile not supported in ngraph, "
                                      "currently only const tensor is supported.")

        # check shapes
        input_shape = input_val.shape
        input_ndims = len(input_shape)
        assert input_ndims >= 1 and input_ndims == len(multiples_val)

        output_val = np.tile(input_val, multiples_val.astype(int))

        # make new constants
        return ng.constant(output_val, shape_to_axes(output_val.shape),
                           name=tf_node.name)

    def ExpandDims(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Inserts a dimension of 1 into a tensor's shape.

        Given a tensor `input`, this operation inserts a dimension of 1 at the
        dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
        zero; if you specify a negative number for `dim` it is counted backward from
        the end.

        This operation is useful if you want to add a batch dimension to a single
        element. For example, if you have a single image of shape `[height, width,
        channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
        which will make the shape `[1, height, width, channels]`.

        Other examples:

        ```prettyprint
        # 't' is a tensor of shape [2]
        shape(expand_dims(t, 0)) ==> [1, 2]
        shape(expand_dims(t, 1)) ==> [2, 1]
        shape(expand_dims(t, -1)) ==> [2, 1]

        # 't2' is a tensor of shape [2, 3, 5]
        shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
        shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
        shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
        ```

        This operation requires that:

        `-1-input.dims() <= dim <= input.dims()`

        This operation is related to `squeeze()`, which removes dimensions of
        size 1.

        Args:
            input: A `Tensor`.
            dim: A `Tensor` of type `int32`.
                 0-D (scalar). Specifies the dimension index at which to
                 expand the shape of `input`.
            name: A name for the operation (optional).

        Returns:
            A `Tensor`. Has the same type as `input`.
            Contains the same data as `input`, but its shape has an additional
            dimension of size 1 added.
        """
        # get input
        tensor, dim = inputs[0], int(inputs[1].const)

        # check `-1-input.dims() <= dim <= input.dims()`
        input_ndims = len(tensor.axes.lengths)
        assert -1 - input_ndims <= dim <= input_ndims

        # deal with negative number
        if dim < 0:
            dim = input_ndims + 1 + dim

        # create new axis
        one_axis = ng.make_axis(length=1)

        # get output axis
        pre_axis = [axis for axis in tensor.axes[:dim]]  # avoid FlattenedAxis
        pos_axis = [axis for axis in tensor.axes[dim:]]  # avoid FlattenedAxis
        out_axis = ng.make_axes(pre_axis + [one_axis] + pos_axis)

        # broadcast
        return ng.broadcast(tensor, out_axis)
