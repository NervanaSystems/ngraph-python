# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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
from __future__ import division

import ngraph as ng
from ngraph.frontends.common.utils import remove_ones_axes
from ngraph.frontends.neon.layer import Dropout


class OpsCompound:
    """
    Bridging compoud operations between CNTK and ngraph.
    """

    def Dense(self, cntk_op, inputs):
        """
        Computes fully-connected layer with optional activation function.

        Arguments:
            cntk_op: CNTK block to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return self._block_import(cntk_op, inputs)

    def _block_import(self, cntk_op, inputs):
        """
        Imports operations from a block layer.

        Arguments:
            cntk_op: CNTK block to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return_op = cntk_op.block_root.uid

        block_ops = []
        stack = [cntk_op.block_root]
        while stack:
            node = stack.pop()
            node = node.root_function

            if node in block_ops:
                continue
            else:
                block_ops.append(node)

            for i in node.inputs:
                if i.is_output:
                    stack.append(i.owner)

        imported_ops = dict()
        while block_ops:
            node = block_ops.pop()
            node_inputs = []
            for i in node.inputs:
                if i.is_placeholder:
                    try:
                        temp = next(iter([
                            v for v in inputs if not isinstance(v, ng.AssignableTensorOp)
                        ]))
                    except StopIteration:
                        temp = next(iter([
                            v for v in inputs if v.is_placeholder
                        ]))
                elif i.is_output:
                    temp = imported_ops.get(i.owner.root_function.uid)
                else:
                    temp = next(iter([
                        v for v in inputs if v.name == i.uid
                    ]))

                if temp is not None:
                    node_inputs.append(temp)
                else:
                    raise ValueError("Unknown input: " + i.uid)
            try:
                imported_ops[node.uid] = getattr(self, node.op_name)(node, node_inputs)
            except AttributeError:
                raise TypeError("Unknown operation: " + node.op_name)

        return imported_ops[return_op]

    def _expand_input_axes(self, inputs):
        """
        Expand 1D or 2D input into 3D input.

        Arguments:
            axes: Convolution input's axes.

        Returns:
            Expanded list of input's axes.
        """
        axes = inputs.axes
        dim = len(axes)
        batch = axes.batch_axis()

        if dim == 5:
            C, D, H, W, N = axes
        elif dim == 4:
            if batch:
                C, H, W, N = axes
                D = ng.make_axis(1)
            else:
                C, D, H, W = axes
                N = ng.make_axis(1, 'N')
        elif dim == 3:
            if batch:
                H, W, N = axes
                C = ng.make_axis(1)
                D = ng.make_axis(1)
            else:
                C, H, W = axes
                D = ng.make_axis(1)
                N = ng.make_axis(1, 'N')
        elif dim == 2:
            if batch:
                H, N = axes
                C = ng.make_axis(1)
                D = ng.make_axis(1)
                W = ng.make_axis(1)
            else:
                H, W = axes
                C = ng.make_axis(1)
                D = ng.make_axis(1)
                N = ng.make_axis(1, 'N')
        else:
            raise ValueError("Convolution input must have 2 to 5 axes.")

        return ng.broadcast(inputs, [C, D, H, W, N])

    def _expand_filters_axes(self, filters, C):
        """
        Expand and cast 1D or 2D filter into 3D filter.

        Arguments:
            axes: Convolution filter's axes.

        Returns:
            Expanded list of filter's axes.
        """
        axes = filters.axes
        dim = len(axes)
        if dim == 5:
            O, _, T, M1, M2 = axes
            filters = ng.cast_axes(filters, [O, C, T, M1, M2])
        elif dim == 4:
            O, _, M1, M2 = axes
            filters = ng.cast_axes(filters, [O, C, M1, M2])
            T = ng.make_axis(1)
        elif dim == 3:
            O, M1, M2 = axes
            T = ng.make_axis(1)
        elif dim == 2:
            O, M1 = axes
            T = ng.make_axis(1)
            M2 = ng.make_axis(1)
        elif dim == 1:
            O = axes
            T = ng.make_axis(1)
            M1 = ng.make_axis(1)
            M2 = ng.make_axis(1)
        else:
            raise ValueError("Convolution filter must have 1 to 5 axes.")

        return ng.broadcast(filters, [C, T, M1, M2, O])

    def _make_out_axes(self, shape):
        """
        Make output convolution axes.

        Arguments:
            shape: CNTK convolution output shape.

        Returns:
            List of dynamic output axes.
        """
        dim = len(shape)
        if dim == 4:
            M = ng.make_axis(shape[1])
            oH = ng.make_axis(shape[2])
            oW = ng.make_axis(shape[3])
        elif dim == 3:
            M = ng.make_axis(1)
            oH = ng.make_axis(shape[1])
            oW = ng.make_axis(shape[2])
        elif dim == 2:
            M = ng.make_axis(1)
            oH = ng.make_axis(shape[1])
            oW = ng.make_axis(1)
        elif dim == 1:
            M = ng.make_axis(1)
            oH = ng.make_axis(1)
            oW = ng.make_axis(1)

        return [M, oH, oW]

    def _make_strides(self, strides):
        """
        Make strides vector out of CNTK convolution's strides.

        Arguments:
            shape: CNTK convolution strides shape.

        Returns:
            List of strides (D, H, W, C).
        """
        dim = len(strides)
        if dim == 4:
            return (strides[1], strides[2], strides[3], strides[0])
        elif dim == 3:
            return (1, strides[1], strides[2], strides[0])
        elif dim == 2:
            return (1, strides[0], strides[1], 1)
        elif dim == 1:
            return (1, strides[0], 1, 1)

    def _make_kernel(self, window):
        """
        Make pool kernel shape.

        Arguments:
            window: CNTK pooling window shape.

        Returns:
            Kernel shape (T, M1, M2, C).
        """
        dim = len(window)
        if dim == 4:
            return (window[3], window[1], window[2], window[0])
        elif dim == 3:
            return (1, window[1], window[2], window[0])
        elif dim == 2:
            return (1, window[0], window[1], 1)
        elif dim == 1:
            return (1, window[0], 1, 1)

    def _make_padding(self, op, padding, input_shape, kernel_shape, output_shape, strides):
        """
        Make padding vector out of CNTK convolution's auto padding values.

        Arguments:
            op: 'conv' or 'pool'
            padding: CNTK convolution auto padding tuple.
            input_shape: Input shape (D, H, W, C).
            kernel_shape: Kernel shape (T, M1, M2, C).
            output_shape: Output shape (M, oH, oW, O).
            strides: Strides values (D, H, W, C).

        Returns:
            List of padding values.
        """
        dim = len(padding)
        if dim == 4:
            padding = (padding[3], padding[1], padding[2], padding[0])
        elif dim == 3:
            padding = (False, padding[1], padding[2], padding[0])
        elif dim == 2:
            padding = (False, padding[0], padding[1], False)
        elif dim == 1:
            padding = (False, padding[0], False, False)

        pad = [0, 0, 0, 0]
        for i, p in enumerate(padding):
            if p is True:
                pad_axis = ((output_shape[i] - 1) * strides[i]) - input_shape[i] + kernel_shape[i]
                pad_side = pad_axis // 2
                if op == 'conv':
                    if pad_side * 2 == pad_axis:
                        pad[i] = pad_side
                    else:
                        pad[i] = pad_side + 1
                else:
                    pad[i] = pad_side

        return pad

    def _convolution_op(self, cntk_op, inputs):
        """
        Computes the convolution of a tensor with operand.
                      CNTK            Ngraph
        in       ((C, H, W) N))   (C, D, H, W, N)
        filter   (O, C, M1, M2)   (C, T, M1, M2, O)
        out      ((O, H, W) N)    (O, M, H, W, N)

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        filters, inputs = inputs

        inputs = self._expand_input_axes(inputs)
        C, D, H, W, N = inputs.axes

        filters = self._expand_filters_axes(filters, C)
        _, T, M1, M2, O = filters.axes

        M, oH, oW = self._make_out_axes(cntk_op.shape)
        out_axes = [O, M, oH, oW, N]

        strides = self._make_strides(cntk_op.attributes['strides'])
        pad = self._make_padding(
            'conv', cntk_op.attributes['autoPadding'],
            (D.length, H.length, W.length, C.length),
            (T.length, M1.length, M2.length, C.length),
            (M.length, oH.length, oW.length, O.length),
            strides
        )

        params = dict(
            pad_d=pad[0], pad_h=pad[1], pad_w=pad[2], pad_c=pad[3],
            str_d=strides[0], str_h=strides[1], str_w=strides[2], str_c=strides[3],
            dil_d=1, dil_h=1, dil_w=1
        )

        conv = ng.convolution(params, inputs, filters, out_axes)
        return remove_ones_axes([conv])[0]

    def Convolution(self, cntk_op, inputs):
        """
        Imports the Convolution block or operation.

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        if cntk_op.is_block:
            ret_op = self._block_import(cntk_op, inputs)
        else:
            ret_op = self._convolution_op(cntk_op, inputs)
        return ret_op.named(cntk_op.uid)

    def _pooling_op(self, cntk_op, inputs, op):
        """
        Computes the pooling of a tensor.
                    CNTK             Ngraph
        in     ((C, H, W), N)   (C, D, H, W, N)
        out    (N, P, Q, K)     (K, M, P, Q, N)

        Arguments:
            cntk_op: CNTK function to be imported.
            inputs: List of inputs to this node.
            op: 'max' for MaxPooling and 'avg' for AvgPooling

        Returns:
            A ngraph Op.
        """
        inputs = self._expand_input_axes(inputs[0])
        C, D, H, W, N = inputs.axes

        M, P, Q = self._make_out_axes(cntk_op.shape)

        strides = self._make_strides(cntk_op.attributes['strides'])
        kernel = self._make_kernel(cntk_op.attributes['poolingWindowShape'])
        pad = self._make_padding(
            'pool', cntk_op.attributes['autoPadding'],
            (D.length, H.length, W.length, C.length),
            kernel,
            (M.length, P.length, Q.length, C.length),
            strides
        )

        params = dict(
            op=op,
            pad_d=pad[0], pad_h=pad[1], pad_w=pad[2], pad_c=pad[3],
            str_d=strides[0], str_h=strides[1], str_w=strides[2], str_c=strides[3],
            T=kernel[0], R=kernel[1], S=kernel[2], J=kernel[3]
        )

        pool = ng.pooling(params, inputs, [C, M, P, Q, N])
        return remove_ones_axes([pool])[0]

    def MaxPooling(self, cntk_op, inputs):
        """
        Computes the max pooling of a tensor.

        Arguments:
            cntk_op: CNTK block to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return self._pooling_op(cntk_op.block_root.root_function, inputs, 'max')

    def AveragePooling(self, cntk_op, inputs):
        """
        Computes the average pooling of a tensor.

        Arguments:
            cntk_op: CNTK block to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return self._pooling_op(cntk_op.block_root.root_function, inputs, 'avg')

    def CrossEntropyWithSoftmax(self, cntk_op, inputs):
        """
        Computes the softmax cross entropy between the inputs[0] and inputs[1].

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        cast_0, cast_1 = remove_ones_axes(inputs)

        if cast_0.axes.lengths != cast_1.axes.lengths:
            cast_0 = ng.Transpose(cast_0)
        assert cast_0.axes.lengths == cast_1.axes.lengths

        cast_0 = ng.cast_axes(cast_0, axes=cast_1.axes)
        loss = ng.cross_entropy_multi(ng.softmax(cast_0), cast_1)

        return ng.mean(loss, out_axes=()).named(cntk_op.uid)

    def Combine(self, cntk_op, inputs):
        """
        Returns combined outputs of inputs list.

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        return ng.stack(inputs, ng.make_axis(len(inputs))).named(cntk_op.uid)

    def Dropout(self, cntk_op, inputs):
        """
        Stochastically dropping activations to prevent overfitting

        Arguments:
            cntk_op: CNTK operation to be imported.
            inputs: List of inputs to this node.

        Returns:
            A ngraph Op.
        """
        node = cntk_op.block_root.root_function
        layer = Dropout(node.attributes['dropoutRate'])
        return layer(inputs[0]).named(cntk_op.uid)
