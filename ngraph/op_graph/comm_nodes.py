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
from __future__ import division
import collections
from ngraph.op_graph.op_graph import TensorOp, compute_reduction_axes, \
    MutateInsteadOfCopyWithNewArgsMixin
from ngraph.op_graph.axes import Axes, make_axes, make_axis
import logging


logger = logging.getLogger(__name__)


def calculate_gather_axes(axes, gather_axis, num_devices):
    new_axes = [make_axis(a.length * num_devices, a.name)
                if gather_axis == a else a for a in axes]
    new_axes = make_axes(new_axes)
    return new_axes


def set_parallel_axes(axes, parallel_axis):
    new_axes = []
    flat_names = dict()
    for i, axis in enumerate(Axes.as_nested_list(axes)):
        if axis == parallel_axis:
            axis = parallel_axis
        elif isinstance(axis, collections.Iterable):
            flat_names[i] = axes[i].name
            axis = [parallel_axis if a == parallel_axis else a for a in axis]
        new_axes.append(axis)
    new_axes = make_axes(new_axes)

    for i in flat_names:
        new_axes[i].name = flat_names[i]
    return new_axes


def get_slices(axes, parallel_axis, num_devices):
    new_slices = list()
    for i in range(num_devices):
        slices = list()
        for a in axes:
            s = slice(None)
            if parallel_axis == a:
                remainder = a.length % num_devices
                new_length = a.length // num_devices
                start = i * new_length
                stop = (i + 1) * new_length
                step = 1
                if remainder > 0:
                    if i == (num_devices - 1):
                        stop += remainder
                s = slice(start, stop, step)
            slices.append(s)
        new_slices.append(slices)
    return new_slices


class ResultOp(TensorOp):
    """
    special op for Hetr distributed case, not supported by other transformers
    note: possible deprecation in future (issue #1115)
    """

    def __init__(self, device_id, args, **kwargs):

        super(ResultOp, self).__init__(args=args)
        self.metadata['device_id'] = device_id
        self.axes = args[0].axes
        self.dtype = args[0].dtype


class CommunicationOp(TensorOp):
    """
    Represents a communication op.

    Arguments:
        None
    """

    def __init__(self, node, args=None, axes=None, dtype=None):
        super(CommunicationOp, self).__init__(args=args, axes=axes, dtype=dtype)
        self.metadata['device'] = node.metadata['device']
        self.metadata['device_id'] = node.metadata['device_id']
        self.metadata['transformer'] = node.metadata['transformer']
        self.metadata['host_transformer'] = node.metadata['host_transformer']

    @property
    def is_communication_op(self):
        return True

    @property
    def has_side_effects(self):
        return True

    @property
    def is_persistent(self):
        return True

    def hetr_axes(self, axes, parallel_axis):
        """
        Function to ensure that HeTr axes are ordered such that parallel_axis
        is least contiguous.
        """
        if parallel_axis is None:
            hetr_axes = axes
        elif parallel_axis not in axes:
            hetr_axes = axes
        else:
            hetr_axes = parallel_axis + (axes - parallel_axis)
        return hetr_axes


class SendOp(CommunicationOp):
    """
    Represents a send op. Sets args, axes, dtype, and metadata.

    Arguments:
        from_node: The source node.
    """

    def __init__(self, from_node, parallel_axis=None):
        super(SendOp, self).__init__(
            node=from_node,
            args=tuple([from_node]),
            axes=self.hetr_axes(from_node.axes, parallel_axis),
            dtype=from_node.dtype)

        # Add native/original axes to op
        # Also ensure that the native axes has the original length
        # of the parallel_axis
        self.native_axes = Axes.as_flattened_list(from_node.axes)
        if parallel_axis is not None and parallel_axis in self.native_axes:
            p_axis_idx = self.native_axes.index(parallel_axis)
            self.native_axes[p_axis_idx] = parallel_axis
        self.native_axes = make_axes(self.native_axes)


class RecvOp(CommunicationOp):
    """
    Represents a recv op. Sets args, axes, dtype, and metadata.

    Arguments:
        to_node: The destination node.
        send_node: The send node associated with this recv node.
    """

    def __init__(self, to_node, send_node, parallel_axis=None, num_devices=None):
        super(RecvOp, self).__init__(
            node=to_node,
            args=(),
            axes=self.hetr_axes(send_node.axes, parallel_axis),
            dtype=send_node.dtype)
        self._send_node = send_node
        self.native_axes = send_node.native_axes
        self.source_id = send_node.metadata['device_id']

    def send_node(self):
        return self._send_node


class ScatterSendOp(SendOp):
    """
    Represents a scatter send op. Sets destination device ids and slices.

    Arguments:
        from_node: The source node.
        to_node: The destination node.
    """

    def __init__(self, from_node, to_node):
        super(ScatterSendOp, self).__init__(from_node, to_node.metadata['parallel'])
        self.to_id = to_node.metadata['device_id']
        self._slices = get_slices(self.axes,
                                  to_node.metadata['parallel'],
                                  len(self.to_id))
        assert to_node.metadata.get('parallel', None) is not None, \
            "to_node must have a specified parallel attribute in metadata"
        self.metadata['parallel'] = to_node.metadata['parallel']

    @property
    def slices(self):
        return self._slices


class ScatterRecvOp(RecvOp):
    """
    Represents a scatter recv op.

    Arguments:
        to_node: The destination node.
        send_node: The scatter send node associated with this node.
    """

    def __init__(self, to_node, send_node):
        super(ScatterRecvOp, self).__init__(to_node, send_node,
                                            parallel_axis=to_node.metadata['parallel'],
                                            num_devices=len(to_node.metadata['device_id']))
        assert to_node.metadata.get('parallel', None) is not None, \
            "to_node must have a specified parallel attribute in metadata"
        self.metadata['parallel'] = to_node.metadata['parallel']

    def copy_with_new_args(self, args):
        """
        Overriding parent function since this op's args can get
        replaced with mklreorder ops during HeTrTensorShaping pass.
        Also, we don't inherit from MutateInsteadOfCopy
        because it adds extra parameters to init function.
        """
        self._set_args(args)
        return self


class GatherSendOp(SendOp):
    """
    Represents a gather send op.

    Arguments:
        from_node: The source node.
    """

    def __init__(self, from_node):
        super(GatherSendOp, self).__init__(from_node, from_node.metadata['parallel'])
        assert from_node.metadata.get('parallel', None) is not None, \
            "from_node must have a specified parallel attribute in metadata"
        self.metadata['parallel'] = from_node.metadata['parallel']

        # todo: replace by real reduce operation
        self.metadata['parallel'] = from_node.metadata['parallel']
        self.use_reduce = False


class GatherRecvOp(RecvOp):
    """
    Represents a gather recv op. Sets metadata, source device ids and
    slices.

    Arguments:
        from_node: The source node.
        to_node: The destination node.
        send_node: The gather send node associated with this node.
    """

    def __init__(self, from_node, to_node, send_node):
        super(GatherRecvOp, self).__init__(to_node, send_node,
                                           parallel_axis=from_node.metadata['parallel'],
                                           num_devices=len(from_node.metadata['device_id']))
        self.metadata['marker'] = 'gather'
        assert from_node.metadata.get('parallel', None) is not None, \
            "from_node must have a specified parallel attribute in metadata"
        self.metadata['parallel'] = from_node.metadata['parallel']
        self.from_id = from_node.metadata['device_id']
        # use _slices to avoid serialization
        self._slices = get_slices(self.axes,
                                  self.metadata['parallel'],
                                  len(self.from_id))

        # todo: replace by real reduce operation
        self.use_reduce = False
        parallel_axis = self.metadata['parallel']
        if parallel_axis == send_node.metadata['parallel'] and \
           self.axes.find_by_name(parallel_axis.name) != parallel_axis.axes:
            self.use_reduce = True
            send_node.use_reduce = True

    def hetr_axes(self, axes, parallel_axis):
        """
        Override hetr_axes function to ensure GatherRecvOp has the full length
        of the parallel_axis rather that parallel_axis.length//num_devices.
        """
        arg_axes = super(GatherRecvOp, self).hetr_axes(axes, parallel_axis)
        if parallel_axis in axes and \
           arg_axes.find_by_name(parallel_axis.name).lengths[0] != parallel_axis.length:
            arg_axes = make_axes([parallel_axis if a == parallel_axis else a for a in arg_axes])
        return arg_axes

    def copy_with_new_args(self, args):
        """
        Overriding parent function since this op's args can get
        replaced with mklreorder ops during HeTrTensorShaping pass.
        Also, we don't inherit from MutateInsteadOfCopy
        because it adds extra parameters to init function.
        """
        self._set_args(args)
        return self

    @property
    def slices(self):
        return self._slices


class GPUCudaSendOp(MutateInsteadOfCopyWithNewArgsMixin, SendOp):

    def __init__(self, from_node, to_node):
        super(GPUCudaSendOp, self).__init__(from_node=from_node)
        self.dest_id = to_node.metadata['device_id']


class GPUCudaRecvOp(RecvOp):

    def __init__(self, to_node, send_node):
        super(GPUCudaRecvOp, self).__init__(to_node, send_node)
        self.source_id = send_node.metadata['device_id']


class GPUCudaScatterSendOp(MutateInsteadOfCopyWithNewArgsMixin, ScatterSendOp):

    def __init__(self, from_node, to_node):
        super(GPUCudaScatterSendOp, self).__init__(from_node=from_node, to_node=to_node)
        self.metadata['parallel'] = to_node.metadata['parallel']


class GPUCudaScatterRecvOp(ScatterRecvOp):

    def __init__(self, to_node, send_node):
        super(GPUCudaScatterRecvOp, self).__init__(to_node, send_node)
        self.idx = 0


class GPUCudaGatherSendOp(MutateInsteadOfCopyWithNewArgsMixin, GatherSendOp):

    def __init__(self, from_node):
        super(GPUCudaGatherSendOp, self).__init__(from_node=from_node)
        self.idx = 0
        self.metadata['parallel'] = from_node.metadata['parallel']


class GPUCudaGatherRecvOp(GatherRecvOp):

    def __init__(self, from_node, to_node, send_node):
        super(GPUCudaGatherRecvOp, self).__init__(from_node, to_node, send_node)


class AllReduceOp(CommunicationOp):
    """
    Represents an AllReduce op. Sets reduction axes and out axes.
    TODO: revisit the need for reduction_axes and out_axes in HeTr.

    Arguments:
        x: The input node.
        reduction_axes: The reduction axes.
        out_axes: The output axes.
        dtype: The data type.
    """
    def __init__(self, x, reduction_axes=None, out_axes=None, dtype=None, func=None, **kwargs):
        reduction_axes, out_axes = compute_reduction_axes(x, reduction_axes, out_axes)
        self.reduction_axes = reduction_axes
        super(AllReduceOp, self).__init__(node=x, args=(x,), axes=out_axes, dtype=dtype, **kwargs)
        self.reduce_func = func
        if (self.reduce_func == 'mean' or self.reduce_func == 'sum') is False:
            raise RuntimeError(
                'Reduce function {} is not supported!'.format(self.reduce_func))


class GPUCudaAllReduceOp(MutateInsteadOfCopyWithNewArgsMixin, AllReduceOp):
    """
    Represents GPU implementation for AllReduce op. Sets reduction function and creates
    shared queues.

    Arguments:
        x: The input node.
        func: The reduction function, e.g. 'sum', 'mean'.
    """
    def __init__(self, input_node, func=None):
        super(GPUCudaAllReduceOp, self).__init__(x=input_node,
                                                 out_axes=input_node.axes,
                                                 dtype=input_node.dtype,
                                                 func=func)
        self.idx = 0
        self.device_ids = input_node.metadata['device_id']


class BroadcastSendOp(SendOp):
    """
    Represents a broadcat send op. Sets destination device ids.

    Arguments:
        from_node: The source node.
        to_node: The destination node.
    """

    def __init__(self, from_node, to_node):
        super(BroadcastSendOp, self).__init__(from_node)
        self.to_id = to_node.metadata['device_id']


class BroadcastRecvOp(RecvOp):
    """
    Represents a broadcast recv op.

    Arguments:
        to_node: The destination node.
        send_node: The broadcast send node associated with this node.
    """

    def __init__(self, to_node, send_node):
        super(BroadcastRecvOp, self).__init__(to_node, send_node)


class CPUMlslSendOp(MutateInsteadOfCopyWithNewArgsMixin, SendOp):

    def __init__(self, from_node):
        super(CPUMlslSendOp, self).__init__(from_node=from_node)


class CPUMlslRecvOp(RecvOp):

    def __init__(self, to_node, send_node):
        super(CPUMlslRecvOp, self).__init__(to_node, send_node)


class CPUMlslScatterSendOp(MutateInsteadOfCopyWithNewArgsMixin, ScatterSendOp):

    def __init__(self, from_node, to_node):
        super(CPUMlslScatterSendOp, self).__init__(from_node=from_node, to_node=to_node)
        self.arr = None


class CPUMlslScatterRecvOp(ScatterRecvOp):

    def __init__(self, to_node, send_node):
        super(CPUMlslScatterRecvOp, self).__init__(to_node, send_node)


class CPUMlslGatherSendOp(MutateInsteadOfCopyWithNewArgsMixin, GatherSendOp):

    def __init__(self, from_node):
        super(CPUMlslGatherSendOp, self).__init__(from_node=from_node)
        self.arr = None


class CPUMlslGatherRecvOp(GatherRecvOp):

    def __init__(self, from_node, to_node, send_node):
        super(CPUMlslGatherRecvOp, self).__init__(from_node, to_node, send_node)


class CPUMlslAllReduceStartOp(MutateInsteadOfCopyWithNewArgsMixin, AllReduceOp):
    """
    Represents CPU-based implementation for AllReduce op over async MLSL::AllReduce.
    Start async communication.

    Arguments:
        x: The input node.
        func: The reduction function, e.g. 'sum', 'mean'.
    """
    def __init__(self, input_node, func=None):
        super(CPUMlslAllReduceStartOp, self).__init__(x=input_node,
                                                      out_axes=input_node.axes,
                                                      dtype=input_node.dtype,
                                                      func=func)
        self._req = [None]  # use mutable field to share it between start and wait ops
        self.metadata['priority'] = 'high'

    @property
    def req(self):
        return self._req[0]

    @req.setter
    def req(self, value):
        self._req[0] = value


class CPUMlslAllReduceWaitOp(MutateInsteadOfCopyWithNewArgsMixin, AllReduceOp):
    """
    Represents CPU-based implementation for AllReduce op over async MLSL::AllReduce.
    Complete async communication.

    Arguments:
        x: The input node.
        func: The reduction function, e.g. 'sum', 'mean'.
    """
    def __init__(self, input_node, start_node, func=None):
        super(CPUMlslAllReduceWaitOp, self).__init__(x=input_node,
                                                     out_axes=input_node.axes,
                                                     dtype=input_node.dtype,
                                                     func=func)
        self.add_control_dep(start_node)
        self._req = start_node._req
        self.metadata['priority'] = 'low'

    @property
    def req(self):
        return self._req[0]

    @req.setter
    def req(self, value):
        self._req[0] = value


class CPUMlslBroadcastSendOp(BroadcastSendOp):
    """
    Represents CPU-based MLSL implementation for BroadcastSend op over MLSL::Bcast
    """
    def __init__(self, from_node, to_node):
        super(CPUMlslBroadcastSendOp, self).__init__(from_node, to_node)
        self.arr = None


class CPUMlslBroadcastRecvOp(BroadcastRecvOp):
    """
    Represents CPU-based queue implementation for BroadcastRecv op over MLSL::Bcast.
    """
    def __init__(self, to_node, send_node):
        super(CPUMlslBroadcastRecvOp, self).__init__(to_node, send_node)


class GatherWrapperOp(CommunicationOp):

    def __init__(self, recv_node, arg_op):
        super(GatherWrapperOp, self).__init__(recv_node, args=tuple([arg_op]),
                                              axes=arg_op.native_axes
                                              if isinstance(arg_op, (SendOp, RecvOp)) else
                                              arg_op.axes)

    def copy_with_new_args(self, args):
        """
        Overriding parent function since this op's args can get
        replaced with mklreorder ops during HeTrTensorShaping pass.
        Also, we don't inherit from MutateInsteadOfCopy
        because it adds extra parameters to init function.
        """
        self._set_args(args)
        return self
