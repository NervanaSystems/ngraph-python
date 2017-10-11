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

import numpy as np
import mlsl
from mpi4py import MPI
import ctypes
from ngraph.op_graph.comm_nodes import \
    CPUMlslGatherSendOp, CPUMlslScatterSendOp, \
    CPUMlslAllReduceStartOp, CPUMlslBroadcastSendOp
import logging


logger = logging.getLogger(__name__)
USER_TAG = 1


class HetrLocals(object):

    mlsl_obj = mlsl.MLSL()
    mlsl_obj.init()
    process_count = mlsl_obj.get_process_count()
    process_idx = mlsl_obj.get_process_idx()

    def __init__(self, send_nodes, recv_nodes,
                 scatter_send_nodes, scatter_recv_nodes,
                 gather_send_nodes, gather_recv_nodes,
                 allreduce_nodes, broadcast_send_nodes,
                 broadcast_recv_nodes, **kwargs):
        super(HetrLocals, self).__init__(**kwargs)
        self.send_nodes = send_nodes
        self.recv_nodes = recv_nodes
        self.scatter_send_nodes = scatter_send_nodes
        self.scatter_recv_nodes = scatter_recv_nodes
        self.gather_send_nodes = gather_send_nodes
        self.gather_recv_nodes = gather_recv_nodes
        self.allreduce_nodes = allreduce_nodes
        self.broadcast_send_nodes = broadcast_send_nodes
        self.broadcast_recv_nodes = broadcast_recv_nodes

        # MLSL-specific
        self.mlsl_obj = mlsl.MLSL()
        self.mlsl_obj.init()
        self.process_count = self.mlsl_obj.get_process_count()
        self.process_idx = self.mlsl_obj.get_process_idx()
        # data parallelism
        self.distribution = None

        # MPI-specific
        self.comm = MPI.COMM_WORLD

    def create_distribution(self):
        if not self.distribution:
            self.distribution = self.mlsl_obj.create_distribution(self.process_count, 1)

    def close(self):
        if self.distribution:
            self.mlsl_obj.delete_distribution(self.distribution)
        self.mlsl_obj.finalize()

    @staticmethod
    def close_mlsl():
        HetrLocals.mlsl_obj.finalize()
        mlsl.close()

    @staticmethod
    def mlsl_alloc(element_count, alignment, dtype):
        if dtype.name == 'float32':
            c_type_name = 'c_float'
        elif dtype.name == 'float64':
            c_type_name = 'c_double'
        else:
            c_type_name = None
        type_size = ctypes.sizeof(getattr(ctypes, c_type_name)(1))
        mlsl_buf = HetrLocals.mlsl_obj.alloc(element_count * type_size, alignment)
        array = ctypes.cast(mlsl_buf, ctypes.POINTER(getattr(ctypes, c_type_name) * element_count))
        np_array = np.frombuffer(array.contents, dtype)
        return np_array

    @staticmethod
    def mlsl_free(array):
        HetrLocals.mlsl_obj.free(array.__array_interface__['data'][0])

    def mlsl_send(self, send_id, x_nparr):
        send_op = self.send_nodes[send_id]
        self.comm.Send(x_nparr, dest=send_op.metadata['peer_id'], tag=USER_TAG)

    def recv_from_mlsl_send(self, recv_id, out):
        recv_op = self.recv_nodes[recv_id]
        self.comm.Recv(out, source=recv_op.metadata['peer_id'], tag=USER_TAG)
        return out

    def mlsl_gather_send(self, gather_send_id, x_nparr):
        gather_send_op = self.gather_send_nodes[gather_send_id]

        # todo: get real root_idx
        root_idx = 0

        x_nparr = np.atleast_1d(x_nparr)     # np.atleast_1d is used in cases when we need to reduce to a scalar value
        if self.process_idx == root_idx:
            # todo: remove that workaround for non-symmetric case
            gather_send_op.arr = x_nparr
        else:
            send_buf = np.ctypeslib.as_ctypes(x_nparr)
            send_count = x_nparr.size
            recv_buf = None
            if gather_send_op.use_reduce:
                req = self.distribution.reduce(send_buf, send_buf, send_count,
                                               mlsl.DataType.FLOAT, mlsl.ReductionType.SUM,
                                               root_idx, mlsl.GroupType.DATA)
            else:
                req = self.distribution.gather(send_buf, send_count, recv_buf,
                                               mlsl.DataType.FLOAT, root_idx,
                                               mlsl.GroupType.DATA)
            self.mlsl_obj.wait(req)

    def gather_recv_from_mlsl_gather_send(self, gather_recv_id, out):
        gather_recv_op = self.gather_recv_nodes[gather_recv_id]

        # todo: get real root_idx
        root_idx = 0

        # todo: remove that workaround for non-symmetric case
        if self.process_idx == root_idx:
            send_node = next(op for op in gather_recv_op.control_deps
                             if isinstance(op, CPUMlslGatherSendOp))
            send_buf = np.ctypeslib.as_ctypes(send_node.arr)
            send_count = send_node.arr.size
            recv_buf = np.ctypeslib.as_ctypes(np.atleast_1d(out))

            if gather_recv_op.use_reduce:
                req = self.distribution.reduce(send_buf, recv_buf, send_count,
                                               mlsl.DataType.FLOAT, mlsl.ReductionType.SUM,
                                               root_idx, mlsl.GroupType.DATA)
            else:
                req = self.distribution.gather(send_buf, send_count, recv_buf,
                                               mlsl.DataType.FLOAT, root_idx,
                                               mlsl.GroupType.DATA)
            self.mlsl_obj.wait(req)

            # todo: replace by real reduce operation
            if gather_recv_op.use_reduce:
                out /= self.process_count

        return out

    def mlsl_scatter_send(self, scatter_send_id, x_nparr):
        scatter_send_op = self.scatter_send_nodes[scatter_send_id]

        # todo: get real root_idx
        root_idx = 0

        # todo: remove that workaround for non-symmetric case
        if self.process_idx == root_idx:
            scatter_send_op.arr = x_nparr

    def scatter_recv_from_mlsl_scatter_send(self, scatter_recv_id, out):
        scatter_recv_op = self.scatter_recv_nodes[scatter_recv_id]

        # todo: get real root_idx
        root_idx = 0

        # todo: remove that workaround for non-symmetric case
        send_buf = None
        if self.process_idx == root_idx:
            send_node = next(op for op in scatter_recv_op.control_deps
                             if isinstance(op, CPUMlslScatterSendOp))
            send_buf = np.ctypeslib.as_ctypes(send_node.arr)
        recv_buf = np.ctypeslib.as_ctypes(out)
        recv_count = out.size

        req = self.distribution.scatter(send_buf, recv_buf, recv_count,
                                        mlsl.DataType.FLOAT, root_idx,
                                        mlsl.GroupType.DATA)
        self.mlsl_obj.wait(req)
        return out

    def mlsl_allreduce_start(self, allreduce_id, out, x_nparr):
        allreduce_op = self.allreduce_nodes[allreduce_id]
        if not hasattr(allreduce_op, '_req'):
            allreduce_op._req = [None]
        if allreduce_op.reduce_func == 'sum' or allreduce_op.reduce_func == 'mean':
            allreduce_op.arr = out
            send_buf = np.ctypeslib.as_ctypes(x_nparr)
            send_count = x_nparr.size
            recv_buf = np.ctypeslib.as_ctypes(out)
            allreduce_op.req = self.distribution.all_reduce(send_buf, recv_buf, send_count,
                                                            mlsl.DataType.FLOAT,
                                                            mlsl.ReductionType.SUM,
                                                            mlsl.GroupType.DATA)
        else:
            raise RuntimeError('Reduce function {} is not supported.'
                               .format(allreduce_op.reduce_func))

    def mlsl_allreduce_wait(self, allreduce_id):
        allreduce_op = self.allreduce_nodes[allreduce_id]
        start_node = next(op for op in allreduce_op.control_deps
                          if isinstance(op, CPUMlslAllReduceStartOp))
        self.mlsl_obj.wait(start_node.req)

        if allreduce_op.reduce_func == 'sum':
            # sum reduction is performed inside MLSL
            pass
        elif allreduce_op.reduce_func == 'mean':
            start_node.arr /= self.process_count
        else:
            raise RuntimeError('Reduce function {} is not supported.'
                               .format(allreduce_op.reduce_func))

    def mlsl_broadcast_send(self, broadcast_send_id, x_nparr):
        broadcast_send_op = self.broadcast_send_nodes[broadcast_send_id]

        # todo: get real root_idx
        root_idx = 0

        # todo: remove that workaround for non-symmetric case
        if self.process_idx == root_idx:
            broadcast_send_op.arr = x_nparr

    def broadcast_recv_from_mlsl_broadcast_send(self, broadcast_recv_id, out):
        broadcast_recv_op = self.broadcast_recv_nodes[broadcast_recv_id]

        # todo: get real root_idx
        root_idx = 0

        # todo: remove that workaround for non-symmetric case
        req = None
        if self.process_idx == root_idx:
            send_buf = None
            send_node = next(op for op in broadcast_recv_op.control_deps
                             if isinstance(op, CPUMlslBroadcastSendOp))
            send_buf = np.ctypeslib.as_ctypes(send_node.arr)
            count = send_node.arr.size
            req = self.distribution.bcast(send_buf, count,
                                          mlsl.DataType.FLOAT, root_idx,
                                          mlsl.GroupType.DATA)
            out[...] = send_node.arr
        else:
            recv_buf = np.ctypeslib.as_ctypes(out)
            count = out.size
            req = self.distribution.bcast(recv_buf, count,
                                          mlsl.DataType.FLOAT, root_idx,
                                          mlsl.GroupType.DATA)
        self.mlsl_obj.wait(req)
        return out
