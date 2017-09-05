from concurrent import futures
import argparse
import time
import socket
import grpc
import hetr_pb2
import hetr_pb2_grpc
from mpi4py import MPI
from ngraph.op_graph.op_graph import Op
from ngraph.op_graph.serde.serde import protobuf_to_op, pb_to_tensor, _deserialize_graph,\
    tensor_to_protobuf, assign_scalar, protobuf_scalar_to_python, is_scalar_type
from ngraph.transformers.hetrtransform import build_transformer


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class HetrServer(hetr_pb2_grpc.HetrServicer):

    def __init__(self, comm, server):
        self.results = dict()
        self.computations = dict()
        self.comp_id_ctr = 0
        self.comm = comm
        self.server = server

    def new_comp_id(self):
        c_id = self.comp_id_ctr
        self.comp_id_ctr += 1
        return c_id

    def BuildTransformer(self, request, context):
        if request.transformer_type[:3] not in ['gpu', 'cpu']:
            return hetr_pb2.BuildReply(status=False)

        try:
            self.transformer = build_transformer(name=request.transformer_type, comm=self.comm)
            return hetr_pb2.BuildReply(status=True)
        except:
            return hetr_pb2.BuildReply(status=False)

    def Computation(self, request, context):
        if not self.transformer:
            return hetr_pb2.ComputationReply(comp_id=-1)

        try:
            comp_id = self.new_comp_id()
            subgraph = _deserialize_graph(request.subgraph)
            returns = []
            placeholders = []
            for pb_op in request.returns:
                returns.append(protobuf_to_op(pb_op))
            for pb_op in request.placeholders:
                placeholders.append(protobuf_to_op(pb_op))
            return_list = []
            placeholder_list = []
            ops = Op.ordered_ops(subgraph)
            for r in returns:
                for op in ops:
                    if op.uuid == r.uuid:
                        return_list.append(op)
            for p in placeholders:
                for op in ops:
                    if op.uuid == p.uuid:
                        placeholder_list.append(op)
            computation = self.transformer.computation(return_list, *placeholder_list)

            self.computations[comp_id] = computation
            return hetr_pb2.ComputationReply(comp_id=comp_id)
        except:
            return hetr_pb2.ComputationReply(comp_id=-1)

    def FeedInput(self, request, context):
        if request.comp_id >= len(self.computations):
            return hetr_pb2.FeedInputReply(status=False)

        try:
            values = []
            for v in request.values:
                if v.HasField('scalar'):
                    values.append(protobuf_scalar_to_python(v.scalar))
                else:
                    values.append(pb_to_tensor(v.tensor))
            computation = self.computations[request.comp_id]

            if self.transformer.transformer_name == "gpu":
                import pycuda.driver as drv
                if self.transformer.runtime and \
                   not self.transformer.runtime.ctx == drv.Context.get_current():
                    self.transformer.runtime.ctx.push()
                outputs = computation(*values)
                self.transformer.runtime.ctx.pop()
            else:
                outputs = computation(*values)

            self.results[request.comp_id] = outputs

            return hetr_pb2.FeedInputReply(status=True)
        except:
            return hetr_pb2.FeedInputReply(status=False)

    def GetResults(self, request, context):
        if request.comp_id >= len(self.results):
            return hetr_pb2.GetResultsReply(status=False)

        try:
            pb_results = []
            for r in self.results[request.comp_id]:
                pb_val = hetr_pb2.Value()
                if is_scalar_type(r):
                    assign_scalar(pb_val.scalar, r)
                else:
                    pb_val.tensor.CopyFrom(tensor_to_protobuf(r))
                pb_results.append(pb_val)
            return hetr_pb2.GetResultsReply(status=True, results=pb_results)
        except:
            return hetr_pb2.GetResultsReply(status=False)


def is_port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(('localhost', int(port)))
    except:
        return False
    s.close()
    return True


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--ports", nargs='+', default=['51051'])
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    options = [('grpc.max_receive_message_length', -1)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    hetr_pb2_grpc.add_HetrServicer_to_server(HetrServer(comm, server), server)
    p = args.ports[comm.Get_rank()]
    if is_port_open(p):
        server.add_insecure_port('[::]:' + p)
    else:
        raise RuntimeError("Port %s is already in use!", p)

    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
