from concurrent import futures
import argparse
import time
import socket
import grpc
import hetr_pb2
import hetr_pb2_grpc
import traceback
from mpi4py import MPI
from ngraph.op_graph.op_graph import Op
from ngraph.op_graph.serde.serde import protobuf_to_op, pb_to_tensor, tensor_to_protobuf,\
    _deserialize_graph_ops_edges, assign_scalar, protobuf_scalar_to_python, is_scalar_type
from ngraph.transformers.hetrtransform import build_transformer
import logging
import os
import fcntl

try:
    # The first "import mlsl" will create internal mlsl object and will init MLSL library.
    # That object will be destroyed explicitly over HetrLocals.close_module()->mlsl.close().
    import mlsl  # noqa: F401
    from ngraph.transformers.cpu.hetr import HetrLocals
    use_mlsl = True
except ImportError:
    use_mlsl = False


_ONE_DAY_IN_SECONDS = 60 * 60 * 24
logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class HetrServer(hetr_pb2_grpc.HetrServicer):

    def __init__(self, comm, server):
        self.results = dict()
        self.computations = dict()
        self.comp_id_ctr = 0
        self.comm = comm
        self.server = server
        self.transformer_type = None

    def new_comp_id(self):
        c_id = self.comp_id_ctr
        self.comp_id_ctr += 1
        return c_id

    def BuildTransformer(self, request, context):
        logger.info("server: build_transformer")
        self.transformer_type = request.transformer_type[:3]
        if self.transformer_type not in ['gpu', 'cpu']:
            message = 'unknown transformer type {}'.format(self.transformer_type)
            return hetr_pb2.BuildTransformerReply(status=False, message=message)

        try:
            self.transformer = build_transformer(name=request.transformer_type, comm=self.comm)
            return hetr_pb2.BuildTransformerReply(status=True)
        except Exception:
            return hetr_pb2.BuildTransformerReply(status=False, message=traceback.format_exc())

    def Computation(self, request_iterator, context):
        logger.info("server: computation")
        if not self.transformer:
            return hetr_pb2.ComputationReply(comp_id=-1,
                                             message="build transformer before computation")
        try:
            comp_id = self.new_comp_id()

            pb_ops, pb_edges = [], []
            returns, placeholders = [], []
            reconstructed_returns, reconstructed_placeholders = [], []
            for request in request_iterator:
                pb_ops.extend(request.ops)
                pb_edges.extend(request.edges)
                returns.extend([protobuf_to_op(op) for op in request.returns])
                placeholders.extend([protobuf_to_op(op) for op in request.placeholders])

            subgraph = _deserialize_graph_ops_edges(pb_ops, pb_edges)

            ops = Op.ordered_ops(subgraph)
            for r in returns:
                for op in ops:
                    if op.uuid == r.uuid:
                        reconstructed_returns.append(op)
            for p in placeholders:
                for op in ops:
                    if op.uuid == p.uuid:
                        reconstructed_placeholders.append(op)

            computation = self.transformer.computation(reconstructed_returns,
                                                       *reconstructed_placeholders)
            self.computations[comp_id] = computation
            return hetr_pb2.ComputationReply(comp_id=comp_id)
        except Exception:
            return hetr_pb2.ComputationReply(comp_id=-1, message=traceback.format_exc())

    def FeedInput(self, request, context):
        logger.info("server: feed_input")
        if request.comp_id not in self.computations:
            message = 'unknown computation id {}'.format(request.comp_id)
            return hetr_pb2.FeedInputReply(status=False, message=message)

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
        except Exception:
            return hetr_pb2.FeedInputReply(status=False, message=traceback.format_exc())

    def GetResults(self, request, context):
        logger.info("server: get_results")
        if request.comp_id not in self.results:
            message = 'unknown computation id {}'.format(request.comp_id)
            return hetr_pb2.GetResultsReply(status=False, message=message)

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
        except Exception:
            return hetr_pb2.GetResultsReply(status=False, message=traceback.format_exc())

    def CloseTransformer(self, request, context):
        logger.info("server: close transformer")
        self.transformer.close()
        return hetr_pb2.CloseTransformerReply(status=True)

    def Close(self, request, context):
        logger.info("server: close, self.transformer_type %s", self.transformer_type)
        if use_mlsl:
            HetrLocals.close_mlsl()
        self.server.stop(0)
        return hetr_pb2.CloseReply()


def is_port_open(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('localhost', int(port)))
        return True
    except Exception as e:
        logger.info("is_port_open: port %s, exception: %s", port, e)
        return False
    finally:
        s.close()


def write_server_info(filename, port):
    pid = os.getpid()
    rank = MPI.COMM_WORLD.Get_rank()
    server_info = '{}:{}:{}'.format(rank, pid, port).strip()
    logger.info("write_server_info: line %s, filename %s", server_info, filename)
    with open(filename, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(server_info + '\n')
        fcntl.flock(f, fcntl.LOCK_UN)
    return server_info


def serve():
    parser = argparse.ArgumentParser()
    parser.add_argument("-tf", "--tmpfile", nargs=1)
    parser.add_argument("-p", "--ports", nargs='+')
    args = parser.parse_args()
    comm = MPI.COMM_WORLD

    options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    hetr_pb2_grpc.add_HetrServicer_to_server(HetrServer(comm, server), server)
    logger.info("server: rank %d, tmpfile %s, ports %s",
                comm.Get_rank(), args.tmpfile[0], args.ports if args.ports is not None else "")

    if args.ports is not None and len(args.ports) > comm.Get_rank():
        p = args.ports[comm.Get_rank()]
        if is_port_open(p):
            port = server.add_insecure_port('[::]:' + p)
        else:
            raise RuntimeError("port %s is already in use!", p)
    else:
        port = server.add_insecure_port('[::]:0')

    server.start()
    write_server_info(args.tmpfile[0], port)

    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
