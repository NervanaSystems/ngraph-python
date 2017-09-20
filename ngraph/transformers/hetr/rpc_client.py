import grpc
from six import iteritems

from . import hetr_pb2
from . import hetr_pb2_grpc
from ngraph.op_graph.serde.serde import op_to_protobuf, tensor_to_protobuf, add_edges,\
    pb_to_tensor, is_scalar_type, assign_scalar, protobuf_scalar_to_python
from ngraph.transformers.hetr.hetr_utils import update_comm_deps
from ngraph.op_graph.op_graph import Op


_TIMEOUT_SECONDS = 600
_SLEEP_SECONDS = 1
_OPS_PER_MSG = 10


def is_channel_ready(channel):
    status = channel._channel.check_connectivity_state(True)
    return ((status == 0) or (status == 2))  # 0: IDLE, 2: READY


class RPCComputationClient(object):
    def __init__(self, comp_id, stub):
        self.comp_id = comp_id
        self.RPC = stub

    def feed_input(self, values):
        pb_values = []
        for v in values:
            pb_val = hetr_pb2.Value()
            if is_scalar_type(v):
                assign_scalar(pb_val.scalar, v)
            else:
                pb_val.tensor.CopyFrom(tensor_to_protobuf(v))
            pb_values.append(pb_val)
        self.feed_input_response_future = self.RPC.FeedInput.future(
            hetr_pb2.FeedInputRequest(
                comp_id=self.comp_id,
                values=pb_values),
            _TIMEOUT_SECONDS)

    def get_results(self):
        response = self.feed_input_response_future.result()
        if not response.status:
            raise RuntimeError("RPC feed_input request failed!")
        response = self.RPC.GetResults(
            hetr_pb2.GetResultsRequest(comp_id=self.comp_id),
            _TIMEOUT_SECONDS)
        if not response.status:
            raise RuntimeError("RPC get_results request failed!")
        return_list = []
        for r in response.results:
            if r.HasField('scalar'):
                return_list.append(protobuf_scalar_to_python(r.scalar))
            else:
                return_list.append(pb_to_tensor(r.tensor))
        return_dict = {op: return_list[mypos]
                       for (op, mypos) in iteritems(self.returns)}
        return return_dict


class RPCTransformerClient(object):

    def __init__(self, transformer_type, port):
        self.transformer_type = transformer_type
        self.computations = dict()
        self.computation_builds = dict()
        self.comp_id_ctr = 0

        options = [('grpc.max_send_message_length', -1)]
        channel = grpc.insecure_channel('localhost:' + port, options=options)
        if not is_channel_ready(channel):
            raise RuntimeError("gRPC channel is not ready...")

        self.RPC = hetr_pb2_grpc.HetrStub(channel)
        response = self.RPC.BuildTransformer(
            hetr_pb2.BuildRequest(transformer_type=transformer_type),
            _TIMEOUT_SECONDS)
        if response.status:
            self.initialized = True
        else:
            self.initialized = False

    def computation(self, returns, placeholders):

        def make_computation_request(pb_ops, pb_edges, pb_returns=None, pb_placeholders=None):
            if pb_returns or pb_placeholders:
                return hetr_pb2.ComputationRequest(
                    ops=pb_ops,
                    edges=pb_edges,
                    returns=pb_returns,
                    placeholders=pb_placeholders)
            else:
                return hetr_pb2.ComputationRequest(
                    ops=pb_ops,
                    edges=pb_edges)

        def generate_returns_placeholders():
            pb_returns = []
            pb_placeholders = []
            for op in returns:
                pb_returns.append(op_to_protobuf(op))
            for op in placeholders:
                pb_placeholders.append(op_to_protobuf(op))
            return pb_returns, pb_placeholders

        def generate_messages():
            pb_ops, pb_edges = [], []
            pb_returns, pb_placeholders = generate_returns_placeholders()
            ops = Op.all_op_references(returns + list(placeholders))
            for i, op in enumerate(ops):
                pb_ops.append(op_to_protobuf(op))
                add_edges(pb_edges, pb_ops, op)
                if (i != 0) and (i % _OPS_PER_MSG == 0 or i == len(ops) - 1):
                    msg = make_computation_request(pb_ops,
                                                   pb_edges,
                                                   pb_returns,
                                                   pb_placeholders)
                    yield msg

                    pb_ops, pb_edges = [], []
                    pb_returns, pb_placeholders = [], []

        if not self.initialized:
            raise RuntimeError("RPC build_transformer request failed!")
        update_comm_deps(returns)
        response = self.RPC.Computation(
            generate_messages(),
            _TIMEOUT_SECONDS)
        if response.comp_id >= 0:
            rpcComputationClient = RPCComputationClient(response.comp_id, self.RPC)
            return rpcComputationClient
        else:
            raise RuntimeError("RPC computation request failed!")

    def close(self):
        pass
