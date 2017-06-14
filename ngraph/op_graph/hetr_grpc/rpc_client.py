import grpc
import hetr_pb2
import hetr_pb2_grpc
import os
from six import iteritems
from ngraph.op_graph.serde.serde import op_to_protobuf, tensor_to_protobuf, _serialize_graph,\
    is_scalar_type
from ngraph.util.hetr_utils import update_comm_deps


_TIMEOUT_SECONDS = 600
_SLEEP_SECONDS = 1


def is_channel_ready(channel):
    status = channel._channel.check_connectivity_state(True)
    return status  # 0: IDLE, 2: READY


class RPCComputationClient(object):
    def __init__(self, comp_id, stub):
        self.comp_id = comp_id
        self.RPC = stub

    def feed_input(self, values):
        pb_scalar_values = []
        pb_tensor_values = []
        for v in values:
            if is_scalar_type(v):
                pb_scalar_values.append(v)
            else:
                pb_tensor_values.append(tensor_to_protobuf(v))
        response = self.RPC.FeedInput(
            hetr_pb2.FeedInputRequest(
                comp_id=self.comp_id,
                scalar_values=pb_scalar_values,
                tensor_values=pb_tensor_values),
            _TIMEOUT_SECONDS)
        if not response.status:
            raise RuntimeError("RPC feed_input request failed!")

    def get_results(self):
        response = self.RPC.GetResults(
            hetr_pb2.GetResultsRequest(comp_id=self.comp_id),
            _TIMEOUT_SECONDS)
        return_list = response.results
        return_dict = {op: return_list[mypos]
                       for (op, mypos) in iteritems(self.returns)}
        return return_dict


class RPCTransformerClient(object):

    def __init__(self, transformer_type):
        self.transformer_type = transformer_type
        self.computations = dict()
        self.computation_builds = dict()
        self.comp_id_ctr = 0
        self.my_pid = os.getpid()

        channel = grpc.insecure_channel('localhost:50051')
        self.RPC = hetr_pb2_grpc.HetrStub(channel)
        if self.RPC.BuildTransformer(
                hetr_pb2.BuildRequest(transformer_type=transformer_type),
                _TIMEOUT_SECONDS):
            self.initialized = True

    def computation(self, returns, placeholders):
        update_comm_deps(returns)
        pb_subgraph = _serialize_graph(returns + list(placeholders))
        pb_returns = []
        pb_placeholders = []
        for op in returns:
            pb_returns.append(op_to_protobuf(op))
        for op in placeholders:
            pb_placeholders.append(op_to_protobuf(op))

        response = self.RPC.Computation(
            hetr_pb2.ComputationRequest(
                subgraph=pb_subgraph,
                returns=pb_returns,
                placeholders=pb_placeholders),
            _TIMEOUT_SECONDS)
        if response.comp_id >= 0:
            rpcComputationClient = RPCComputationClient(response.comp_id, self.RPC)
            return rpcComputationClient
        else:
            raise RuntimeError("RPC computation request failed!")

    def close(self):
        if self.my_pid != os.getpid():
            return
