import grpc
from six import iteritems

from . import hetr_pb2
from . import hetr_pb2_grpc
from ngraph.op_graph.serde.serde import op_to_protobuf, tensor_to_protobuf, add_edges,\
    pb_to_tensor, is_scalar_type, assign_scalar, protobuf_scalar_to_python
from ngraph.transformers.hetr.hetr_utils import update_comm_deps
from ngraph.op_graph.op_graph import Op
import logging


_TIMEOUT_SECONDS = 600
_OPS_PER_MSG = 10
logger = logging.getLogger(__name__)


def is_channel_ready(channel):
    status = channel._channel.check_connectivity_state(True)
    return ((status == 0) or (status == 2))  # 0: IDLE, 2: READY


class RPCComputationClient(object):
    def __init__(self, comp_id, stub):
        self.comp_id = comp_id
        self.RPC = stub
        self.feed_input_response_future = None

    def feed_input(self, values):
        logger.info("client: feed input")
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
        logger.info("client: get results")
        if self.feed_input_response_future is None:
            raise RuntimeError("call feed_input before get_results")
        response = self.feed_input_response_future.result()
        self.feed_input_response_future = None
        if not response.status:
            raise RuntimeError("RPC feed_input request failed: {}".format(response.message))
        response = self.RPC.GetResults(
            hetr_pb2.GetResultsRequest(comp_id=self.comp_id),
            _TIMEOUT_SECONDS)
        if not response.status:
            raise RuntimeError("RPC get_results request failed: {}".format(response.message))
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

    def __init__(self, transformer_type, server_address='localhost'):
        logger.info("client: init, transformer: %s, server_address: %s",
                    transformer_type, server_address)
        self.transformer_type = transformer_type
        self.server_address = server_address
        self.computations = dict()
        self.computation_builds = dict()
        self.comp_id_ctr = 0
        self.is_trans_built = False
        self.computation_response_future = None
        self.close_transformer_response_future = None

    def set_server_address(self, address):
        if self.is_trans_built:
            logger.info("client: set_server_address: transformer is already built, \
                        skip server address")
            return
        self.server_address = address

    def build_transformer(self):
        logger.info("client: build_transformer, server address: %s", self.server_address)
        if self.is_trans_built:
            logger.info("client: build_transformer: transformer is already built")
            return
        options = [('grpc.max_send_message_length', -1), ('grpc.max_receive_message_length', -1)]
        channel = grpc.insecure_channel(self.server_address, options=options)
        if not is_channel_ready(channel):
            raise RuntimeError("gRPC channel is not ready...")
        self.RPC = hetr_pb2_grpc.HetrStub(channel)

        if self.close_transformer_response_future is not None:
            response = self.close_transformer_response_future.result()
            if not response.status:
                raise RuntimeError("RPC close_transformer request failed: {}"
                                   .format(response.message))
            self.is_trans_built = False
            self.close_transformer_response_future = None

        response = self.RPC.BuildTransformer(
            hetr_pb2.BuildTransformerRequest(transformer_type=self.transformer_type),
            _TIMEOUT_SECONDS)
        if response.status:
            self.is_trans_built = True
        else:
            self.is_trans_built = False
            raise RuntimeError("RPC build_transformer request failed: {}".format(response.message))

    def create_computation(self, pb_ops, pb_edges, returns, placeholders):
        logger.debug("client: create_computation")

        def make_computation_request(pb_ops, pb_edges, pb_returns=None, pb_placeholders=None):
            return hetr_pb2.ComputationRequest(
                ops=pb_ops,
                edges=pb_edges,
                returns=pb_returns,
                placeholders=pb_placeholders)

        def generate_messages():
            pb_returns = [op_to_protobuf(o) for o in returns]
            pb_placeholders = [op_to_protobuf(o) for o in placeholders]

            ops = Op.all_op_references(returns + list(placeholders))
            yield make_computation_request(pb_ops,
                                           pb_edges,
                                           pb_returns,
                                           pb_placeholders)

        if not self.is_trans_built:
            raise RuntimeError("call build_transformer before create_computation")

        update_comm_deps(returns)

        self.computation_response_future = self.RPC.Computation.future(
            generate_messages(), _TIMEOUT_SECONDS)

    def get_computation(self):
        logger.info("client: get_computation")
        if self.computation_response_future is None:
            raise RuntimeError("call create_computation before get_computation")
        response = self.computation_response_future.result()
        self.computation_response_future = None
        if response.comp_id >= 0:
            rpcComputationClient = RPCComputationClient(response.comp_id, self.RPC)
            return rpcComputationClient
        else:
            raise RuntimeError("RPC computation request failed: {}".format(response.message))

    def close_transformer(self):
        logger.info("client: close_transformer")
        if self.is_trans_built:
            self.close_transformer_response_future = self.RPC.CloseTransformer.future(
                hetr_pb2.CloseTransformerRequest(),
                _TIMEOUT_SECONDS)

    def close(self):
        logger.info("client: close")
        if self.close_transformer_response_future is not None:
            response = self.close_transformer_response_future.result()
            if not response.status:
                raise RuntimeError("RPC close_transformer request failed: {}"
                                   .format(response.message))
            self.is_trans_built = False
            self.close_transformer_response_future = None
        try:
            self.RPC.Close.future(
                hetr_pb2.CloseRequest(),
                _TIMEOUT_SECONDS)
        except:
            pass
