from concurrent import futures
import time
import grpc
import hetr_pb2
import hetr_pb2_grpc
from ngraph.op_graph.op_graph import Op
from ngraph.op_graph.serde.serde import protobuf_to_op, pb_to_tensor, _deserialize_graph
from ngraph.transformers.hetrtransform import build_transformer


_ONE_DAY_IN_SECONDS = 60 * 60 * 24


class HetrServer(hetr_pb2_grpc.HetrServicer):

    def __init__(self):
        self.results = dict()
        self.computations = dict()
        self.comp_id_ctr = 0

    def new_comp_id(self):
        c_id = self.comp_id_ctr
        self.comp_id_ctr += 1
        return c_id

    def BuildTransformer(self, request, context):
        self.transformer = build_transformer(request.transformer_type)
        return hetr_pb2.BuildReply(status=True)

    def Computation(self, request, context):
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
        for op in Op.ordered_ops(subgraph):
            for r in returns:
                if op.uuid == r.uuid:
                    return_list.append(op)
        for op in Op.ordered_ops(subgraph):
            for p in placeholders:
                if op.uuid == p.uuid:
                    placeholder_list.append(op)
        computation = self.transformer.computation(return_list, *placeholder_list)
        self.computations[comp_id] = computation
        return hetr_pb2.ComputationReply(comp_id=comp_id)

    def FeedInput(self, request, context):
        values = []
        # TODO do we need to support both scalars and non-scalars in one request?
        for v in request.scalar_values:
            values.append(v)
        for v in request.tensor_values:
            values.append(pb_to_tensor(v))
        computation = self.computations[request.comp_id]
        outputs = computation(*values)
        self.results[request.comp_id] = outputs
        return hetr_pb2.FeedInputReply(status=True)

    def GetResults(self, request, context):
        return hetr_pb2.GetResultsReply(results=self.results[request.comp_id])


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    hetr_pb2_grpc.add_HetrServicer_to_server(HetrServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()
