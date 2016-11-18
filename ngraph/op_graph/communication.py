from op_graph import TensorOp

class Send(TensorOp):
    def __init__(self, from_nodes, device=None):
        super(Send, self).__init__(self)
        self.args = tuple([from_nodes])
        self.metadata['device'] = device

class Recv(TensorOp):
    def __init__(self, from_nodes, device=None):
        super(Recv, self).__init__(self)
        self.args = tuple([from_nodes])
        self.metadata['device'] = device

