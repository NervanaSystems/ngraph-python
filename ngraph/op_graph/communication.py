from op_graph import TensorOp

class Send(TensorOp):
    def __init__(self, from_node, device=None):
        super(Send, self).__init__(self)
        self.args = tuple([from_node])
        self.metadata['device'] = device
        self.axes = from_node.axes

class Recv(TensorOp):
    def __init__(self, from_node, device=None):
        super(Recv, self).__init__(self)
        self.args = tuple([from_node])
        self.metadata['device'] = device
        self.axes = from_node.axes

