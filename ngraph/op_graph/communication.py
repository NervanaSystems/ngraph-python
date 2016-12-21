from op_graph import TensorOp


class Send(TensorOp):
    def __init__(self, from_node, q, device=None, device_id=None):
        super(Send, self).__init__(self)
        self.args = tuple([from_node])
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.axes = from_node.axes
        self.dtype = from_node.dtype
        self.shared_q = q


class Recv(TensorOp):
    def __init__(self, axes, dtype, q, device=None, device_id=None):
        super(Recv, self).__init__(self)
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.args = ()
        self.axes = axes
        self.dtype = dtype
        self.shared_q = q
