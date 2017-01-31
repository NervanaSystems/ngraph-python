from op_graph import TensorOp


class Send(TensorOp):

    def __init__(self, from_node, queue, device=None, device_id=None):
        super(Send, self).__init__()
        self.args = tuple([from_node])
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.axes = from_node.axes
        self.dtype = from_node.dtype
        self.shared_q = queue


class Receiver(TensorOp):

    def __init__(self, send_node):
        super(Receiver, self).__init__()
        self._send_node = send_node

    def send_node(self):
        return self._send_node


class Recv(Receiver):

    def __init__(self, axes, dtype, queue, send_node, device=None, device_id=None):
        super(Recv, self).__init__(send_node)
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.args = ()
        self.axes = axes
        self.dtype = dtype
        self.shared_q = queue


class Scatter_Send(TensorOp):

    def __init__(
            self,
            from_node,
            axes,
            parallel_axis,
            queues,
            device=None,
            device_id=None,
            to_id=None):
        super(Scatter_Send, self).__init__(self)
        self.args = tuple([from_node])
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.dtype = from_node.dtype
        self.shared_queue_list = queues
        self.to_id = to_id
        self.axes = axes
        self.slices = list()
        for i in range(len(to_id)):
            slices = list()
            for a in self.axes:
                s = slice(None)
                if parallel_axis == a:
                    remainder = a.length % len(to_id)
                    new_length = a.length / len(to_id)
                    start = i * new_length
                    stop = (i + 1) * new_length
                    step = 1
                    if remainder > 0:
                        if i == (len(to_id) - 1):
                            stop += remainder
                    s = slice(start, stop, step)
                slices.append(s)
            self.slices.append(slices)


class Scatter_Recv(Receiver):

    def __init__(self, axes, dtype, queue, send_node, device=None, device_id=None):
        super(Scatter_Recv, self).__init__(send_node)
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.axes = axes
        self.dtype = dtype
        self.shared_queue = queue


class Gather_Send(TensorOp):

    def __init__(self, from_node, axes, queue, device=None, device_id=None, **kwargs):
        super(Gather_Send, self).__init__(self)
        self.args = tuple([from_node])
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.axes = axes
        self.dtype = from_node.dtype
        self.shared_queue = queue


class Gather_Recv(Receiver):

    def __init__(
            self,
            axes,
            dtype,
            parallel_axis,
            queues,
            send_node,
            device=None,
            device_id=None,
            from_id=None):
        super(Gather_Recv, self).__init__(send_node)
        self.metadata['device'] = device
        self.metadata['device_id'] = device_id
        self.metadata['marker'] = 'gather'
        self.metadata['parallel'] = parallel_axis
        self.shared_queue_list = queues
        self.from_id = from_id
        self.axes = axes
        self.slices = list()
        for i in range(len(from_id)):
            slices = list()
            for a in self.axes:
                s = slice(None)
                if parallel_axis == a:
                    remainder = a.length % len(from_id)
                    new_length = a.length / len(from_id)
                    start = i * new_length
                    stop = (i + 1) * new_length
                    step = 1
                    if remainder > 0:
                        if i == (len(from_id) - 1):
                            stop += remainder
                    s = slice(start, stop, step)
                slices.append(s)
            self.slices.append(slices)
