from passes import PeepholeGraphPass
from ngraph.op_graph.communication import Send

class DeviceAssignPass(PeepholeGraphPass):
    def __init__(self, default_device):
        super(DeviceAssignPass, self).__init__()

        self.default_device = default_device

    def visit(self, op):
        if 'device' not in op.metadata:
            op.metadata['device'] = self.default_device
        print(op)


class CommunicationPass(PeepholeGraphPass):
    def __init__(self):
        super(CommunicationPass, self).__init__()

    def visit(self, op):

        args = list()
        for arg in op.args:
            if op.metadata['device'] != arg.metadata['device']:
                args.append(Send(arg, device=op.metadata['device']))
            else:
                args.append(arg)

        if type(op.args) == tuple:
            op.args = tuple(args)
        else:

            op.args(args) # setter is called args

