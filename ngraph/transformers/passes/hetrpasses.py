from passes import PeepholeGraphPass
from ngraph.op_graph.communication import Send
from ngraph.op_graph.communication import Recv

class DeviceAssignPass(PeepholeGraphPass):
    def __init__(self, default_device):
        super(DeviceAssignPass, self).__init__()

        self.default_device = default_device

    def visit(self, op):
        if 'device' not in op.metadata:
            op.metadata['device'] = self.default_device
        #print(op, op.metadata['device'])

class CommunicationPass(PeepholeGraphPass):
    def __init__(self):
        super(CommunicationPass, self).__init__()

    def visit(self, op):
        args = list()
        for arg in op.args:
            if op.metadata['device'] != arg.metadata['device']:
                args.append(Send(Recv(arg, device=arg.metadata['device']), device=op.metadata['device']))
            else:
                args.append(arg)

        if type(op.args) == tuple:
            op.args = tuple(args)
        else:
            op.args(args) # setter is called args

        
class ChildTransformerPass(PeepholeGraphPass):
    def __init__(self, transformer_list):
        super(ChildTransformerPass, self).__init__()

        self.transformer_list = transformer_list

    def visit(self, op):
        if op.metadata['device'] not in self.transformer_list:
            op.metadata['transformer'] = op.metadata['device']
            self.transformer_list.append(op.metadata['transformer'])