from passes import PeepholeGraphPass
from ngraph.op_graph.communication import Send
from ngraph.op_graph.communication import Recv


class DeviceAssignPass(PeepholeGraphPass):
    def __init__(self, default_device, default_device_id):
        super(DeviceAssignPass, self).__init__()

        self.default_device = default_device
        self.default_device_id = default_device_id

    def visit(self, op):
        if 'device' not in op.metadata:
            op.metadata['device'] = self.default_device
        if 'device_id' not in op.metadata:
            op.metadata['device_id'] = self.default_device_id
        #print(op, op.metadata['device'])


class CommunicationPass(PeepholeGraphPass):
    def __init__(self):
        super(CommunicationPass, self).__init__()

    def visit(self, op):
        args = list()
        for arg in op.args:
            if op.metadata['device'] != arg.metadata['device']:
                args.append(Send(Recv(arg, device=arg.metadata['device'], device_id=arg.metadata['device_id']), 
                    device=op.metadata['device'], device_id=op.metadata['device_id']))
            else:
                args.append(arg)

        if type(op.args) == tuple:
            op.args = tuple(args)
        else:
            op.args(args)  # setter is called args


class ChildTransformerPass(PeepholeGraphPass):
    def __init__(self, transformer_list):
        super(ChildTransformerPass, self).__init__()

        self.transformer_list = transformer_list

    def visit(self, op):
        if 'parallel' in op.metadata:
            print "axis:", op.metadata['parallel'].name, "==> length:", op.metadata['parallel'].length
            assert(isinstance(op.metadata['device_id'], (list, tuple)))
            #TODO: implement scatter/gather

        if isinstance(op.metadata['device_id'], (list, tuple)):
            op.metadata['transformer'] = list()
            for device_id in op.metadata['device_id']:
                transformer = op.metadata['device'] + str(device_id)
                op.metadata['transformer'].append(transformer)
                if transformer not in self.transformer_list:
                    self.transformer_list.append(transformer)
        else:
            transformer = op.metadata['device'] + str(op.metadata['device_id'])
            op.metadata['transformer'] = transformer
            if transformer not in self.transformer_list:
                self.transformer_list.append(transformer)

