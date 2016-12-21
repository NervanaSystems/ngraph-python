from passes import PeepholeGraphPass
from ngraph.op_graph.communication import Send
from ngraph.op_graph.communication import Recv
import multiprocessing


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


class CommunicationPass(PeepholeGraphPass):

    def __init__(self, sendnodes):
        super(CommunicationPass, self).__init__()
        self.send_nodes = sendnodes
        self.dict_transformer_to_op = dict()

    def visit(self, op):
        args = list()
        for arg in op.args:
            if op.metadata['device_id'] != arg.metadata['device_id']:
                shared_q = multiprocessing.Queue()
                self.send_nodes.append(Send(from_node=arg, q=shared_q,
                                            device=arg.metadata['device'],
                                            device_id=arg.metadata['device_id']))

                tname = arg.metadata['device'] + arg.metadata['device_id']
                self.dict_transformer_to_op[tname] = self.send_nodes[-1]
                args.append(Recv(axes=arg.axes, dtype=arg.dtype, q=shared_q,
                                 device=op.metadata['device'],
                                 device_id=arg.metadata['device_id']))

            else:
                args.append(arg)

        if isinstance(op.args, tuple):
            op.args = tuple(args)
        else:
            op.args(args)  # setter is called args


class ChildTransformerPass(PeepholeGraphPass):

    def __init__(self, transformer_list):
        super(ChildTransformerPass, self).__init__()

        self.transformer_list = transformer_list

    def visit(self, op):
        if 'parallel' in op.metadata:
            print "axis:", op.metadata['parallel'].name, \
                  "==> length:", op.metadata['parallel'].length
            assert(isinstance(op.metadata['device_id'], (list, tuple)))
            # TODO: implement scatter/gather

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
