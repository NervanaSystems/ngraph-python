from neon import NervanaObject  # noqa

from ngraph.transformers.base import Transformer, make_transformer_factory, set_transformer_factory
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, CommunicationPass

#DeviceBufferStorage, DeviceBufferReference, \
#    DeviceTensor

class HeTrTransformer(Transformer):
    """
    Transformer for executing graphs on a CPU, backed by numpy.

    Given a list of ops you want to compute the results of, this transformer
    will compile the graph required to compute those results and exposes an
    evaluate method to execute the compiled graph.
    """

    transformer_name = "hetr"

    def __init__(self, **kwargs):
        super(HeTrTransformer, self).__init__(**kwargs)
        self.register_graph_pass(DeviceAssignPass(default_device='gpu'))
        self.register_graph_pass(CommunicationPass())




    def device_buffer_storage(self, bytes, dtype, name):
        """
        Make a DeviceBuffer.

        Arguments:
            bytes: Size of buffer.
            alignment: Alignment of buffer.

        Returns: A DeviceBuffer.
        """
        print("device_buffer_storage")
        return []


    def device_buffer_reference(self):
        """
        Make a DeviceBufferReference.

        Returns: A DeviceBufferReference.
        """
        print("device_buffer_reference")
        return None


    def start_transform_allocate(self):
        print("start_transform_allocate")

    def finish_transform_allocate(self):
        print("finish_transform_allocate")

    def transform_ordered_ops(self, ordered_ops, name):
        print(name, ordered_ops)
        return name + 1


    def finish_transform(self):
        pass


    def allocate_storage(self):
        pass

set_transformer_factory(
    make_transformer_factory(HeTrTransformer.transformer_name))