from __future__ import print_function
import ngraph.transformers as ngt
from ngraph.flex.names import flex_gpu_transformer_name


class FlexNgraphArgparser():
    """
    Flex specific command line args
    """

    @staticmethod
    def setup_flex_args(argParser):
        """
        Add flex specific arguments to other default args used by ngraph
        """
        argParser.add_argument('--fixed_point',
                               action="store_true",
                               help='use fixed point for flex backend')
        argParser.add_argument('--flex_verbose',
                               action="store_true",
                               help='turn on flex verbosity for debug')
        argParser.add_argument('--collect_flex_data',
                               action="store_true",
                               help="collect flex data and save it to h5py File")

    @staticmethod
    def make_and_set_transformer_factory(args):

        flex_args = ('fixed_point', 'flex_verbose', 'collect_flex_data')
        # default value for all flex args if not given, confusing with store_true in add_argument
        default = False

        if args.backend == flex_gpu_transformer_name:
                flex_args_dict = dict((a, getattr(args, a, default)) for a in flex_args)
                factory = ngt.make_transformer_factory(args.backend, **flex_args_dict)
        else:
            factory = ngt.make_transformer_factory(args.backend)

        ngt.set_transformer_factory(factory)
