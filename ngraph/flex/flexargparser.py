from __future__ import print_function
from ngraph.frontends.neon.argparser import NgraphArgparser
import ngraph.transformers as ngt
from ngraph.flex.names import flex_gpu_transformer_name


class FlexNgraphArgparser(NgraphArgparser):
    """
    Flex specific command line args
    """

    def __init__(self, *args, **kwargs):

        super(FlexNgraphArgparser, self).__init__(*args, **kwargs)

        self.setup_flex_args()

    def setup_flex_args(self):
        """
        Add flex specific arguments to other default args used by ngraph
        """
        self.add_argument('--fixed_point',
                          action="store_true",
                          help='use fixed point for flex backend')
        self.add_argument('--flex_verbose',
                          action="store_true",
                          help='turn on flex verbosity for debug')

    def backend_names(self):
        backend_names = super(FlexNgraphArgparser, self).backend_names()
        # only add flex gpu transformer as an option if autoflex installed
        try:
            # import GPUFlexManager to check if autoflex installed
            from ngraph.flex import GPUFlexManager  # noqa
            backend_names.append(flex_gpu_transformer_name)
        except ImportError:
            print('{} transformer not available'.format(flex_gpu_transformer_name))
            print('please check if autoflex package is installed')
        return backend_names

    def make_and_set_transformer_factory(self, args):

        flex_args = ('fixed_point', 'flex_verbose')
        # default value for all flex args if not given, confusing with store_true in add_argument
        default = False

        if args.backend == flex_gpu_transformer_name and \
           any([hasattr(args, a) for a in flex_args]):
                flex_args_dict = dict((a, getattr(args, a, default)) for a in flex_args)
                factory = ngt.make_transformer_factory(args.backend, **flex_args_dict)
        else:
            factory = ngt.make_transformer_factory(args.backend)

        ngt.set_transformer_factory(factory)
