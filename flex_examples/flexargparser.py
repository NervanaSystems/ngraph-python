from ngraph.frontends.neon.argparser import NgraphArgparser
import ngraph.transformers as ngt

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

    def make_and_set_transformer_factory(self, args):

        flex_args = ('fixed_point', 'flex_verbose')
        default = False  # default value for all flex args if not given
                         # this is confusing with store_true in add_argument

        if args.backend == 'gpuflex' and any([hasattr(args, a) for a in flex_args]):
            flex_args_dict = dict((a, getattr(args, a, default)) for a in flex_args)
            factory = ngt.make_transformer_factory(args.backend, **flex_args_dict)
        else:
            factory = ngt.make_transformer_factory(args.backend)

        ngt.set_transformer_factory(factory)
