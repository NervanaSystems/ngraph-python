from geon.backends.graph.nptransform import NumPyTransformer
import inspect


class DebugPauser(object):
    pass


class DebugTransformer(NumPyTransformer):
    def __init__(self, debug_pauser, *args, **kargs):
        super(DebugTransformer, self).__init__(*args, **kargs)
        self.debug_pauser = debug_pauser
        self.breakpoints = set()

    def dot(self, x, y, out):
        super(NumPyTransformer, self).dot(x, y, out)
        calframe = inspect.getouterframes(inspect.getcurrentframe(), 2)