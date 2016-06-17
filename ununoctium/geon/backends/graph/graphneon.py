import geon.backends.graph.funs as be
from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
from neon.initializers import Uniform
import geon.backends.graph.axis as ax

from geon.backends.graph.layer import *
from geon.backends.graph.model import Model

from neon.callbacks.callbacks import *

def CrossEntropyMulti():
    return be.cross_entropy_multi


def CrossEntropyBinary():
    return be.cross_entropy_binary