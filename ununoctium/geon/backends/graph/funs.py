import geon.backends.graph.dataloaderbackend
import geon.backends.graph.npbackend
import geon.backends.graph.graph as graph

from geon.backends.graph.names import name_scope_list, with_name_scope, name_scope,  \
    next_name_scope
from geon.backends.graph.graph import Model, with_graph_scope, with_environment, \
    get_current_environment
from geon.backends.graph.environment import bound_environment, Environment

from geon.backends.graph.transform import deriv, placeholder, Variable, Constant, linear_map_axes, sample_axes, \
    batch_axes, assign
from geon.backends.graph.arrayaxes import Axis, Axes, AxisVar, NumericAxis,\
    AxisID, AxisIDTuple, set_batch_axes, get_batch_axes, set_phase_axes, get_phase_axes

from geon.backends.graph.transform import absolute, add, argmax, argmin, cos, divide, dot, equal, exp, log, max, \
    maximum, \
    mean, min, minimum, multiply, greater, greater_equal, less, less_equal, power
from geon.backends.graph.transform import negative, not_equal, NumPyTensor, reciprocal, sig, sin, softmax, sqrt, square, subtract, \
    sum
from geon.backends.graph.transform import tanh, safelog, cross_entropy_binary, cross_entropy_multi
from geon.backends.graph.transform import doall, RNG, NumPyTensor, Temporary, tensor_size

from geon.backends.graph.transform import AllReduce
from geon.backends.graph.nptransform import NumPyTransformer

try:
    from geon.backends.graph.artransform import ArgonTransformer
except ImportError:
    print("Argon backend and tensor are defined in argon repo.")


# TODO These are just here as placeholders
def add_fc_bias(self, inputs, bias):
    pass


def batched_dot(self, A, B, C, alpha=None, beta=None, relu=None):
    pass


def check_cafe_compat(self):
    pass


def clip(self, a, a_min, a_max, out=None):
    pass


def compound_dot(self, A, B, C, alpha=None, beta=None, relu=None):
    pass


def exp2(self, a, out=None):
    pass


def fabs(self, a, out=None):
    pass


def finite(self, a, out=None):
    pass


def gen_rng(self, seed=None):
    pass


def log2(self, a, out=None):
    pass


def make_binary_mask(self, out, keepthresh=None):
    pass


def output_dim(self, X, S, padding, strides, pooling=None):
    pass


def rng_get_state(self, state):
    pass


def rng_reset(self):
    pass


def rng_set_state(self, state):
    pass


def set_caffe_compat(self):
    pass


def sig2(self, a, out=None):
    pass


def std(self, a, axis=None, partial=None, out=None, keepdims=None):
    pass


def take(self, a, indices, axis, out=None):
    pass


def tanh2(self, a, out=None):
    pass


def true_divide(self, a, b, out=None):
    pass


def update_fc_bias(self, err, out):
    pass


def var(self, a, axis=None, partial=None, out=None, keepdims=None):
    pass
