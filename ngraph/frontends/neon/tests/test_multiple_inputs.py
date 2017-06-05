import numpy as np
import ngraph as ng
from ngraph.frontends import neon

def make_model(action_axes):
    """
    Given the expected action axes, return a model mapping from observation to
    action axes for use by the dqn agent.
    """
    return neon.Sequential([
        neon.Convolution(
            (8, 8, 32),
            neon.XavierInit(),
            strides=4,
            activation=neon.Rectlin(),
            batch_norm=True,
        ),
        neon.Affine(
            weight_init=neon.XavierInit(),
            bias_init=neon.ConstantInit(),
            activation=neon.Rectlin(),
            batch_norm=True,
            axes=(action_axes, )
        ),
    ])


def test_convolution():
    action_axis = ng.make_axis(name='action', length=5)
    state_axes = ng.make_axes([
        ng.make_axis(4, name='feature'),
        ng.make_axis(84, name='width'),
        ng.make_axis(84, name='height'),
    ])
    batch_axis_1 = ng.make_axis(name='N', length=1)
    batch_axis_all = ng.make_axis(name='N', length=32)

    state_placeholder = ng.placeholder(state_axes + [batch_axis_1])
    state_placeholder_all = ng.placeholder(state_axes + [batch_axis_all])

    model = make_model(action_axis)

    def make_function(placeholder):
        computation = ng.computation(model(placeholder), placeholder)

        transformer = ng.transformers.make_transformer()
        return transformer.add_computation(computation)

    # make a computation using the batch size of 32, though it doesn't need to be used
    make_function(state_placeholder_all)
    # now make another competition with the batch size of 1.
    # the convolution fails because part of the graph is thinking that the batch size
    # is 32 and part of the graph is thinking it is a batch size of 1
    make_function(state_placeholder)(np.zeros(state_placeholder.axes.lengths))
