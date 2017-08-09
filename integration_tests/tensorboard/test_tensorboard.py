import numpy as np
import ngraph as ng
from ngraph.op_graph.tensorboard.tensorboard import Tensorboard, ngraph_to_tf_graph_def


tb = Tensorboard("/tmp/test_tensorboard_integration")


def get_simple_graph():
    ax = ng.make_axes([ng.make_axis(name='C', length=1)])
    base_op = ng.constant(5.0, ax)
    simple_graph = ng.log(ng.exp(base_op))
    return base_op, simple_graph


# These tests are intentionally light because we don't want to test specifics
# of TensorFlow's internal `GraphDef` or `Record` representations and then
# have our tests fail when these internal representations change.

def test_graph_def_conversion():
    base, graph = get_simple_graph()
    tf_graphdef = ngraph_to_tf_graph_def(graph)
    tf_graphdef_2 = ngraph_to_tf_graph_def([base, graph])
    assert len(tf_graphdef.node) == 5
    assert tf_graphdef == tf_graphdef_2


def plot_tensorboard_graph():
    _, graph = get_simple_graph()
    tb.add_graph(graph)


def plot_tensorboard_scalar():

    sine = np.sin(np.arange(200) * 2 * np.pi / 50)
    for ii, value in enumerate(sine):
        tb.add_scalar("Sin", value, step=ii)


def plot_tensorboard_histogram():

    gaussian = np.random.normal(0, 1, (10000,))
    uniform = np.random.uniform(0, 1, (10000,))
    tb.add_histogram("Gaussian", gaussian, step=0)
    tb.add_histogram("Uniform", uniform, step=0)


def plot_tensorboard_image():

    rgb = np.random.uniform(0, 255, (64, 64, 3)).astype("uint8")
    bw = np.random.uniform(0, 255, (64, 64, 1)).astype("uint8")
    tb.add_image("RGB", rgb)
    tb.add_image("BW", bw)


def plot_tensorboard_audio():

    def normalize(arr):
        return (arr / np.abs(arr).max() * (2 ** 15 - 1)).astype("int16")

    sample_rate = 16000.
    white_noise = normalize(np.random.normal(0, 1, (32000, 1)))
    sine_wave = normalize(np.sin(np.arange(32000) / sample_rate * 2 * np.pi * 100))
    tb.add_audio("WhiteNoise", white_noise, sample_rate)
    tb.add_audio("100HzSin", sine_wave, sample_rate)
