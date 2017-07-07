import os

import ngraph as ng
import ngraph.op_graph.tensorboard.tensorboard as tb


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
    tf_graphdef = tb.ngraph_to_tf_graph_def(graph)
    tf_graphdef_2 = tb.ngraph_to_tf_graph_def([base, graph])
    assert len(tf_graphdef.node) == 5
    assert tf_graphdef == tf_graphdef_2


def test_tensorboard():
    _, graph = get_simple_graph()
    fname = tb.ngraph_to_tensorboard(graph)
    assert os.path.exists(fname)
