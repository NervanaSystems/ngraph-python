# ----------------------------------------------------------------------------
# copyright 2016 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import numpy as np

import ngraph as ng
import ngraph.transformers as ngt
from ngraph.transformers.passes.hetrpasses import DeviceAssignPass, \
    CommunicationPass, ChildTransformerPass


def check_result_values(input_vector, result_expected, placeholder, op_list=[], *args):

    """
    This function checks the result values return by the hetr computation object
    against the expected result values
    it also checks if the value returned by the hetr object matches the order in
    the expected result list

    :param: input_vector: list specifying the differnt values to be passed to
            the placeholder
    :param: result_expected: list of tuples specifying the expected result
            values from the hetr computation object
    :param: placeholder: list of placeholder to be passed for hetrcomputation
    :param: op_list: list of result handlers to be paased for hetrcomputation

    """
    # Select the transformer
    transformer = ngt.make_transformer_factory('hetr')()

    # Build the hetr computation object
    computation = transformer.computation(op_list, placeholder)
    result_obtained = []

    # Check for the return result list
    for i in input_vector:
        result_obtained.append(computation(i))

    # if return result is tuple
    if len(result_expected) > 1:
        np.testing.assert_array_equal(result_expected, result_obtained)

    # if return result is  scalar
    else:
        assert (np.array(tuple(result_obtained)) ==
                np.array(result_expected[0])).all()

    transformer.cleanup()


def check_device_assign_pass(default_device, default_device_id,
                             graph_op_metadata, graph_op=[], *args):
    """
    The Device assign pass should inject the metadata{device_id, device} as
    specified by the user for each op,
    if not specified then the default {device_id:0, device:numpy} should be
    inserted for each op.

    :param: default_device: string, the default device for each op,
            if not specified by user ex: "numpy"
    :param: default_device_id: string, the default device number for each op,
            if not specified by user ex: "0"
    :param: graph_op_metadata: dict, dictionary of list specifying  the expected
            metadata {device_id, device} for each op
    :param: graph_op: list of ops to do the graph traversal

    """
    transformers = set()
    expected_transformers = set()
    obj = DeviceAssignPass(default_device, default_device_id, transformers)
    obj.do_pass(graph_op, [])

    for op in graph_op_metadata.keys():
        assert op.metadata['device'] == graph_op_metadata[op][0]
        assert op.metadata['device_id'] == graph_op_metadata[op][1]
        assert op.metadata['transformer'] == graph_op_metadata[op][0] +  \
            str(graph_op_metadata[op][1])

        expected_transformers.add(op.metadata['transformer'])
    assert transformers == expected_transformers


def check_communication_pass(ops_to_transform, expected_recv_nodes):

    """
    The communication pass should insert send/recv nodes wherever
    the metadata[transformer] differs between nodes.
    This checks that the recv nodes are inserted in the right place, and counts
    that the expected number of send
    nodes are found.

    :param ops_to_transform: list of ops to do the garph traversal
    :param expected_recv_nodes: lits of ops where receive nodes are expected to
           be inserted after the communication pass

    """
    send_nodes = list()
    obj = CommunicationPass(send_nodes)
    obj.do_pass(ops_to_transform, [])

    op_list_instance_type = list()
    num_expected_sendnodes = len(expected_recv_nodes)

    # Count if the communication pass inserted the expected number of send nodes
    assert num_expected_sendnodes == len(send_nodes)

    # verify if Recv nodes are inserted in the right place
    for op in expected_recv_nodes:
        for each_arg in op.args:
            op_list_instance_type.append(type(each_arg))

        assert ng.op_graph.communication.Recv in op_list_instance_type
        del op_list_instance_type[:]


def test_hetr_graph_passes():

    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    y = ng.placeholder(())
    x_plus_y = x + y

    # Build the graph metadata
    graph_op_list = [x_plus_y, x, y]

    graph_op_metadata = {op: list() for op in graph_op_list}
    graph_op_metadata[x] = ["numpy", '1']
    graph_op_metadata[y] = ["numpy", '0']
    graph_op_metadata[x_plus_y] = ["numpy", '0']

    transformer_list = ["numpy1", "numpy0"]

    # Run the hetr passes one by one, and verify they did the expected things to the graph
    check_device_assign_pass("numpy", "0", graph_op_metadata, graph_op_list)
    check_communication_pass(ops_to_transform=graph_op_list,
                             expected_recv_nodes=[x_plus_y])

    # Check if the hetr pass (childTransfromer pass) generates the expected transformer list
    obj = ChildTransformerPass([])
    obj.do_pass(graph_op_list, [])
    assert set(transformer_list) == set(obj.transformer_list)


def test_simple_graph():

    # Build the graph
    with ng.metadata(device_id='1'):
        x = ng.placeholder(())

    x_plus_one = x + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(11, 21, 31)],
                        placeholder=x, op_list=[x_plus_one])

    x_plus_one = x + 1
    x_plus_two = x + 2
    x_mul_three = x * 3

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(11, 12, 30),
                                         (21, 22, 60),
                                         (31, 32, 90)],
                        placeholder=x,
                        op_list=[x_plus_one, x_plus_two, x_mul_three])


def test_gpu_send_and_recv():

    # put x+1 on cpu numpy
    with ng.metadata(device='numpy'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on gpu numpy
    with ng.metadata(device='gpu'):
        x_plus_two = x_plus_one + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(12), (22), (32)],
                        placeholder=x, op_list=[x_plus_two])

    # put x+1 on gpu numpy
    with ng.metadata(device='gpu'):
        x = ng.placeholder(())
        x_plus_one = x + 1
    # put x+2 on cpu numpy
    with ng.metadata(device='numpy'):
        x_plus_two = x_plus_one + 1

    check_result_values(input_vector=[10, 20, 30],
                        result_expected=[(12), (22), (32)],
                        placeholder=x, op_list=[x_plus_two])
