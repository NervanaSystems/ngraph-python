# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

from __future__ import print_function
from __future__ import division

import inspect

import cntk as C
import numpy as np

import ngraph as ng
from ngraph.frontends.cntk.cntk_importer.ops_bridge import OpsBridge


def cross_entropy_with_softmax(model, labels):
    """
    Auxiliary function to add cross entropy and softmax (loss function)
    to imported model for training.

    Arguments:
        model - imported CNTK model
        labels - placeholder for one-hot labels array

    Returns:
        Loss function (mean for batch)
    """
    if model.axes.lengths != labels.axes.lengths:
        model = ng.Transpose(model)
    assert model.axes.lengths == labels.axes.lengths
    model = ng.cast_axes(model, axes=labels.axes)

    loss = ng.cross_entropy_multi(ng.softmax(model), labels)
    return ng.mean(loss, out_axes=())


def classification_error(model, labels):
    """
    Auxiliary function to add classification error function to
    imported model for testing.

    Arguments:
        model - imported CNTK model
        labels - placeholder for one-hot labels array

    Returns:
        Classification error function (mean for batch)
    """
    if labels.axes.batch_axis():
        errors = ng.not_equal(
            ng.argmax(model, out_axes=[labels.axes.batch_axis()]),
            ng.argmax(labels, out_axes=[labels.axes.batch_axis()])
        )
    else:
        errors = ng.not_equal(ng.argmax(model), ng.argmax(labels))
    return ng.mean(errors, out_axes=())


class CNTKImporter:
    """
    Importer for CNTK graph's definition
    """

    def __init__(self, batch_size=1, debug=False):
        self.uid_op_map = dict()
        self.placeholders = []
        self.ops_bridge = OpsBridge()
        self.batch_size = batch_size
        self.debug = debug

    def load_operations(self, cntk_model):
        """
        Save CNTK graph's functions list in reverse (first to last) order.

        Arguments:
            cntk_model: CNTK network model (last operation).
        """
        stack = [cntk_model]
        visited = list()
        if self.debug:
            functions = set()

        while stack:
            node = stack.pop()
            node = node.root_function

            if node.uid in visited:
                continue

            if self.debug:
                functions.add(node.op_name)

            visited.append(node.uid)
            self.uid_op_map[node.uid] = node

            for i in node.inputs:
                if i.is_output:
                    stack.append(i.owner)

        if self.debug:
            print("Functions used in model: " + str(functions))
            print("All operations in model:")
            for i in visited:
                print("  " + i + "(" + self.uid_op_map[i].op_name + ")")
            print("")

    def import_operation(self, cntk_op):
        """
        Recursively import and translate CNTK operations.

        Arguments:
            cntk_op: CNTK operation to be imported.

        Returns:
            Translated operation.
        """
        if self.debug:
            for _ in range(len(inspect.stack())):
                print(' ', end="")
            print("Importing: " + cntk_op.uid + "(", end="")
            for i in cntk_op.inputs:
                print(i.uid + str(i.shape) + ",", end="")
            print(")")

        inputs = []
        for i in cntk_op.inputs:
            axes = [
                ng.make_axis(dim) for dim in i.shape
            ]
            dtype = np.dtype(i.dtype)

            if i.is_output:
                uid = i.owner.root_function.uid
                temp = self.uid_op_map[uid]
                if isinstance(temp, C.Function):
                    temp = self.import_operation(temp)
                    if temp is None:
                        raise ValueError("Error translating: " + uid)
                    else:
                        if self.debug:
                            for _ in range(len(inspect.stack()) + 1):
                                print(' ', end="")
                            print("Finished importing: " +
                                  uid + str(cntk_op.shape) + " -> " +
                                  temp.name + str(temp.shape.full_lengths))
                        if len(temp.axes) == 1:
                            if temp.axes[0].length == 1:
                                temp = ng.sum(temp, out_axes=())
                        self.uid_op_map[uid] = temp
                inputs.append(temp)
            elif i.is_input:
                if self.batch_size > 1:
                    axes.append(ng.make_axis(self.batch_size, 'N'))
                temp = ng.placeholder(axes, dtype).named(i.uid)
                inputs.append(temp)
                self.placeholders.append(temp)
            else:
                try:
                    input_value = i.value
                except AttributeError:
                    input_value = C.plus(i, np.zeros(i.shape)).eval()
                if i.is_constant:
                    inputs.append(ng.constant(input_value, axes, dtype))
                elif i.is_parameter:
                    inputs.append(ng.variable(axes, dtype, input_value).named(i.uid))
                else:
                    raise ValueError("Unknown input: " + i.uid)

        return self.ops_bridge(cntk_op, inputs)

    def import_model(self, cntk_model):
        """
        Import and translate CNTK network model to ngraph network.

        Arguments:
            cntk_model: Model (with inputs) to be translated to ngraph.

        Returns:
            Translated model - last ngraph operation.
            List of placeholders.
        """
        self.load_operations(cntk_model)

        temp = self.import_operation(cntk_model.root_function)
        if temp is None:
            raise ValueError("Error translating: " + cntk_model.root_function.uid)
        else:
            if self.debug:
                for _ in range(len(inspect.stack()) + 1):
                    print(' ', end="")
                print("Finished importing: " +
                      cntk_model.root_function.uid + str(cntk_model.root_function.shape) + " -> " +
                      temp.name + str(temp.shape.full_lengths))
                print("")
            self.uid_op_map[cntk_model.root_function.uid] = temp

        return temp, self.placeholders
