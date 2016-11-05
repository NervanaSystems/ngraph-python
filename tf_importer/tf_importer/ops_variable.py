# ----------------------------------------------------------------------------
# Copyright 2016 Nervana Systems Inc.
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

from tf_importer.tf_importer.utils import tf_to_shape_axes
from tf_importer.tf_importer.ops_base import OpsBase
import ngraph as ng


class OpsVariable(OpsBase):
    """
    Mix-in class for variable op
    """

    def Variable(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        See the [Variables How To](../../how_tos/variables/index.md) for a high
        level overview.

        A variable maintains state in the graph across calls to `run()`. You add a
        variable to the graph by constructing an instance of the class `Variable`.

        The `Variable()` constructor requires an initial value for the variable,
        which can be a `Tensor` of any type and shape. The initial value defines the
        type and shape of the variable. After construction, the type and shape of
        the variable are fixed. The value can be changed using one of the assign
        methods.

        If you want to change the shape of a variable later you have to use an
        `assign` Op with `validate_shape=False`.

        Just like any `Tensor`, variables created with `Variable()` can be used as
        inputs for other Ops in the graph. Additionally, all the operators
        overloaded for the `Tensor` class are carried over to variables, so you can
        also add nodes to the graph by just doing arithmetic on variables.

        ```python
        import tensorflow as tf

        # Create a variable.
        w = tf.Variable(<initial-value>, name=<optional-name>)

        # Use the variable in the graph like any Tensor.
        y = tf.matmul(w, ...another variable or tensor...)

        # The overloaded operators are available too.
        z = tf.sigmoid(w + y)

        # Assign a new value to the variable with `assign()` or a related method.
        w.assign(w + 1.0)
        w.assign_add(1.0)
        ```

        When you launch the graph, variables have to be explicitly initialized before
        you can run Ops that use their value. You can initialize a variable by
        running its *initializer op*, restoring the variable from a save file, or
        simply running an `assign` Op that assigns a value to the variable. In fact,
        the variable *initializer op* is just an `assign` Op that assigns the
        variable's initial value to the variable itself.

        ```python
        # Launch the graph in a session.
        with tf.Session() as sess:
                # Run the variable initializer.
                sess.run(w.initializer)
                # ...you now can run ops that use the value of 'w'...
        ```

        The most common initialization pattern is to use the convenience function
        `initialize_all_variables()` to add an Op to the graph that initializes
        all the variables. You then run that Op after launching the graph.

        ```python
        # Add an Op to initialize all variables.
        init_op = tf.initialize_all_variables()

        # Launch the graph in a session.
        with tf.Session() as sess:
                # Run the Op that initializes all variables.
                sess.run(init_op)
                # ...you can now run any Op that uses variable values...
        ```

        If you need to create a variable with an initial value dependent on another
        variable, use the other variable's `initialized_value()`. This ensures that
        variables are initialized in the right order.

        All variables are automatically collected in the graph where they are
        created. By default, the constructor adds the new variable to the graph
        collection `GraphKeys.VARIABLES`. The convenience function
        `all_variables()` returns the contents of that collection.

        When building a machine learning model it is often convenient to distinguish
        betwen variables holding the trainable model parameters and other variables
        such as a `global step` variable used to count training steps. To make this
        easier, the variable constructor supports a `trainable=<bool>` parameter. If
        `True`, the new variable is also added to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES`. The convenience function
        `trainable_variables()` returns the contents of this collection. The
        various `Optimizer` classes use this collection as the default list of
        variables to optimize.


        Creating a variable.

        @@__init__
        @@initialized_value

        Changing a variable value.

        @@assign
        @@assign_add
        @@assign_sub
        @@scatter_sub
        @@count_up_to

        @@eval

        Properties.

        @@name
        @@dtype
        @@get_shape
        @@device
        @@initializer
        @@graph
        @@op
        """

        # get axes
        try:
            axes = tf_to_shape_axes(tf_node.attr['shape'])
        except:
            raise NotImplementedError('Shape must be know prior to execution')

        return ng.variable(axes=axes, name=tf_node.name)

    def Assign(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        def assign(ref, value, validate_shape=None, use_locking=None,
               name=None):
        Update 'ref' by assigning 'value' to it.

        This operation outputs "ref" after the assignment is done.
        This makes it easier to chain operations that need to use the reset value.

        Args:
            ref: A mutable `Tensor`.
                 Should be from a `Variable` node. May be uninitialized.
            value: A `Tensor`. Must have the same type as `ref`.
                   The value to be assigned to the variable.
            validate_shape: An optional `bool`. Defaults to `True`.
                            If true, the operation will validate that the shape
                            of 'value' matches the shape of the Tensor being assigned to. If false,
                            'ref' will take on the shape of 'value'.
            use_locking: An optional `bool`. Defaults to `True`.
                         If True, the assignment will be protected by a lock;
                         otherwise the behavior is undefined, but may exhibit less contention.
            name: A name for the operation (optional).

        Returns:
            Same as "ref". Returned as a convenience for operations that want
            to use the new value after the variable has been reset.
        """

        """
        TODO: currently cannot fully support the TensorFlow semantics.
        1. Assign in TF returns the assigned tensor, in ngraph, it returns
           None
        2. In TF, is the assigned tensor is not used, then it retain the
           original value
        """
        ref, value = inputs
        assert ref.axes.lengths == value.axes.lengths, "shape not the same"
        value = ng.cast_axes(value, ref.axes)

        if tf_node.name in self.init_assign_op_names:
            with ng.Op.saved_user_deps():
                return ng.assign(ref, value)
        else:
            return ng.assign(ref, value)

    def AssignAdd(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Update 'ref' by adding 'value' to it.

        This operation outputs "ref" after the update is done.
        This makes it easier to chain operations that need to use the reset value.

        Args:
            ref: A mutable `Tensor`. Must be one of the following types:
                 `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`,
                 `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`,
                 `qint32`, `half`. Should be from a `Variable` node.
            value: A `Tensor`. Must have the same type as `ref`.
                   The value to be added to the variable.
            use_locking: An optional `bool`. Defaults to `False`.
                         If True, the addition will be protected by a lock;
                         otherwise the behavior is undefined, but may exhibit less contention.
            name: A name for the operation (optional).

        Returns:
            Same as "ref". Returned as a convenience for operations that want
            to use the new value after the variable has been updated.
        """
        ref, value = inputs
        assert ref.axes.lengths == value.axes.lengths, "shape not the same"
        value = ng.cast_axes(value, ref.axes)

        if tf_node.name in self.init_assign_op_names:
            with ng.Op.saved_user_deps():
                return ng.assign(ref, ref + value)
        else:
            return ng.assign(ref, ref + value)

    def NoOp(self, tf_node, inputs):
        """
        [TensorFlow Docs]
        Does nothing. Only useful as a placeholder for control edges.

        Args:
            name: A name for the operation (optional).

        Returns:
            The created Operation.
        """
        # TODO remove hardcoded name by passing in names for op

        if tf_node.name == "init":
            return ng.doall(all=inputs)
        else:
            raise NotImplementedError
