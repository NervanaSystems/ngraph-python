#  Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# flake8: noqa

class AggregationMethod(object):
    """
    [TensorFlow Docs]
    A class listing aggregation methods used to combine gradients.

    Computing partial derivatives can require aggregating gradient
    contributions. This class lists the various methods that can
    be used to combine gradients in the graph:

    *  `ADD_N`: All of the gradient terms are summed as part of one
     operation using the "AddN" op. It has the property that all
     gradients must be ready before any aggregation is performed.
    *  `DEFAULT`: The system-chosen default aggregation method.
    """
    pass


def Assert(condition, data, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Asserts that the given condition is true.

    If `condition` evaluates to false, print the list of tensors in `data`.
    `summarize` determines how many entries of the tensors to print.

    NOTE: To ensure that Assert executes, one usually attaches a dependency:

    ```python
   # Ensure maximum element of x is smaller or equal to 1
    assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
    x = tf.with_dependencies([assert_op], x)
    ```

    Args:
        condition: The condition to evaluate.
        data: The tensors to print out when condition is false.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional).

    Returns:
        assert_op: An `Operation` that, when executed, raises a
        `tf.errors.InvalidArgumentError` if `condition` is not true.
    """
    pass


class DType(object):
    """
    [TensorFlow Docs]
    Represents the type of the elements in a `Tensor`.

    The following `DType` objects are defined:

    * `tf.float16`: 16-bit half-precision floating-point.
    * `tf.float32`: 32-bit single-precision floating-point.
    * `tf.float64`: 64-bit double-precision floating-point.
    * `tf.bfloat16`: 16-bit truncated floating-point.
    * `tf.complex64`: 64-bit single-precision complex.
    * `tf.complex128`: 128-bit double-precision complex.

    * `tf.int8`: 8-bit signed integer.
    * `tf.uint8`: 8-bit unsigned integer.
    * `tf.uint16`: 16-bit unsigned integer.
    * `tf.int16`: 16-bit signed integer.
    * `tf.int32`: 32-bit signed integer.
    * `tf.int64`: 64-bit signed integer.

    * `tf.bool`: Boolean.

    * `tf.string`: String.

    * `tf.qint8`: Quantized 8-bit signed integer.
    * `tf.quint8`: Quantized 8-bit unsigned integer.
    * `tf.qint16`: Quantized 16-bit signed integer.
    * `tf.quint16`: Quantized 16-bit unsigned integer.
    * `tf.qint32`: Quantized 32-bit signed integer.

    In addition, variants of these types with the `_ref` suffix are
    defined for reference-typed tensors.

    The `tf.as_dtype()` function converts numpy types and string type
    names to a `DType` object.

    @@is_compatible_with
    @@name
    @@base_dtype
    @@real_dtype
    @@is_ref_dtype
    @@as_ref
    @@is_floating
    @@is_complex
    @@is_integer
    @@is_quantized
    @@is_unsigned

    @@as_numpy_dtype
    @@as_datatype_enum
    """
    pass


class DeviceSpec(object):
    """
    [TensorFlow Docs]
    Represents a (possibly partial) specification for a TensorFlow device.

    `DeviceSpec`s are used throughout TensorFlow to describe where state is stored
    and computations occur. Using `DeviceSpec` allows you to parse device spec
    strings to verify their validity, merge them or compose them programmatically.

    Example:
    ```python
    # Place the operations on device "GPU:0" in the "ps" job.
    device_spec = DeviceSpec(job="ps", device_type="GPU", device_index=0)
    with tf.device(device_spec):
        # Both my_var and squared_var will be placed on /job:ps/device:GPU:0.
        my_var = tf.Variable(..., name="my_variable")
        squared_var = tf.square(my_var)
    ```

    If a `DeviceSpec` is partially specified, it will be merged with other
    `DeviceSpec`s according to the scope in which it is defined. `DeviceSpec`
    components defined in inner scopes take precedence over those defined in
    outer scopes.

    ```python
    with tf.device(DeviceSpec(job="train", )):
        with tf.device(DeviceSpec(job="ps", device_type="GPU", device_index=0):
            # Nodes created here will be assigned to /job:ps/device:GPU:0.
        with tf.device(DeviceSpec(device_type="GPU", device_index=1):
            # Nodes created here will be assigned to /job:train/device:GPU:1.
    ```

    A `DeviceSpec` consists of 5 components -- each of
    which is optionally specified:

    * Job: The job name.
    * Replica: The replica index.
    * Task: The task index.
    * Device type: The device type string (e.g. "CPU" or "GPU").
    * Device index: The device index.
    """
    pass


class Dimension(object):
    """
    [TensorFlow Docs]
    Represents the value of one dimension in a TensorShape."""
    pass


class FIFOQueue(QueueBase):
    """
    [TensorFlow Docs]
    A queue implementation that dequeues elements in first-in-first out order.

    See [`tf.QueueBase`](#QueueBase) for a description of the methods on
    this class.

    @@__init__
    """
    pass


class FixedLenFeature(collections.namedtuple(
    "FixedLenFeature", ["shape", "dtype", "default_value"])):
    """
    [TensorFlow Docs]
    Configuration for parsing a fixed-length input feature.

    To treat sparse input as dense, provide a `default_value`; otherwise,
    the parse functions will fail on any examples missing this feature.

    Fields:
        shape: Shape of input data.
        dtype: Data type of input.
        default_value: Value to be used if an example is missing this feature. It
                       must be compatible with `dtype`.
                       """
    pass


class FixedLenSequenceFeature(collections.namedtuple(
    "FixedLenSequenceFeature", ["shape", "dtype", "allow_missing"])):
    """
    [TensorFlow Docs]
    Configuration for a dense input feature in a sequence item.

    To treat a sparse input as dense, provide `allow_missing=True`; otherwise,
    the parse functions will fail on any examples missing this feature.

    Fields:
        shape: Shape of input data.
        dtype: Data type of input.
               allow_missing: Whether to allow this feature to be missing from a feature
               list item.
               """
    pass


class FixedLengthRecordReader(ReaderBase):
    """
    [TensorFlow Docs]
    A Reader that outputs fixed-length records from a file.

    See ReaderBase for supported methods.
    """
    pass


class Graph(object):
    """
    [TensorFlow Docs]
    A TensorFlow computation, represented as a dataflow graph.

    A `Graph` contains a set of
    [`Operation`](../../api_docs/python/framework.md#Operation) objects,
    which represent units of computation; and
    [`Tensor`](../../api_docs/python/framework.md#Tensor) objects, which represent
    the units of data that flow between operations.

    A default `Graph` is always registered, and accessible by calling
    [`tf.get_default_graph()`](../../api_docs/python/framework.md#get_default_graph).
    To add an operation to the default graph, simply call one of the functions
    that defines a new `Operation`:

    ```
    c = tf.constant(4.0)
    assert c.graph is tf.get_default_graph()
    ```

    Another typical usage involves the
    [`Graph.as_default()`](../../api_docs/python/framework.md#Graph.as_default)
    context manager, which overrides the current default graph for the
    lifetime of the context:

    ```python
    g = tf.Graph()
    with g.as_default():
        # Define operations and tensors in `g`.
        c = tf.constant(30.0)
        assert c.graph is g
    ```

    Important note: This class *is not* thread-safe for graph construction. All
    operations should be created from a single thread, or external
    synchronization must be provided. Unless otherwise specified, all methods
    are not thread-safe.

    @@__init__
    @@as_default
    @@as_graph_def
    @@finalize
    @@finalized

    @@control_dependencies
    @@device
    @@name_scope

    A `Graph` instance supports an arbitrary number of "collections"
    that are identified by name. For convenience when building a large
    graph, collections can store groups of related objects: for
    example, the `tf.Variable` uses a collection (named
    [`tf.GraphKeys.VARIABLES`](../../api_docs/python/framework.md#GraphKeys)) for
    all variables that are created during the construction of a graph. The caller
    may define additional collections by specifying a new name.

    @@add_to_collection
    @@add_to_collections
    @@get_collection
    @@get_collection_ref

    @@as_graph_element
    @@get_operation_by_name
    @@get_tensor_by_name
    @@get_operations

    @@seed
    @@unique_name
    @@version
    @@graph_def_versions

    @@create_op
    @@gradient_override_map
    """
    pass


class GraphKeys(object):
    """
    [TensorFlow Docs]
    Standard names to use for graph collections.

    The standard library uses various well-known names to collect and
    retrieve values associated with a graph. For example, the
    `tf.Optimizer` subclasses default to optimizing the variables
    collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
    specified, but it is also possible to pass an explicit list of
    variables.

    The following standard keys are defined:

    * `VARIABLES`: the `Variable` objects that comprise a model, and
        must be saved and restored together. See
        [`tf.all_variables()`](../../api_docs/python/state_ops.md#all_variables)
        for more details.
    * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
        be trained by an optimizer. See
        [`tf.trainable_variables()`](../../api_docs/python/state_ops.md#trainable_variables)
        for more details.
    * `SUMMARIES`: the summary `Tensor` objects that have been created in the
        graph. See
        [`tf.merge_all_summaries()`](../../api_docs/python/train.md#merge_all_summaries)
        for more details.
    * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
        produce input for a computation. See
        [`tf.start_queue_runners()`](../../api_docs/python/train.md#start_queue_runners)
        for more details.
    * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
        keep moving averages. See
        [`tf.moving_average_variables()`](../../api_docs/python/state_ops.md#moving_average_variables)
        for more details.
    * `REGULARIZATION_LOSSES`: regularization losses collected during graph
        construction.
    * `WEIGHTS`: weights inside neural network layers
    * `BIASES`: biases inside neural network layers
    * `ACTIVATIONS`: activations of neural network layers
    """
    pass


class IdentityReader(ReaderBase):
    """
    [TensorFlow Docs]
    A Reader that outputs the queued work as both the key and value.

    To use, enqueue strings in a Queue. Read will take the front
    work string and output (work, work).

    See ReaderBase for supported methods.
    """
    pass


class IndexedSlices(object):
    """
    [TensorFlow Docs]
    A sparse representation of a set of tensor slices at given indices.

    This class is a simple wrapper for a pair of `Tensor` objects:

    * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
    * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

    An `IndexedSlices` is typically used to represent a subset of a larger
    tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
    The values in `indices` are the indices in the first dimension of
    the slices that have been extracted from the larger tensor.

    The dense tensor `dense` represented by an `IndexedSlices` `slices` has

    ```python
    dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
    ```

    The `IndexedSlices` class is used principally in the definition of
    gradients for operations that have sparse gradients
    (e.g. [`tf.gather`](../../api_docs/python/array_ops.md#gather)).

    Contrast this representation with
    [`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
    which uses multi-dimensional indices and scalar values.

    @@__init__

    @@values
    @@indices
    @@dense_shape

    @@name
    @@dtype
    @@device
    @@op
    """
    pass


class InteractiveSession(BaseSession):
    """
    [TensorFlow Docs]
    A TensorFlow `Session` for use in interactive contexts, such as a shell.

    The only difference with a regular `Session` is that an `InteractiveSession`
    installs itself as the default session on construction.
    The methods [`Tensor.eval()`](../../api_docs/python/framework.md#Tensor.eval)
    and [`Operation.run()`](../../api_docs/python/framework.md#Operation.run)
    will use that session to run ops.

    This is convenient in interactive shells and [IPython
    notebooks](http://ipython.org), as it avoids having to pass an explicit
    `Session` object to run ops.

    For example:

    ```python
    sess = tf.InteractiveSession()
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b
    # We can just use 'c.eval()' without passing 'sess'
    print(c.eval())
    sess.close()
    ```

    Note that a regular session installs itself as the default session when it
    is created in a `with` statement. The common usage in non-interactive
    programs is to follow that pattern:

    ```python
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b
    with tf.Session():
        # We can also use 'c.eval()' here.
        print(c.eval())
    ```

    @@__init__
    @@close
    """
    pass


def NoGradient(op_type):
    """
    [TensorFlow Docs]
    Specifies that ops of type `op_type` do not have a defined gradient.

    This function is only used when defining a new op type. It may be
    used for ops such as `tf.size()` that are not differentiable. For
    example:

    ```python
    tf.NoGradient("Size")
    ```

    Args:
        op_type: The string type of an operation. This corresponds to the
                 `OpDef.name` field for the proto that defines the operation.

    Raises:
        TypeError: If `op_type` is not a string.

    """
    pass


class OpError(Exception):
    """
    [TensorFlow Docs]
    A generic error that is raised when TensorFlow execution fails.

    Whenever possible, the session will raise a more specific subclass
    of `OpError` from the `tf.errors` module.

    @@op
    @@node_def
    """
    pass


class Operation(object):
    """
    [TensorFlow Docs]
    Represents a graph node that performs computation on tensors.

    An `Operation` is a node in a TensorFlow `Graph` that takes zero or
    more `Tensor` objects as input, and produces zero or more `Tensor`
    objects as output. Objects of type `Operation` are created by
    calling a Python op constructor (such as
    [`tf.matmul()`](../../api_docs/python/math_ops.md#matmul))
    or [`Graph.create_op()`](../../api_docs/python/framework.md#Graph.create_op).

    For example `c = tf.matmul(a, b)` creates an `Operation` of type
    "MatMul" that takes tensors `a` and `b` as input, and produces `c`
    as output.

    After the graph has been launched in a session, an `Operation` can
    be executed by passing it to
    [`Session.run()`](../../api_docs/python/client.md#Session.run).
    `op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.

    @@name
    @@type
    @@inputs
    @@control_inputs
    @@outputs
    @@device
    @@graph

    @@run

    @@get_attr
    @@traceback
    """
    pass


class PaddingFIFOQueue(QueueBase):
    """
    [TensorFlow Docs]
    A FIFOQueue that supports batching variable-sized tensors by padding.

    A `PaddingFIFOQueue` may contain components with dynamic shape, while also
    supporting `dequeue_many`. See the constructor for more details.

    See [`tf.QueueBase`](#QueueBase) for a description of the methods on
    this class.

    @@__init__
    """
    pass


def Print(input_, data, message=None, first_n=None, summarize=None,
          name=None):
    """
    [TensorFlow Docs]
    Prints a list of tensors.

    This is an identity op with the side effect of printing `data` when
    evaluating.

    Args:
        input_: A tensor passed through this op.
        data: A list of tensors to print out when op is evaluated.
        message: A string, prefix of the error message.
                 first_n: Only log `first_n` number of times. Negative numbers log always;
                 this is the default.
        summarize: Only print this many entries of each tensor. If None, then a
                   maximum of 3 elements are printed per input tensor.
        name: A name for the operation (optional).

    Returns:
        Same tensor as `input_`.
    """
    pass


class QueueBase(object):
    """
    [TensorFlow Docs]
    Base class for queue implementations.

    A queue is a TensorFlow data structure that stores tensors across
    multiple steps, and exposes operations that enqueue and dequeue
    tensors.

    Each queue element is a tuple of one or more tensors, where each
    tuple component has a static dtype, and may have a static shape. The
    queue implementations support versions of enqueue and dequeue that
    handle single elements, versions that support enqueuing and
    dequeuing a batch of elements at once.

    See [`tf.FIFOQueue`](#FIFOQueue) and
    [`tf.RandomShuffleQueue`](#RandomShuffleQueue) for concrete
    implementations of this class, and instructions on how to create
    them.

    @@enqueue
    @@enqueue_many

    @@dequeue
    @@dequeue_many

    @@size

    @@close

    """
    pass


class RandomShuffleQueue(QueueBase):
    """
    [TensorFlow Docs]
    A queue implementation that dequeues elements in a random order.

    See [`tf.QueueBase`](#QueueBase) for a description of the methods on
    this class.

    @@__init__
    """
    pass


class ReaderBase(object):
    """
    [TensorFlow Docs]
    Base class for different Reader types, that produce a record every step.

    Conceptually, Readers convert string 'work units' into records (key,
    value pairs). Typically the 'work units' are filenames and the
    records are extracted from the contents of those files. We want a
    single record produced per step, but a work unit can correspond to
    many records.

    Therefore we introduce some decoupling using a queue. The queue
    contains the work units and the Reader dequeues from the queue when
    it is asked to produce a record (via Read()) but it has finished the
    last work unit.
    """
    pass


class RegisterGradient(object):
    """
    [TensorFlow Docs]
    A decorator for registering the gradient function for an op type.

    This decorator is only used when defining a new op type. For an op
    with `m` inputs and `n` outputs, the gradient function is a function
    that takes the original `Operation` and `n` `Tensor` objects
    (representing the gradients with respect to each output of the op),
    and returns `m` `Tensor` objects (representing the partial gradients
    with respect to each input of the op).

    For example, assuming that operations of type `"Sub"` take two
    inputs `x` and `y`, and return a single output `x - y`, the
    following gradient function would be registered:

    ```python
    @tf.RegisterGradient("Sub")
    def _sub_grad(unused_op, grad):
        return grad, tf.neg(grad)
    ```

    The decorator argument `op_type` is the string type of an
    operation. This corresponds to the `OpDef.name` field for the proto
    that defines the operation.

    @@__init__
    """
    pass


class RegisterShape(object):
    """
    [TensorFlow Docs]
    A decorator for registering the shape function for an op type.

    This decorator is only used when defining a new op type. A shape
    function is a function from an `Operation` object to a list of
    `TensorShape` objects, with one `TensorShape` for each output of the
    operation.

    For example, assuming that operations of type `"Sub"` take two
    inputs `x` and `y`, and return a single output `x - y`, all with the
    same shape, the following shape function would be registered:

    ```python
    @tf.RegisterShape("Sub")
    def _sub_shape(op):
        return [op.inputs[0].get_shape().merge_with(op.inputs[1].get_shape())]
    ```

    The decorator argument `op_type` is the string type of an
    operation. This corresponds to the `OpDef.name` field for the proto
    that defines the operation.

    """
    pass


class Session(BaseSession):
    """
    [TensorFlow Docs]
    A class for running TensorFlow operations.

    A `Session` object encapsulates the environment in which `Operation`
    objects are executed, and `Tensor` objects are evaluated. For
    example:

    ```python
    # Build a graph.
    a = tf.constant(5.0)
    b = tf.constant(6.0)
    c = a * b

    # Launch the graph in a session.
    sess = tf.Session()

    # Evaluate the tensor `c`.
    print(sess.run(c))
    ```

    A session may own resources, such as
    [variables](../../api_docs/python/state_ops.md#Variable), [queues](../../api_docs/python/io_ops.md#QueueBase),
    and [readers](../../api_docs/python/io_ops.md#ReaderBase). It is important to release
    these resources when they are no longer required. To do this, either
    invoke the [`close()`](#Session.close) method on the session, or use
    the session as a context manager. The following two examples are
    equivalent:

    ```python
    # Using the `close()` method.
    sess = tf.Session()
    sess.run(...)
    sess.close()

    # Using the context manager.
    with tf.Session() as sess:
        sess.run(...)
    ```

    The [`ConfigProto`]
    (https://www.tensorflow.org/code/tensorflow/core/protobuf/config.proto)
    protocol buffer exposes various configuration options for a
    session. For example, to create a session that uses soft constraints
    for device placement, and log the resulting placement decisions,
    create a session as follows:

    ```python
    # Launch the graph in a session that allows soft device placement and
    # logs the placement decisions.
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=True))
    ```

    @@__init__
    @@run
    @@close

    @@graph

    @@as_default

    """
    pass


class SparseTensor(object):
    """
    [TensorFlow Docs]
    Represents a sparse tensor.

    Tensorflow represents a sparse tensor as three separate dense tensors:
    `indices`, `values`, and `shape`. In Python, the three tensors are
    collected into a `SparseTensor` class for ease of use. If you have separate
    `indices`, `values`, and `shape` tensors, wrap them in a `SparseTensor`
    object before passing to the ops below.

    Concretely, the sparse tensor `SparseTensor(indices, values, shape)` is

    * `indices`: A 2-D int64 tensor of shape `[N, ndims]`.
    * `values`: A 1-D tensor of any type and shape `[N]`.
    * `shape`: A 1-D int64 tensor of shape `[ndims]`.

    where `N` and `ndims` are the number of values, and number of dimensions in
    the `SparseTensor` respectively.

    The corresponding dense tensor satisfies

    ```python
    dense.shape = shape
    dense[tuple(indices[i])] = values[i]
    ```

    By convention, `indices` should be sorted in row-major order (or equivalently
    lexicographic order on the tuples `indices[i]`). This is not enforced when
    `SparseTensor` objects are constructed, but most ops assume correct ordering.
    If the ordering of sparse tensor `st` is wrong, a fixed version can be
    obtained by calling `tf.sparse_reorder(st)`.

    Example: The sparse tensor

    ```python
    SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], shape=[3, 4])
    ```

    represents the dense tensor

    ```python
    [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
    ```

    @@__init__
    @@indices
    @@values
    @@shape
    @@dtype
    @@op
    @@graph
    """
    pass


class TFRecordReader(ReaderBase):
    """
    [TensorFlow Docs]
    A Reader that outputs the records from a TFRecords file.

    See ReaderBase for supported methods.
    """
    pass


class Tensor(object):
    """
    [TensorFlow Docs]
    Represents a value produced by an `Operation`.

    A `Tensor` is a symbolic handle to one of the outputs of an
    `Operation`. It does not hold the values of that operation's output,
    but instead provides a means of computing those values in a
    TensorFlow [`Session`](../../api_docs/python/client.md#Session).

    This class has two primary purposes:

    1. A `Tensor` can be passed as an input to another `Operation`.
     This builds a dataflow connection between operations, which
     enables TensorFlow to execute an entire `Graph` that represents a
     large, multi-step computation.

    2. After the graph has been launched in a session, the value of the
     `Tensor` can be computed by passing it to
     [`Session.run()`](../../api_docs/python/client.md#Session.run).
     `t.eval()` is a shortcut for calling
     `tf.get_default_session().run(t)`.

    In the following example, `c`, `d`, and `e` are symbolic `Tensor`
    objects, whereas `result` is a numpy array that stores a concrete
    value:

    ```python
    # Build a dataflow graph.
    c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    e = tf.matmul(c, d)

    # Construct a `Session` to execute the graph.
    sess = tf.Session()

    # Execute the graph and store the value that `e` represents in `result`.
    result = sess.run(e)
    ```

    @@dtype
    @@name
    @@value_index
    @@graph
    @@op
    @@consumers

    @@eval

    @@get_shape
    @@set_shape

    """
    pass


class TensorArray(object):
    """
    [TensorFlow Docs]
    Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.

    This class is meant to be used with dynamic iteration primitives such as
    `while_loop` and `map_fn`. It supports gradient back-propagation via special
    "flow" control flow dependencies.

    @@handle
    @@flow

    @@read
    @@unpack
    @@split

    @@write
    @@pack
    @@concat

    @@grad
    """
    pass


class TensorShape(object):
    """
    [TensorFlow Docs]
    Represents the shape of a `Tensor`.

    A `TensorShape` represents a possibly-partial shape specification for a
    `Tensor`. It may be one of the following:

    * *Fully-known shape:* has a known number of dimensions and a known size
        for each dimension.
    * *Partially-known shape:* has a known number of dimensions, and an unknown
        size for one or more dimension.
    * *Unknown shape:* has an unknown number of dimensions, and an unknown
        size in all dimensions.

    If a tensor is produced by an operation of type `"Foo"`, its shape
    may be inferred if there is a registered shape function for
    `"Foo"`. See [`tf.RegisterShape()`](../../api_docs/python/framework.md#RegisterShape)
    for details of shape
    functions and how to register them. Alternatively, the shape may be set
    explicitly using [`Tensor.set_shape()`](../../api_docs/python/framework.md#Tensor.set_shape).

    @@merge_with
    @@concatenate

    @@ndims
    @@dims
    @@as_list
    @@as_proto
    @@is_compatible_with
    @@is_fully_defined

    @@with_rank
    @@with_rank_at_least
    @@with_rank_at_most

    @@assert_has_rank
    @@assert_same_rank
    @@assert_is_compatible_with
    @@assert_is_fully_defined
    """
    pass


class TextLineReader(ReaderBase):
    """
    [TensorFlow Docs]
    A Reader that outputs the lines of a file delimited by newlines.

    Newlines are stripped from the output.
    See ReaderBase for supported methods.
    """
    pass


class VarLenFeature(collections.namedtuple("VarLenFeature", ["dtype"])):
    """
    [TensorFlow Docs]
    Configuration for parsing a variable-length input feature.

    Fields:
        dtype: Data type of input.
               """
    pass


class Variable(object):
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
    pass


class VariableScope(object):
    """
    [TensorFlow Docs]
    Variable scope object to carry defaults to provide to get_variable.

    Many of the arguments we need for get_variable in a variable store are most
    easily handled with a context. This object is used for the defaults.

    Attributes:
        name: name of the current scope, used as prefix in get_variable.
        initializer: default initializer passed to get_variable.
                     regularizer: default regularizer passed to get_variable.
                     reuse: Boolean or None, setting the reuse in get_variable.
                     caching_device: string, callable, or None: the caching device passed to
                     get_variable.
                     partitioner: callable or `None`: the partitioner passed to `get_variable`.
                     name_scope: The name passed to `tf.name_scope`.
                     """
    pass


class WholeFileReader(ReaderBase):
    """
    [TensorFlow Docs]
    A Reader that outputs the entire contents of a file as a value.

    To use, enqueue filenames in a Queue. The output of Read will
    be a filename (key) and the contents of that file (value).

    See ReaderBase for supported methods.
    """
    pass


def abs(x, name=None):
    """
    [TensorFlow Docs]
    Computes the absolute value of a tensor.

    Given a tensor of real numbers `x`, this operation returns a tensor
    containing the absolute value of each element in `x`. For example, if x is
    an input element and y is an output element, this operation computes
    \\\\(y = |x|\\\\).

    See [`tf.complex_abs()`](#tf_complex_abs) to compute the absolute value of a complex
    number.

    Args:
        x: A `Tensor` of type `float`, `double`, `int32`, or `int64`.
        name: A name for the operation (optional).

    Returns:
     A `Tensor` the same size and type as `x` with absolute values.
    """
    pass


def accumulate_n(inputs, shape=None, tensor_dtype=None, name=None):
    """
    [TensorFlow Docs]
    Returns the element-wise sum of a list of tensors.

    Optionally, pass `shape` and `tensor_dtype` for shape and type checking,
    otherwise, these are inferred.

    For example:

    ```python
    # tensor 'a' is [[1, 2], [3, 4]]
    # tensor `b` is [[5, 0], [0, 6]]
    tf.accumulate_n([a, b, a]) ==> [[7, 4], [6, 14]]

    # Explicitly pass shape and type
    tf.accumulate_n([a, b, a], shape=[2, 2], tensor_dtype=tf.int32)
        ==> [[7, 4], [6, 14]]
    ```

    Args:
        inputs: A list of `Tensor` objects, each with same shape and type.
        shape: Shape of elements of `inputs`.
               tensor_dtype: The type of `inputs`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of same shape and type as the elements of `inputs`.

    Raises:
        ValueError: If `inputs` don't all have same shape and dtype or the shape
        cannot be inferred.
    """
    pass


def acos(x, name=None):
    """
    [TensorFlow Docs]
    Computes acos of x element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def add(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns x + y element-wise.

    *NOTE*: Add supports broadcasting. AddN does not.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`, `string`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def add_check_numerics_ops():
    """
    [TensorFlow Docs]
    Connect a `check_numerics` to every floating point tensor.

    `check_numerics` operations themselves are added for each `float` or `double`
    tensor in the graph. For all ops in the graph, the `check_numerics` op for
    all of its (`float` or `double`) inputs is guaranteed to run before the
    `check_numerics` op on any of its outputs.

    Returns:
        A `group` op depending on all `check_numerics` ops added.
    """
    pass


def add_n(inputs, name=None):
    """
    [TensorFlow Docs]
    Add all input tensors element wise.

    Args:
        inputs: A list of at least 1 `Tensor` objects of the same type in: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
                Must all be the same size and shape.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `inputs`.
    """
    pass


def add_to_collection(name, value):
    """
    [TensorFlow Docs]
    Wrapper for `Graph.add_to_collection()` using the default graph.

    See [`Graph.add_to_collection()`](../../api_docs/python/framework.md#Graph.add_to_collection)
    for more details.

    Args:
        name: The key for the collection. For example, the `GraphKeys` class
              contains many standard names for collections.
        value: The value to add to the collection.
               """
    pass


def all_variables():
    """
    [TensorFlow Docs]
    Returns all variables that must be saved/restored.

    The `Variable()` constructor automatically adds new variables to the graph
    collection `GraphKeys.VARIABLES`. This convenience function returns the
    contents of that collection.

    Returns:
        A list of `Variable` objects.
    """
    pass


def arg_max(input, dimension, name=None):
    """
    [TensorFlow Docs]
    Returns the index with the largest value across dimensions of a tensor.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        dimension: A `Tensor` of type `int32`.
                   int32, 0 <= dimension < rank(input). Describes which dimension
                   of the input Tensor to reduce across. For vectors, use dimension = 0.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
    """
    pass


def arg_min(input, dimension, name=None):
    """
    [TensorFlow Docs]
    Returns the index with the smallest value across dimensions of a tensor.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        dimension: A `Tensor` of type `int32`.
                   int32, 0 <= dimension < rank(input). Describes which dimension
                   of the input Tensor to reduce across. For vectors, use dimension = 0.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
    """
    pass


def arg_max(input, dimension, name=None):
    """
    [TensorFlow Docs]
    Returns the index with the largest value across dimensions of a tensor.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        dimension: A `Tensor` of type `int32`.
                   int32, 0 <= dimension < rank(input). Describes which dimension
                   of the input Tensor to reduce across. For vectors, use dimension = 0.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
    """
    pass


def arg_min(input, dimension, name=None):
    """
    [TensorFlow Docs]
    Returns the index with the smallest value across dimensions of a tensor.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        dimension: A `Tensor` of type `int32`.
                   int32, 0 <= dimension < rank(input). Describes which dimension
                   of the input Tensor to reduce across. For vectors, use dimension = 0.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
    """
    pass


def as_dtype(type_value):
    """
    [TensorFlow Docs]
    Converts the given `type_value` to a `DType`.

    Args:
        type_value: A value that can be converted to a `tf.DType`
                    object. This may currently be a `tf.DType` object, a
                    [`DataType` enum](https://www.tensorflow.org/code/tensorflow/core/framework/types.proto),
                    a string type name, or a `numpy.dtype`.

    Returns:
        A `DType` corresponding to `type_value`.

    Raises:
        TypeError: If `type_value` cannot be converted to a `DType`.
    """
    pass


def asin(x, name=None):
    """
    [TensorFlow Docs]
    Computes asin of x element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def assert_equal(x, y, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert the condition `x == y` holds element-wise.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_equal(x, y)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_equal(x, y)], x)
    ```

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] == y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    Args:
        x: Numeric `Tensor`.
        y: Numeric `Tensor`, same dtype as and broadcastable to `x`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`, `y`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional). Defaults to "assert_equal".

    Returns:
        Op that raises `InvalidArgumentError` if `x == y` is False.
    """
    pass


def assert_integer(x, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert that `x` is of integer dtype.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_integer(x)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_integer(x)], x)
    ```

    Args:
        x: `Tensor` whose basetype is integer and is not quantized.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional). Defaults to "assert_integer".

    Returns:
        Op that raises `InvalidArgumentError` if `x == y` is False.
    """
    pass


def assert_less(x, y, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert the condition `x < y` holds element-wise.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_less(x, y)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_less(x, y)], x)
    ```

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] < y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    Args:
        x: Numeric `Tensor`.
        y: Numeric `Tensor`, same dtype as and broadcastable to `x`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`, `y`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional). Defaults to "assert_less".

    Returns:
        Op that raises `InvalidArgumentError` if `x < y` is False.
    """
    pass


def assert_less_equal(x, y, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert the condition `x <= y` holds element-wise.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_less_equal(x, y)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_less_equal(x, y)], x)
    ```

    This condition holds if for every pair of (possibly broadcast) elements
    `x[i]`, `y[i]`, we have `x[i] <= y[i]`.
    If both `x` and `y` are empty, this is trivially satisfied.

    Args:
        x: Numeric `Tensor`.
        y: Numeric `Tensor`, same dtype as and broadcastable to `x`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`, `y`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional). Defaults to "assert_less_equal"

    Returns:
        Op that raises `InvalidArgumentError` if `x <= y` is False.
    """
    pass


def assert_negative(x, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert the condition `x < 0` holds element-wise.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_negative(x)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_negative(x)], x)
    ```

    Negative means, for every element `x[i]` of `x`, we have `x[i] < 0`.
    If `x` is empty this is trivially satisfied.

    Args:
        x: Numeric `Tensor`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional). Defaults to "assert_negative".

    Returns:
        Op raising `InvalidArgumentError` unless `x` is all negative.
    """
    pass


def assert_non_negative(x, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert the condition `x >= 0` holds element-wise.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_non_negative(x)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_non_negative(x)], x)
    ```

    Non-negative means, for every element `x[i]` of `x`, we have `x[i] >= 0`.
    If `x` is empty this is trivially satisfied.

    Args:
        x: Numeric `Tensor`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional).
              Defaults to "assert_non_negative".

    Returns:
        Op raising `InvalidArgumentError` unless `x` is all non-negative.
    """
    pass


def assert_non_positive(x, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert the condition `x <= 0` holds element-wise.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_non_positive(x)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_non_positive(x)], x)
    ```

    Non-positive means, for every element `x[i]` of `x`, we have `x[i] <= 0`.
    If `x` is empty this is trivially satisfied.

    Args:
        x: Numeric `Tensor`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional).
              Defaults to "assert_non_positive".

    Returns:
        Op raising `InvalidArgumentError` unless `x` is all non-positive.
    """
    pass


def assert_positive(x, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert the condition `x > 0` holds element-wise.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_positive(x)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_positive(x)], x)
    ```

    Positive means, for every element `x[i]` of `x`, we have `x[i] > 0`.
    If `x` is empty this is trivially satisfied.

    Args:
        x: Numeric `Tensor`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional). Defaults to "assert_positive".

    Returns:
        Op raising `InvalidArgumentError` unless `x` is all positive.
    """
    pass


def assert_proper_iterable(values):
    """
    [TensorFlow Docs]
    Static assert that values is a "proper" iterable.

    `Ops` that expect iterables of `Tensor` can call this to validate input.
    Useful since `Tensor`, `ndarray`, byte/text type are all iterables themselves.

    Args:
        values: Object to be checked.

    Raises:
        TypeError: If `values` is not iterable or is one of
            `Tensor`, `SparseTensor`, `np.array`, `tf.compat.bytes_or_text_types`.
    """
    pass


def assert_rank(x, rank, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert `x` has rank equal to `rank`.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_rank(x, 2)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_rank(x, 2)], x)
    ```

    Args:
        x: Numeric `Tensor`.
        rank: Scalar integer `Tensor`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional). Defaults to "assert_rank".

    Returns:
        Op raising `InvalidArgumentError` unless `x` has specified rank.

    Raises:
        ValueError: If static checks determine `x` has wrong rank.
    """
    pass


def assert_rank_at_least(x, rank, data=None, summarize=None, name=None):
    """
    [TensorFlow Docs]
    Assert `x` has rank equal to `rank` or higher.

    Example of adding a dependency to an operation:

    ```python
    with tf.control_dependencies([tf.assert_rank_at_least(x, 2)]):
        output = tf.reduce_sum(x)
    ```

    Example of adding dependency to the tensor being checked:

    ```python
    x = tf.with_dependencies([tf.assert_rank_at_least(x, 2)], x)
    ```

    Args:
        x: Numeric `Tensor`.
        rank: Scalar `Tensor`.
        data: The tensors to print out if the condition is False. Defaults to
              error message and first few entries of `x`.
        summarize: Print this many entries of each tensor.
        name: A name for this operation (optional).
              Defaults to "assert_rank_at_least".

    Returns:
        Op raising `InvalidArgumentError` unless `x` has specified rank or higher.

    Raises:
        ValueError: If static checks determine `x` has wrong rank.
    """
    pass


def assert_type(tensor, tf_type):
    """
    [TensorFlow Docs]
    Asserts that the given `Tensor` is of the specified type.

    Args:
        tensor: A tensorflow `Tensor`.
        tf_type: A tensorflow type (dtypes.float32, tf.int64, dtypes.bool, etc).

    Raises:
        ValueError: If the tensors data type doesn't match tf_type.
    """
    pass


def assert_variables_initialized(var_list=None):
    """
    [TensorFlow Docs]
    Returns an Op to check if variables are initialized.

    NOTE: This function is obsolete and will be removed in 6 months. Please
    change your implementation to use `report_uninitialized_variables()`.

    When run, the returned Op will raise the exception `FailedPreconditionError`
    if any of the variables has not yet been initialized.

    Note: This function is implemented by trying to fetch the values of the
    variables. If one of the variables is not initialized a message may be
    logged by the C++ runtime. This is expected.

    Args:
        var_list: List of `Variable` objects to check. Defaults to the
                  value of `all_variables().`

    Returns:
        An Op, or None if there are no variables.
    """
    pass


def assign(ref, value, validate_shape=None, use_locking=None, name=None):
    """
    [TensorFlow Docs]
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
    pass


def assign_add(ref, value, use_locking=None, name=None):
    """
    [TensorFlow Docs]
    Update 'ref' by adding 'value' to it.

    This operation outputs "ref" after the update is done.
    This makes it easier to chain operations that need to use the reset value.

    Args:
        ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
             Should be from a `Variable` node.
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
    pass


def assign_sub(ref, value, use_locking=None, name=None):
    """
    [TensorFlow Docs]
    Update 'ref' by subtracting 'value' from it.

    This operation outputs "ref" after the update is done.
    This makes it easier to chain operations that need to use the reset value.

    Args:
        ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
             Should be from a `Variable` node.
        value: A `Tensor`. Must have the same type as `ref`.
               The value to be subtracted to the variable.
        use_locking: An optional `bool`. Defaults to `False`.
                     If True, the subtraction will be protected by a lock;
                     otherwise the behavior is undefined, but may exhibit less contention.
        name: A name for the operation (optional).

    Returns:
        Same as "ref". Returned as a convenience for operations that want
        to use the new value after the variable has been updated.
    """
    pass


def atan(x, name=None):
    """
    [TensorFlow Docs]
    Computes atan of x element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def audio_summary(tag,
                  tensor,
                  sample_rate,
                  max_outputs=3,
                  collections=None,
                  name=None):
    """
    [TensorFlow Docs]
    Outputs a `Summary` protocol buffer with audio.

    The summary has up to `max_outputs` summary values containing audio. The
    audio is built from `tensor` which must be 3-D with shape `[batch_size,
    frames, channels]` or 2-D with shape `[batch_size, frames]`. The values are
    assumed to be in the range of `[-1.0, 1.0]` with a sample rate of
    `sample_rate`.

    The `tag` argument is a scalar `Tensor` of type `string`. It is used to
    build the `tag` of the summary values:

    *  If `max_outputs` is 1, the summary value tag is '*tag*/audio'.
    *  If `max_outputs` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/audio/0', '*tag*/audio/1', etc.

    Args:
        tag: A scalar `Tensor` of type `string`. Used to build the `tag`
             of the summary values.
        tensor: A 3-D `float32` `Tensor` of shape `[batch_size, frames, channels]`
                or a 2-D `float32` `Tensor` of shape `[batch_size, frames]`.
        sample_rate: The sample rate of the signal in hertz.
        max_outputs: Max number of batch elements to generate audio for.
        collections: Optional list of ops.GraphKeys. The collections to add the
                     summary to. Defaults to [ops.GraphKeys.SUMMARIES]
        name: A name for the operation (optional).

    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol
        buffer.
    """
    pass


def batch_cholesky(input, name=None):
    """
    [TensorFlow Docs]
    Calculates the Cholesky decomposition of a batch of square matrices.

    The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices, with the same constraints as the single matrix Cholesky
    decomposition above. The output is a tensor of the same shape as the input
    containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

    Args:
        input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
               Shape is `[..., M, M]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
    """
    pass


def batch_cholesky_solve(chol, rhs, name=None):
    """
    [TensorFlow Docs]
    Solve batches of linear eqns `A X = RHS`, given Cholesky factorizations.

    ```python
    # Solve one linear system (K = 1) for every member of the length 10 batch.
    A = ... # shape 10 x 2 x 2
    RHS = ... # shape 10 x 2 x 1
    chol = tf.batch_cholesky(A)  # shape 10 x 2 x 2
    X = tf.batch_cholesky_solve(chol, RHS)  # shape 10 x 2 x 1
    # tf.matmul(A, X) ~ RHS
    X[3, :, 0]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 0]

    # Solve five linear systems (K = 5) for every member of the length 10 batch.
    A = ... # shape 10 x 2 x 2
    RHS = ... # shape 10 x 2 x 5
    ...
    X[3, :, 2]  # Solution to the linear system A[3, :, :] x = RHS[3, :, 2]
    ```

    Args:
        chol: A `Tensor`. Must be `float32` or `float64`, shape is `[..., M, M]`.
              Cholesky factorization of `A`, e.g. `chol = tf.batch_cholesky(A)`.
              For that reason, only the lower triangular parts (including the diagonal)
              of the last two dimensions of `chol` are used. The strictly upper part is
              assumed to be zero and not accessed.
        rhs: A `Tensor`, same type as `chol`, shape is `[..., M, K]`.
        name: A name to give this `Op`. Defaults to `batch_cholesky_solve`.

    Returns:
        Solution to `A x = rhs`, shape `[..., M, K]`.
    """
    pass


def batch_fft(input, name=None):
    """
    [TensorFlow Docs]
    Compute the 1-dimensional discrete Fourier Transform over the inner-most

    dimension of `input`.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        A complex64 tensor of the same shape as `input`. The inner-most
        dimension of `input` is replaced with its 1D Fourier Transform.
    """
    pass


def batch_fft2d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the 2-dimensional discrete Fourier Transform over the inner-most

    2 dimensions of `input`.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        A complex64 tensor of the same shape as `input`. The inner-most 2
        dimensions of `input` are replaced with their 2D Fourier Transform.
    """
    pass


def batch_fft3d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the 3-dimensional discrete Fourier Transform over the inner-most 3

    dimensions of `input`.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        A complex64 tensor of the same shape as `input`. The inner-most 3
        dimensions of `input` are replaced with their 3D Fourier Transform.
    """
    pass


def batch_ifft(input, name=None):
    """
    [TensorFlow Docs]
    Compute the inverse 1-dimensional discrete Fourier Transform over the inner-most

    dimension of `input`.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        A complex64 tensor of the same shape as `input`. The inner-most
        dimension of `input` is replaced with its inverse 1D Fourier Transform.
    """
    pass


def batch_ifft2d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the inverse 2-dimensional discrete Fourier Transform over the inner-most

    2 dimensions of `input`.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        A complex64 tensor of the same shape as `input`. The inner-most 2
        dimensions of `input` are replaced with their inverse 2D Fourier Transform.
    """
    pass


def batch_ifft3d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the inverse 3-dimensional discrete Fourier Transform over the inner-most

    3 dimensions of `input`.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        A complex64 tensor of the same shape as `input`. The inner-most 3
        dimensions of `input` are replaced with their inverse 3D Fourier Transform.
    """
    pass


def _batch_mat_mul(x, y, adj_x=None, adj_y=None, name=None):
    """
    [TensorFlow Docs]
    Multiplies slices of two tensors in batches.

    Multiplies all slices of `Tensor` `x` and `y` (each slice can be
    viewed as an element of a batch), and arranges the individual results
    in a single output tensor of the same batch size. Each of the
    individual slices can optionally be adjointed (to adjoint a matrix
    means to transpose and conjugate it) before multiplication by setting
    the `adj_x` or `adj_y` flag to `True`, which are by default `False`.

    The input tensors `x` and `y` are 3-D or higher with shape `[..., r_x, c_x]`
    and `[..., r_y, c_y]`.

    The output tensor is 3-D or higher with shape `[..., r_o, c_o]`, where:

            r_o = c_x if adj_x else r_x
            c_o = r_y if adj_y else c_y

    It is computed as:

            output[..., :, :] = matrix(x[..., :, :]) * matrix(y[..., :, :])

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `complex64`, `complex128`.
           3-D or higher with shape `[..., r_x, c_x]`.
        y: A `Tensor`. Must have the same type as `x`.
           3-D or higher with shape `[..., r_y, c_y]`.
        adj_x: An optional `bool`. Defaults to `False`.
               If `True`, adjoint the slices of `x`. Defaults to `False`.
               adj_y: An optional `bool`. Defaults to `False`.
               If `True`, adjoint the slices of `y`. Defaults to `False`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
        3-D or higher with shape `[..., r_o, c_o]`
    """
    pass


def batch_matrix_band_part(input, num_lower, num_upper, name=None):
    """
    [TensorFlow Docs]
    Copy a tensor setting everything outside a central band in each innermost matrix

    to zero.

    The `band` part is computed as follows:
    Assume `input` has `k` dimensions `[I, J, K, ..., M, N]`, then the output is a
    tensor with the same shape where

    `band[i, j, k, ..., m, n] = in_band(m, n) * input[i, j, k, ..., m, n]`.

    The indicator function 'in_band(m, n)` is one if
    `(num_lower < 0 || (m-n) <= num_lower)) &&
    (num_upper < 0 || (n-m) <= num_upper)`, and zero otherwise.

    For example:

    ```prettyprint
    # if 'input' is [[ 0, 1, 2, 3]
                   [-1, 0, 1, 2]
                   [-2, -1, 0, 1]
                   [-3, -2, -1, 0]],

    tf.batch_matrix_band_part(input, 1, -1) ==> [[ 0, 1, 2, 3]
                                               [-1, 0, 1, 2]
                                               [ 0, -1, 0, 1]
                                               [ 0, 0, -1, 0]],

    tf.batch_matrix_band_part(input, 2, 1) ==> [[ 0, 1, 0, 0]
                                              [-1, 0, 1, 0]
                                              [-2, -1, 0, 1]
                                              [ 0, -2, -1, 0]]
    ```

    Useful special cases:

    ```prettyprint
   tf.batch_matrix_band_part(input, 0, -1) ==> Upper triangular part.
   tf.batch_matrix_band_part(input, -1, 0) ==> Lower triangular part.
   tf.batch_matrix_band_part(input, 0, 0) ==> Diagonal.
    ```

    Args:
        input: A `Tensor`. Rank `k` tensor.
        num_lower: A `Tensor` of type `int64`.
                   0-D tensor. Number of subdiagonals to keep. If negative, keep entire
                   lower triangle.
        num_upper: A `Tensor` of type `int64`.
                   0-D tensor. Number of superdiagonals to keep. If negative, keep
                   entire upper triangle.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        Rank `k` tensor of the same shape as input. The extracted banded tensor.
    """
    pass


def batch_matrix_determinant(input, name=None):
    """
    [TensorFlow Docs]
    Calculates the determinants for a batch of square matrices.

    The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices. The output is a 1-D tensor containing the determinants
    for all input submatrices `[..., :, :]`.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
               Shape is `[..., M, M]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. Shape is `[...]`.
    """
    pass


def batch_matrix_diag(diagonal, name=None):
    """
    [TensorFlow Docs]
    Returns a batched diagonal tensor with a given batched diagonal values.

    Given a `diagonal`, this operation returns a tensor with the `diagonal` and
    everything else padded with zeros. The diagonal is computed as follows:

    Assume `diagonal` has `k` dimensions `[I, J, K, ..., N]`, then the output is a
    tensor of rank `k+1` with dimensions [I, J, K, ..., N, N]` where:

    `output[i, j, k, ..., m, n] = 1{m=n} * diagonal[i, j, k, ..., n]`.

    For example:

    ```prettyprint
    # 'diagonal' is [[1, 2, 3, 4], [5, 6, 7, 8]]

    and diagonal.shape = (2, 4)

    tf.batch_matrix_diag(diagonal) ==> [[[1, 0, 0, 0]
                                       [0, 2, 0, 0]
                                       [0, 0, 3, 0]
                                       [0, 0, 0, 4]],
                                      [[5, 0, 0, 0]
                                       [0, 6, 0, 0]
                                       [0, 0, 7, 0]
                                       [0, 0, 0, 8]]]

    which has shape (2, 4, 4)
    ```

    Args:
        diagonal: A `Tensor`. Rank `k`, where `k >= 1`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `diagonal`.
        Rank `k+1`, with `output.shape = diagonal.shape + [diagonal.shape[-1]]`.
    """
    pass


def batch_matrix_diag_part(input, name=None):
    """
    [TensorFlow Docs]
    Returns the batched diagonal part of a batched tensor.

    This operation returns a tensor with the `diagonal` part
    of the batched `input`. The `diagonal` part is computed as follows:

    Assume `input` has `k` dimensions `[I, J, K, ..., N, N]`, then the output is a
    tensor of rank `k - 1` with dimensions `[I, J, K, ..., N]` where:

    `diagonal[i, j, k, ..., n] = input[i, j, k, ..., n, n]`.

    The input must be at least a matrix.

    For example:

    ```prettyprint
    # 'input' is [[[1, 0, 0, 0]
                 [0, 2, 0, 0]
                 [0, 0, 3, 0]
                 [0, 0, 0, 4]],
                [[5, 0, 0, 0]
                 [0, 6, 0, 0]
                 [0, 0, 7, 0]
                 [0, 0, 0, 8]]]

    and input.shape = (2, 4, 4)

    tf.batch_matrix_diag_part(input) ==> [[1, 2, 3, 4], [5, 6, 7, 8]]

    which has shape (2, 4)
    ```

    Args:
        input: A `Tensor`.
               Rank `k` tensor where `k >= 2` and the last two dimensions are equal.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        The extracted diagonal(s) having shape
        `diagonal.shape = input.shape[:-1]`.
    """
    pass


def batch_matrix_inverse(input, adjoint=None, name=None):
    """
    [TensorFlow Docs]
    Calculates the inverse of square invertible matrices or their adjoints

    (conjugate transposes).

    The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices. The output is a tensor of the same shape as the input
    containing the inverse for all input submatrices `[..., :, :]`.

    The op uses LU decomposition with partial pivoting to compute the inverses.

    If a matrix is not invertible there is no guarantee what the op does. It
    may detect the condition and raise an exception or it may simply return a
    garbage result.

    Args:
        input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
               Shape is `[..., M, M]`.
        adjoint: An optional `bool`. Defaults to `False`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
    """
    pass


def batch_matrix_solve(matrix, rhs, adjoint=None, name=None):
    """
    [TensorFlow Docs]
    Solves systems of linear equations. Checks for invertibility.

    Matrix is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices. Rhs is a tensor of shape
    `[..., M, K]`. The output is a tensor shape `[..., M, K]`. If `adjoint` is `False` then each output
    matrix satisfies `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
    If `adjoint` is `True` then each output
    matrix satisfies `adjoint(matrix[..., :, :]) * output[..., :, :] = rhs[..., :, :]`.

    Args:
        matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
                Shape is `[..., M, M]`.
        rhs: A `Tensor`. Must have the same type as `matrix`.
             Shape is `[..., M, K]`.
        adjoint: An optional `bool`. Defaults to `False`.
                 Boolean indicating whether to solve with `matrix` or its (block-wise)
                 adjoint.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.
    """
    pass


def batch_matrix_solve_ls(matrix,
                          rhs,
                          l2_regularizer=0.0,
                          fast=True,
                          name=None):
    """
    [TensorFlow Docs]
    Solves multiple linear least-squares problems.

    `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
    form `M`-by-`N` matrices. Rhs is a tensor of shape `[..., M, K]` whose
    inner-most 2 dimensions form `M`-by-`K` matrices. The computed output is a
    `Tensor` of shape `[..., N, K]` whose inner-most 2 dimensions form `M`-by-`K`
    matrices that solve the equations
    `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]` in the least squares
    sense.

    Below we will use the following notation for each pair of
    matrix and right-hand sides in the batch:

    `matrix`=\\(A \in \Re^{m \times n}\\),
    `rhs`=\\(B  \in \Re^{m \times k}\\),
    `output`=\\(X  \in \Re^{n \times k}\\),
    `l2_regularizer`=\\(\lambda\\).

    If `fast` is `True`, then the solution is computed by solving the normal
    equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
    \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
    problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
    \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
    \\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is
    the minimum-norm solution to the under-determined linear system, i.e.
    \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\), subject to
    \\(A Z = B\\). Notice that the fast path is only numerically stable when
    \\(A\\) is numerically full rank and has a condition number
    \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\) or\\(\lambda\\)
    is sufficiently large.

    If `fast` is `False` an algorithm based on the numerically robust complete
    orthogonal decomposition is used. This computes the minimum-norm
    least-squares solution, even when \\(A\\) is rank deficient. This path is
    typically 6-7 times slower than the fast path. If `fast` is `False` then
    `l2_regularizer` is ignored.

    Args:
        matrix: `Tensor` of shape `[..., M, N]`.
        rhs: `Tensor` of shape `[..., M, K]`.
        l2_regularizer: 0-D `double` `Tensor`. Ignored if `fast=False`.
                        fast: bool. Defaults to `True`.
        name: string, optional name of the operation.

    Returns:
        output: `Tensor` of shape `[..., N, K]` whose inner-most 2 dimensions form
            `M`-by-`K` matrices that solve the equations
            `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]` in the least
            squares sense.
    """
    pass


def batch_matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None,
                                  name=None):
    """
    [TensorFlow Docs]
    Solves systems of linear equations with upper or lower triangular matrices by

    backsubstitution.

    `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
    square matrices. If `lower` is `True` then the strictly upper triangular part
    of each inner-most matrix is assumed to be zero and not accessed.
    If `lower` is False then the strictly lower triangular part of each inner-most
    matrix is assumed to be zero and not accessed.
    `rhs` is a tensor of shape [..., M, K]`.

    The output is a tensor of shape `[..., M, K]`. If `adjoint` is `True` then the
    innermost matrices in output` satisfy matrix equations
    `matrix[..., :, :] * output[..., :, :] = rhs[..., :, :]`.
    If `adjoint` is `False` then the strictly then the  innermost matrices in
    `output` satisfy matrix equations
    `adjoint(matrix[..., i, k]) * output[..., k, j] = rhs[..., i, j]`.

    Args:
        matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
                Shape is `[..., M, M]`.
        rhs: A `Tensor`. Must have the same type as `matrix`.
             Shape is `[..., M, K]`.
        lower: An optional `bool`. Defaults to `True`.
               Boolean indicating whether the innermost matrices in `matrix` are
               lower or upper triangular.
        adjoint: An optional `bool`. Defaults to `False`.
                 Boolean indicating whether to solve with `matrix` or its (block-wise)
                 adjoint.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.
    """
    pass


def batch_self_adjoint_eig(input, name=None):
    """
    [TensorFlow Docs]
    Calculates the Eigen Decomposition of a batch of square self-adjoint matrices.

    The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
    form square matrices, with the same constraints as the single matrix
    SelfAdjointEig.

    The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
    eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

    Args:
        input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
               Shape is `[..., M, M]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. Shape is `[..., M+1, M]`.
    """
    pass


def batch_to_space(input, crops, block_size, name=None):
    """
    [TensorFlow Docs]
    BatchToSpace for 4-D tensors of type T.

    Rearranges (permutes) data from batch into blocks of spatial data, followed by
    cropping. This is the reverse transformation of SpaceToBatch. More specifically,
    this op outputs a copy of the input tensor where values from the `batch`
    dimension are moved in spatial blocks to the `height` and `width` dimensions,
    followed by cropping along the `height` and `width` dimensions.

    Args:
        input: A `Tensor`. 4-D tensor with shape
               `[batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
               depth]`. Note that the batch size of the input tensor must be divisible by
               `block_size * block_size`.
        crops: A `Tensor` of type `int32`.
               2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
               how many elements to crop from the intermediate result across the spatial
               dimensions as follows:

          crops = [[crop_top, crop_bottom], [crop_left, crop_right]]
        block_size: An `int`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        4-D with shape `[batch, height, width, depth]`, where:

          height = height_pad - crop_top - crop_bottom
          width = width_pad - crop_left - crop_right

        The attr `block_size` must be greater than one. It indicates the block size.
    """
    pass


def bitcast(input, type, name=None):
    """
    [TensorFlow Docs]
    Bitcasts a tensor from one type to another without copying data.

    Given a tensor `input`, this operation returns a tensor that has the same buffer
    data as `input` with datatype `type`.

    If the input datatype `T` is larger than the output datatype `type` then the
    shape changes from [...] to [..., sizeof(`T`)/sizeof(`type`)].

    If `T` is smaller than `type`, the operator requires that the rightmost
    dimension be equal to sizeof(`type`)/sizeof(`T`). The shape then goes from
    [..., sizeof(`type`)/sizeof(`T`)] to [...].

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        type: A `tf.DType` from: `tf.float32, tf.float64, tf.int64, tf.int32, tf.uint8, tf.uint16, tf.int16, tf.int8, tf.complex64, tf.complex128, tf.qint8, tf.quint8, tf.qint32, tf.half`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `type`.
    """
    pass


def boolean_mask(tensor, mask, name="boolean_mask"):
    """
    [TensorFlow Docs]
    Apply boolean mask to tensor. Numpy equivalent is `tensor[mask]`.

    ```python
    # 1-D example
    tensor = [0, 1, 2, 3]
    mask = [True, False, True, False]
    boolean_mask(tensor, mask) ==> [0, 2]
    ```

    In general, `0 < dim(mask) = K <= dim(tensor)`, and `mask`'s shape must match
    the first K dimensions of `tensor`'s shape. We then have:
        `boolean_mask(tensor, mask)[i, j1,...,jd] = tensor[i1,...,iK,j1,...,jd]`
    where `(i1,...,iK)` is the ith `True` entry of `mask` (row-major order).

    Args:
        tensor: N-D tensor.
        mask: K-D boolean tensor, K <= N and K must be known statically.
        name: A name for this operation (optional).

    Returns:
        Tensor populated by entries in `tensor` corresponding to `True` values in
            `mask`.

    Raises:
        ValueError: If shapes do not conform.

    Examples:

    ```python
    # 2-D example
    tensor = [[1, 2], [3, 4], [5, 6]]
    mask = [True, False, True]
    boolean_mask(tensor, mask) ==> [[1, 2], [5, 6]]
    ```
    """
    pass


def case(pred_fn_pairs, default, exclusive=False, name="case"):
    """
    [TensorFlow Docs]
    Create a case operation.

    The `pred_fn_pairs` parameter is a dict or list of pairs of size N.
    Each pair contains a boolean scalar tensor and a python callable that
    creates the tensors to be returned if the boolean evaluates to True.
    `default` is a callable generating a list of tensors. All the callables
    in `pred_fn_pairs` as well as `default` should return the same number
    and types of tensors.

    If `exclusive==True`, all predicates are evaluated, and a logging operation
    with an error is returned if more than one of the predicates evaluates to
    True. If `exclusive==False`, execution stops are the first predicate which
    evaluates to True, and the tensors generated by the corresponding function
    are returned immediately. If none of the predicates evaluate to True, this
    operation returns the tensors generated by `default`.

    Example 1:
        Pseudocode:
        ```
            if (x < y) return 17;
            else return 23;
        ```

        Expressions:
        ```
            f1 = lambda: tf.constant(17)
            f2 = lambda: tf.constant(23)
            r = case([(tf.less(x, y), f1)], default=f2)
        ```

    Example 2:
        Pseudocode:
        ```
            if (x < y && x > z) raise OpError("Only one predicate may evaluate true");
            if (x < y) return 17;
            else if (x > z) return 23;
            else return -1;
        ```

        Expressions:
        ```
            x = tf.constant(0)
            y = tf.constant(1)
            z = tf.constant(2)
            def f1(): return tf.constant(17)
            def f2(): return tf.constant(23)
            def f3(): return tf.constant(-1)
            r = case({tf.less(x, y): f1, tf.greater(x, z): f2},
               default=f3, exclusive=True)
        ```

    Args:
        pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                       callable which returns a list of tensors.
        default: A callable that returns a list of tensors.
        exclusive: True iff more than one predicate is allowed to evaluate to True.
        name: A name for this operation (optional).

    Returns:
        The tensors returned by the first pair whose predicate evaluated to True, or
        those returned by `default` if none does.

    Raises:
        TypeError: If `pred_fn_pairs` is not a list/dictionary.
        TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
        TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
    """
    pass


def cast(x, dtype, name=None):
    """
    [TensorFlow Docs]
    Casts a tensor to a new type.

    The operation casts `x` (in case of `Tensor`) or `x.values`
    (in case of `SparseTensor`) to `dtype`.

    For example:

    ```python
    # tensor `a` is [1.8, 2.2], dtype=tf.float
    tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
    ```

    Args:
        x: A `Tensor` or `SparseTensor`.
        dtype: The destination type.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` or `SparseTensor` with same shape as `x`.

    Raises:
        TypeError: If `x` cannot be cast to the `dtype`.
    """
    pass


def ceil(x, name=None):
    """
    [TensorFlow Docs]
    Returns element-wise smallest integer in not less than x.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def check_numerics(tensor, message, name=None):
    """
    [TensorFlow Docs]
    Checks a tensor for NaN and Inf values.

    When run, reports an `InvalidArgument` error if `tensor` has any values
    that are not a number (NaN) or infinity (Inf). Otherwise, passes `tensor` as-is.

    Args:
        tensor: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        message: A `string`. Prefix of the error message.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `tensor`.
    """
    pass


def cholesky(input, name=None):
    """
    [TensorFlow Docs]
    Calculates the Cholesky decomposition of a square matrix.

    The input has to be symmetric and positive definite. Only the lower-triangular
    part of the input will be used for this operation. The upper-triangular part
    will not be read.

    The result is the lower-triangular matrix of the Cholesky decomposition of the
    input, `L`, so that `input = L L^*`.

    Args:
        input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
               Shape is `[M, M]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. Shape is `[M, M]`.
    """
    pass


def cholesky_solve(chol, rhs, name=None):
    """
    [TensorFlow Docs]
    Solve linear equations `A X = RHS`, given Cholesky factorization of `A`.

    ```python
    # Solve one system of linear equations (K = 1).
    A = [[3, 1], [1, 3]]
    RHS = [[2], [22]]  # shape 2 x 1
    chol = tf.cholesky(A)
    X = tf.cholesky_solve(chol, RHS)
    # tf.matmul(A, X) ~ RHS
    X[:, 0]  # Solution to the linear system A x = RHS[:, 0]

    # Solve five systems of linear equations (K = 5).
    A = [[3, 1], [1, 3]]
    RHS = [[1, 2, 3, 4, 5], [11, 22, 33, 44, 55]]  # shape 2 x 5
    ...
    X[:, 2]  # Solution to the linear system A x = RHS[:, 2]
    ```

    Args:
        chol: A `Tensor`. Must be `float32` or `float64`, shape is `[M, M]`.
              Cholesky factorization of `A`, e.g. `chol = tf.cholesky(A)`. For that
              reason, only the lower triangular part (including the diagonal) of `chol`
              is used. The strictly upper part is assumed to be zero and not accessed.
        rhs: A `Tensor`, same type as `chol`, shape is `[M, K]`, designating `K`
             systems of linear equations.
        name: A name to give this `Op`. Defaults to `cholesky_solve`.

    Returns:
        Solution to `A X = RHS`, shape `[M, K]`. The solutions to the `K` systems.
    """
    pass


def clip_by_average_norm(t, clip_norm, name=None):
    """
    [TensorFlow Docs]
    Clips tensor values to a maximum average L2-norm.

    Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
    normalizes `t` so that its average L2-norm is less than or equal to
    `clip_norm`. Specifically, if the average L2-norm is already less than or
    equal to `clip_norm`, then `t` is not modified. If the average L2-norm is
    greater than `clip_norm`, then this operation returns a tensor of the same
    type and shape as `t` with its values set to:

    `t * clip_norm / l2norm_avg(t)`

    In this case, the average L2-norm of the output tensor is `clip_norm`.

    This operation is typically used to clip gradients before applying them with
    an optimizer.

    Args:
        t: A `Tensor`.
        clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
        name: A name for the operation (optional).

    Returns:
        A clipped `Tensor`.
    """
    pass


def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):
    """
    [TensorFlow Docs]
    Clips values of multiple tensors by the ratio of the sum of their norms.

    Given a tuple or list of tensors `t_list`, and a clipping ratio `clip_norm`,
    this operation returns a list of clipped tensors `list_clipped`
    and the global norm (`global_norm`) of all tensors in `t_list`. Optionally,
    if you've already computed the global norm for `t_list`, you can specify
    the global norm with `use_norm`.

    To perform the clipping, the values `t_list[i]` are set to:

            t_list[i] * clip_norm / max(global_norm, clip_norm)

    where:

            global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))

    If `clip_norm > global_norm` then the entries in `t_list` remain as they are,
    otherwise they're all shrunk by the global ratio.

    Any of the entries of `t_list` that are of type `None` are ignored.

    This is the correct way to perform gradient clipping (for example, see
    [Pascanu et al., 2012](http://arxiv.org/abs/1211.5063)
    ([pdf](http://arxiv.org/pdf/1211.5063.pdf))).

    However, it is slower than `clip_by_norm()` because all the parameters must be
    ready before the clipping operation can be performed.

    Args:
        t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
        clip_norm: A 0-D (scalar) `Tensor` > 0. The clipping ratio.
        use_norm: A 0-D (scalar) `Tensor` of type `float` (optional). The global
                  norm to use. If not provided, `global_norm()` is used to compute the norm.
        name: A name for the operation (optional).

    Returns:
        list_clipped: A list of `Tensors` of the same type as `list_t`.
        global_norm: A 0-D (scalar) `Tensor` representing the global norm.

    Raises:
        TypeError: If `t_list` is not a sequence.
    """
    pass


def clip_by_norm(t, clip_norm, name=None):
    """
    [TensorFlow Docs]
    Clips tensor values to a maximum L2-norm.

    Given a tensor `t`, and a maximum clip value `clip_norm`, this operation
    normalizes `t` so that its L2-norm is less than or equal to `clip_norm`.
    Specifically, if the L2-norm is already less than or equal to `clip_norm`,
    then `t` is not modified. If the L2-norm is greater than `clip_norm`, then
    this operation returns a tensor of the same type and shape as `t` with its
    values set to:

    `t * clip_norm / l2norm(t)`

    In this case, the L2-norm of the output tensor is `clip_norm`.

    This operation is typically used to clip gradients before applying them with
    an optimizer.

    Args:
        t: A `Tensor`.
        clip_norm: A 0-D (scalar) `Tensor` > 0. A maximum clipping value.
        name: A name for the operation (optional).

    Returns:
        A clipped `Tensor`.
    """
    pass


def clip_by_value(t, clip_value_min, clip_value_max,
                  name=None):
    """
    [TensorFlow Docs]
    Clips tensor values to a specified min and max.

    Given a tensor `t`, this operation returns a tensor of the same type and
    shape as `t` with its values clipped to `clip_value_min` and `clip_value_max`.
    Any values less than `clip_value_min` are set to `clip_value_min`. Any values
    greater than `clip_value_max` are set to `clip_value_max`.

    Args:
        t: A `Tensor`.
        clip_value_min: A 0-D (scalar) `Tensor`. The minimum value to clip by.
        clip_value_max: A 0-D (scalar) `Tensor`. The maximum value to clip by.
        name: A name for the operation (optional).

    Returns:
        A clipped `Tensor`.
    """
    pass


def complex(real, imag, name=None):
    """
    [TensorFlow Docs]
    Converts two real numbers to a complex number.

    Given a tensor `real` representing the real part of a complex number, and a
    tensor `imag` representing the imaginary part of a complex number, this
    operation returns complex numbers elementwise of the form \\(a + bj\\), where
    *a* represents the `real` part and *b* represents the `imag` part.

    The input tensors `real` and `imag` must have the same shape.

    For example:

    ```
    # tensor 'real' is [2.25, 3.25]
    # tensor `imag` is [4.75, 5.75]
    tf.complex(real, imag) ==> [[2.25 + 4.75j], [3.25 + 5.75j]]
    ```

    Args:
        real: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        imag: A `Tensor`. Must have the same type as `real`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64` or `complex128`.
    """
    pass


def complex_abs(x, name=None):
    """
    [TensorFlow Docs]
    Computes the complex absolute value of a tensor.

    Given a tensor `x` of complex numbers, this operation returns a tensor of type
    `float` or `double` that is the absolute value of each element in `x`. All
    elements in `x` must be complex numbers of the form \\(a + bj\\). The
    absolute value is computed as \\( \sqrt{a^2 + b^2}\\).

    For example:

    ```
    # tensor 'x' is [[-2.25 + 4.75j], [-3.25 + 5.75j]]
    tf.complex_abs(x) ==> [5.25594902, 6.60492229]
    ```

    Args:
        x: A `Tensor` of type `complex64` or `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `float32` or `float64`.
    """
    pass


def concat(concat_dim, values, name="concat"):
    """
    [TensorFlow Docs]
    Concatenates tensors along one dimension.

    Concatenates the list of tensors `values` along dimension `concat_dim`. If
    `values[i].shape = [D0, D1, ... Dconcat_dim(i), ...Dn]`, the concatenated
    result has shape

            [D0, D1, ... Rconcat_dim, ...Dn]

    where

            Rconcat_dim = sum(Dconcat_dim(i))

    That is, the data from the input tensors is joined along the `concat_dim`
    dimension.

    The number of dimensions of the input tensors must match, and all dimensions
    except `concat_dim` must be equal.

    For example:

    ```python
    t1 = [[1, 2, 3], [4, 5, 6]]
    t2 = [[7, 8, 9], [10, 11, 12]]
    tf.concat(0, [t1, t2]) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

    # tensor t3 with shape [2, 3]
    # tensor t4 with shape [2, 3]
    tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]
    tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]
    ```

    Args:
        concat_dim: 0-D `int32` `Tensor`. Dimension along which to concatenate.
        values: A list of `Tensor` objects or a single `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` resulting from concatenation of the input tensors.
    """
    pass


def cond(pred, fn1, fn2, name=None):
    """
    [TensorFlow Docs]
    Return either fn1() or fn2() based on the boolean predicate `pred`.

    `fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have
    the same non-zero number and type of outputs.

    Note that the conditional execution applies only to the operations defined in
    fn1 and fn2. Consider the following simple program:

    ```python
    z = tf.mul(a, b)
    result = tf.cond(x < y, lambda: tf.add(x, z), lambda: tf.square(y))
    ```

    If x < y, the tf.add operation will be executed and tf.square
    operation will not be executed. Since z is needed for at least one
    branch of the cond, the tf.mul operation is always executed, unconditionally.
    Although this behavior is consistent with the dataflow model of TensorFlow,
    it has occasionally surprised some users who expected a lazier semantics.

    Args:
        pred: A scalar determining whether to return the result of `fn1` or `fn2`.
        fn1: The callable to be performed if pred is true.
        fn2: The callable to be performed if pref is false.
        name: Optional name prefix for the returned tensors.

    Returns:
        Tensors returned by the call to either `fn1` or `fn2`. If the callables
        return a singleton list, the element is extracted from the list.

    Raises:
        TypeError: if `fn1` or `fn2` is not callable.
        ValueError: if `fn1` and `fn2` do not return the same number of tensors, or
                return tensors of different types.

    Example:

    ```python
        x = tf.constant(2)
        y = tf.constant(5)
        def f1(): return tf.mul(x, 17)
        def f2(): return tf.add(y, 23)
        r = cond(tf.less(x, y), f1, f2)
        # r is set to f1().
        # Operations in f2 (e.g., tf.add) are not executed.
    ```

    """
    pass


def conj(input, name=None):
    """
    [TensorFlow Docs]
    Returns the complex conjugate of a complex number.

    Given a tensor `input` of complex numbers, this operation returns a tensor of
    complex numbers that are the complex conjugate of each element in `input`. The
    complex numbers in `input` must be of the form \\(a + bj\\), where *a* is the
    real part and *b* is the imaginary part.

    The complex conjugate returned by this operation is of the form \\(a - bj\\).

    For example:

    ```
    # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    tf.conj(input) ==> [-2.25 - 4.75j, 3.25 - 5.75j]
    ```

    Args:
        input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def constant(value, dtype=None, shape=None, name="Const"):
    """
    [TensorFlow Docs]
    Creates a constant tensor.

   The resulting tensor is populated with values of type `dtype`, as
   specified by arguments `value` and (optionally) `shape` (see examples
   below).

   The argument `value` can be a constant value, or a list of values of type
   `dtype`. If `value` is a list, then the length of the list must be less
   than or equal to the number of elements implied by the `shape` argument (if
   specified). In the case where the list length is less than the number of
   elements specified by `shape`, the last element in the list will be used
   to fill the remaining entries.

   The argument `shape` is optional. If present, it specifies the dimensions of
   the resulting tensor. If not present, the shape of `value` is used.

   If the argument `dtype` is not specified, then the type is inferred from
   the type of `value`.

   For example:

   ```python
   # Constant 1-D Tensor populated with value list.
   tensor = tf.constant([1, 2, 3, 4, 5, 6, 7]) => [1 2 3 4 5 6 7]

   # Constant 2-D tensor populated with scalar value -1.
   tensor = tf.constant(-1.0, shape=[2, 3]) => [[-1. -1. -1.]
                                                [-1. -1. -1.]]
   ```

    Args:
        value:  A constant value (or list) of output type `dtype`.

        dtype:  The type of the elements of the resulting tensor.

        shape:  Optional dimensions of resulting tensor.

        name:   Optional name for the tensor.

    Returns:
        A Constant Tensor.
    """
    pass


def constant_initializer(value=0.0, dtype=dtypes.float32):
    """
    [TensorFlow Docs]
    Returns an initializer that generates tensors with a single value.

    Args:
        value: A Python scalar. All elements of the initialized variable
               will be set to this value.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer that generates tensors with a single value.

    Raises:
        ValueError: if `dtype` is not a floating point type.
    """
    pass


def control_dependencies(control_inputs):
    """
    [TensorFlow Docs]
    Wrapper for `Graph.control_dependencies()` using the default graph.

    See [`Graph.control_dependencies()`](../../api_docs/python/framework.md#Graph.control_dependencies)
    for more details.

    Args:
        control_inputs: A list of `Operation` or `Tensor` objects which
                        must be executed or computed before running the operations
                        defined in the context. Can also be `None` to clear the control
                        dependencies.

    Returns:
   A context manager that specifies control dependencies for all
   operations constructed within the context.
    """
    pass


def convert_to_tensor(value, dtype=None, name=None, as_ref=False):
    """
    [TensorFlow Docs]
    Converts the given `value` to a `Tensor`.

    This function converts Python objects of various types to `Tensor`
    objects. It accepts `Tensor` objects, numpy arrays, Python lists,
    and Python scalars. For example:

    ```python
    import numpy as np

    def my_func(arg):
        arg = tf.convert_to_tensor(arg, dtype=tf.float32)
        return tf.matmul(arg, arg) + arg

    # The following calls are equivalent.
    value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
    value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
    value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    ```

    This function can be useful when composing a new operation in Python
    (such as `my_func` in the example above). All standard Python op
    constructors apply this function to each of their Tensor-valued
    inputs, which allows those ops to accept numpy arrays, Python lists,
    and scalars in addition to `Tensor` objects.

    Args:
        value: An object whose type has a registered `Tensor` conversion function.
        dtype: Optional element type for the returned tensor. If missing, the
               type is inferred from the type of `value`.
        name: Optional name to use if a new `Tensor` is created.
              as_ref: True if we want the result as a ref tensor. Only used if a new
              `Tensor` is created.

    Returns:
        A `Tensor` based on `value`.

    Raises:
        TypeError: If no conversion function is registered for `value`.
        RuntimeError: If a registered conversion function returns an invalid value.

    """
    pass


def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None,
                                        as_ref=False):
    """
    [TensorFlow Docs]
    Converts the given object to a `Tensor` or an `IndexedSlices`.

    If `value` is an `IndexedSlices` or `SparseTensor` it is returned
    unmodified. Otherwise, it is converted to a `Tensor` using
    `convert_to_tensor()`.

    Args:
        value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
               by `convert_to_tensor()`.
        dtype: (Optional.) The required `DType` of the returned `Tensor` or
               `IndexedSlices`.
        name: (Optional.) A name to use if a new `Tensor` is created.
              as_ref: True if the caller wants the results as ref tensors.

    Returns:
        An `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

    Raises:
        ValueError: If `dtype` does not match the element type of `value`.
    """
    pass


def cos(x, name=None):
    """
    [TensorFlow Docs]
    Computes cos of x element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def count_up_to(ref, limit, name=None):
    """
    [TensorFlow Docs]
    Increments 'ref' until it reaches 'limit'.

    This operation outputs "ref" after the update is done. This makes it
    easier to chain operations that need to use the updated value.

    Args:
        ref: A mutable `Tensor`. Must be one of the following types: `int32`, `int64`.
             Should be from a scalar `Variable` node.
        limit: An `int`.
               If incrementing ref would bring it above limit, instead generates an
               'OutOfRange' error.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `ref`.
        A copy of the input before increment. If nothing else modifies the
        input, the values produced will all be distinct.
    """
    pass


def create_partitioned_variables(
        shape, slicing, initializer, dtype=dtypes.float32,
        trainable=True, collections=None, name=None, reuse=None):
    """
    [TensorFlow Docs]
    Create a list of partitioned variables according to the given `slicing`.

    Currently only one dimension of the full variable can be sliced, and the
    full variable can be reconstructed by the concatenation of the returned
    list along that dimension.

    Args:
        shape: List of integers. The shape of the full variable.
        slicing: List of integers. How to partition the variable.
                 Must be of the same length as `shape`. Each value
                 indicate how many slices to create in the corresponding
                 dimension. Presently only one of the values can be more than 1;
                 that is, the variable can only be sliced along one dimension.

            For convenience, The requested number of partitions does not have to
            divide the corresponding dimension evenly. If it does not, the
            shapes of the partitions are incremented by 1 starting from partition
            0 until all slack is absorbed. The adjustment rules may change in the
            future, but as you can save/restore these variables with different
            slicing specifications this should not be a problem.
        initializer: A `Tensor` of shape `shape` or a variable initializer
                     function. If a function, it will be called once for each slice,
                     passing the shape and data type of the slice as parameters. The
                     function must return a tensor with the same shape as the slice.
        dtype: Type of the variables. Ignored if `initializer` is a `Tensor`.
               trainable: If True also add all the variables to the graph collection
               `GraphKeys.TRAINABLE_VARIABLES`.
        collections: List of graph collections keys to add the variables to.
                     Defaults to `[GraphKeys.VARIABLES]`.
        name: Optional name for the full variable. Defaults to
              `"PartitionedVariable"` and gets uniquified automatically.
              reuse: Boolean or `None`; if `True` and name is set, it would reuse
              previously created variables. if `False` it will create new variables.
              if `None`, it would inherit the parent scope reuse.

    Returns:
        A list of Variables corresponding to the slicing.

    Raises:
        ValueError: If any of the arguments is malformed.
    """
    pass


def cross(a, b, name=None):
    """
    [TensorFlow Docs]
    Compute the pairwise cross product.

    `a` and `b` must be the same shape; they can either be simple 3-element vectors,
    or any shape where the innermost dimension is 3. In the latter case, each pair
    of corresponding 3-element vectors is cross-multiplied independently.

    Args:
        a: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
           A tensor containing 3-element vectors.
        b: A `Tensor`. Must have the same type as `a`.
           Another tensor, of same type and shape as `a`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `a`.
        Pairwise cross product of the vectors in `a` and `b`.
    """
    pass


def decode_csv(records, record_defaults, field_delim=None, name=None):
    """
    [TensorFlow Docs]
    Convert CSV records to tensors. Each column maps to one tensor.

    RFC 4180 format is expected for the CSV records.
    (https://tools.ietf.org/html/rfc4180)
    Note that we allow leading and trailing spaces with int or float field.

    Args:
        records: A `Tensor` of type `string`.
                 Each string is a record/row in the csv and all records should have
                 the same format.
        record_defaults: A list of `Tensor` objects with types from: `float32`, `int32`, `int64`, `string`.
                         One tensor per column of the input record, with either a
                         scalar default value for that column or empty if the column is required.
        field_delim: An optional `string`. Defaults to `","`.
                     delimiter to separate fields in a record.
        name: A name for the operation (optional).

    Returns:
        A list of `Tensor` objects. Has the same type as `record_defaults`.
        Each tensor will have the same shape as records.
    """
    pass


def decode_json_example(json_examples, name=None):
    """
    [TensorFlow Docs]
    Convert JSON-encoded Example records to binary protocol buffer strings.

    This op translates a tensor containing Example records, encoded using
    the [standard JSON
    mapping](https://developers.google.com/protocol-buffers/docs/proto3#json),
    into a tensor containing the same records encoded as binary protocol
    buffers. The resulting tensor can then be fed to any of the other
    Example-parsing ops.

    Args:
        json_examples: A `Tensor` of type `string`.
                       Each string is a JSON object serialized according to the JSON
                       mapping of the Example proto.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `string`.
        Each string is a binary Example protocol buffer corresponding
        to the respective element of `json_examples`.
    """
    pass


def decode_raw(bytes, out_type, little_endian=None, name=None):
    """
    [TensorFlow Docs]
    Reinterpret the bytes of a string as a vector of numbers.

    Args:
        bytes: A `Tensor` of type `string`.
               All the elements must have the same length.
        out_type: A `tf.DType` from: `tf.float32, tf.float64, tf.int32, tf.uint8, tf.int16, tf.int8, tf.int64`.
        little_endian: An optional `bool`. Defaults to `True`.
                       Whether the input `bytes` are in little-endian order.
                       Ignored for `out_type` values that are stored in a single byte like
                       `uint8`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `out_type`.
        A Tensor with one more dimension than the input `bytes`. The
        added dimension will have size equal to the length of the elements
        of `bytes` divided by the number of bytes to represent `out_type`.
    """
    pass


def delete_session_tensor(name=None):
    """
    [TensorFlow Docs]
    Delete the tensor by feeding a tensor handle.

    This is EXPERIMENTAL and subject to change.

    Delete the tensor of a given tensor handle. The tensor is produced
    in a previous run() and stored in the state of the session.

    Args:
        name: Optional name prefix for the return tensor.

    Returns:
        A pair of graph elements. The first is a placeholder for feeding a
        tensor handle and the second is a deletion operation.
    """
    pass


def depth_to_space(input, block_size, name=None):
    """
    [TensorFlow Docs]
    DepthToSpace for tensors of type T.

    Rearranges data from depth into blocks of spatial data.
    This is the reverse transformation of SpaceToDepth. More specifically,
    this op outputs a copy of the input tensor where values from the `depth`
    dimension are moved in spatial blocks to the `height` and `width` dimensions.
    The attr `block_size` indicates the input block size and how the data is moved.

        * Chunks of data of size `block_size * block_size` from depth are rearranged
            into non-overlapping blocks of size `block_size x block_size`
        * The width the output tensor is `input_depth * block_size`, whereas the
            height is `input_height * block_size`.
        * The depth of the input tensor must be divisible by
            `block_size * block_size`.

    That is, assuming the input is in the shape:
    `[batch, height, width, depth]`,
    the shape of the output will be:
    `[batch, height*block_size, width*block_size, depth/(block_size*block_size)]`

    This operation requires that the input tensor be of rank 4, and that
    `block_size` be >=1 and that `block_size * block_size` be a divisor of the
    input depth.

    This operation is useful for resizing the activations between convolutions
    (but keeping all data), e.g. instead of pooling. It is also useful for training
    purely convolutional models.

    For example, given this input of shape `[1, 1, 1, 4]`, and a block size of 2:

    ```prettyprint
    x = [[[[1, 2, 3, 4]]]]

    ```

    This operation will output a tensor of shape `[1, 2, 2, 1]`:

    ```prettyprint
     [[[[1], [2]],
       [[3], [4]]]]
    ```

    Here, the input has a batch of 1 and each batch element has shape `[1, 1, 4]`,
    the corresponding output will have 2x2 elements and will have a depth of
    1 channel (1 = `4 / (block_size * block_size)`).
    The output element shape is `[2, 2, 1]`.

    For an input tensor with larger depth, here of shape `[1, 1, 1, 12]`, e.g.

    ```prettyprint
    x = [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    ```

    This operation, for block size of 2, will return the following tensor of shape
    `[1, 2, 2, 3]`

    ```prettyprint
     [[[[1, 2, 3], [4, 5, 6]],
       [[7, 8, 9], [10, 11, 12]]]]

    ```

    Similarly, for the following input of shape `[1 2 2 4]`, and a block size of 2:

    ```prettyprint
    x =  [[[[1, 2, 3, 4],
         [5, 6, 7, 8]],
                [[9, 10, 11, 12],
         [13, 14, 15, 16]]]]
    ```

    the operator will return the following tensor of shape `[1 4 4 1]`:

    ```prettyprint
    x = [[ [1], [2], [5], [6]],
       [ [3], [4], [7], [8]],
       [ [9], [10], [13], [14]],
       [ [11], [12], [15], [16]]]

    ```

    Args:
        input: A `Tensor`.
        block_size: An `int`.
                    The size of the spatial block, same as in Space2Depth.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def deserialize_many_sparse(serialized_sparse, dtype, rank=None, name=None):
    """
    [TensorFlow Docs]
    Deserialize and concatenate `SparseTensors` from a serialized minibatch.

    The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
    `N` is the minibatch size and the rows correspond to packed outputs of
    `serialize_sparse`. The ranks of the original `SparseTensor` objects
    must all match. When the final `SparseTensor` is created, it has rank one
    higher than the ranks of the incoming `SparseTensor` objects (they have been
    concatenated along a new row dimension).

    The output `SparseTensor` object's shape values for all dimensions but the
    first are the max across the input `SparseTensor` objects' shape values
    for the corresponding dimensions. Its first shape value is `N`, the minibatch
    size.

    The input `SparseTensor` objects' indices are assumed ordered in
    standard lexicographic order. If this is not the case, after this
    step run `sparse_reorder` to restore index ordering.

    For example, if the serialized input is a `[2, 3]` matrix representing two
    original `SparseTensor` objects:

            index = [ 0]
              [10]
              [20]
            values = [1, 2, 3]
            shape = [50]

    and

            index = [ 2]
              [10]
            values = [4, 5]
            shape = [30]

    then the final deserialized `SparseTensor` will be:

            index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
            values = [1, 2, 3, 4, 5]
            shape = [2 50]

    Args:
        serialized_sparse: 2-D `Tensor` of type `string` of shape `[N, 3]`.
                           The serialized and packed `SparseTensor` objects.
        dtype: The `dtype` of the serialized `SparseTensor` objects.
        rank: (optional) Python int, the rank of the `SparseTensor` objects.
        name: A name prefix for the returned tensors (optional)

    Returns:
        A `SparseTensor` representing the deserialized `SparseTensor`s,
        concatenated along the `SparseTensor`s' first dimension.

        All of the serialized `SparseTensor`s must have had the same rank and type.
    """
    pass


def device(device_name_or_function):
    """
    [TensorFlow Docs]
    Wrapper for `Graph.device()` using the default graph.

    See
    [`Graph.device()`](../../api_docs/python/framework.md#Graph.device)
    for more details.

    Args:
        device_name_or_function: The device name or function to use in
                                 the context.

    Returns:
        A context manager that specifies the default device to use for newly
        created ops.
    """
    pass


def diag(diagonal, name=None):
    """
    [TensorFlow Docs]
    Returns a diagonal tensor with a given diagonal values.

    Given a `diagonal`, this operation returns a tensor with the `diagonal` and
    everything else padded with zeros. The diagonal is computed as follows:

    Assume `diagonal` has dimensions [D1,..., Dk], then the output is a tensor of
    rank 2k with dimensions [D1,..., Dk, D1,..., Dk] where:

    `output[i1,..., ik, i1,..., ik] = diagonal[i1, ..., ik]` and 0 everywhere else.

    For example:

    ```prettyprint
    # 'diagonal' is [1, 2, 3, 4]
    tf.diag(diagonal) ==> [[1, 0, 0, 0]
                         [0, 2, 0, 0]
                         [0, 0, 3, 0]
                         [0, 0, 0, 4]]
    ```

    Args:
        diagonal: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`.
                  Rank k tensor where k is at most 3.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `diagonal`.
    """
    pass


def diag_part(input, name=None):
    """
    [TensorFlow Docs]
    Returns the diagonal part of the tensor.

    This operation returns a tensor with the `diagonal` part
    of the `input`. The `diagonal` part is computed as follows:

    Assume `input` has dimensions `[D1,..., Dk, D1,..., Dk]`, then the output is a
    tensor of rank `k` with dimensions `[D1,..., Dk]` where:

    `diagonal[i1,..., ik] = input[i1, ..., ik, i1,..., ik]`.

    For example:

    ```prettyprint
    # 'input' is [[1, 0, 0, 0]
                [0, 2, 0, 0]
                [0, 0, 3, 0]
                [0, 0, 0, 4]]

    tf.diag_part(input) ==> [1, 2, 3, 4]
    ```

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `complex64`.
               Rank k tensor where k is 2, 4, or 6.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. The extracted diagonal.
    """
    pass


def digamma(x, name=None):
    """
    [TensorFlow Docs]
    Computes Psi, the derivative of Lgamma (the log of the absolute value of

    `Gamma(x)`), element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def div(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns x / y element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def dynamic_partition(data, partitions, num_partitions, name=None):
    """
    [TensorFlow Docs]
    Partitions `data` into `num_partitions` tensors using indices from `partitions`.

    For each index tuple `js` of size `partitions.ndim`, the slice `data[js, ...]`
    becomes part of `outputs[partitions[js]]`. The slices with `partitions[js] = i`
    are placed in `outputs[i]` in lexicographic order of `js`, and the first
    dimension of `outputs[i]` is the number of entries in `partitions` equal to `i`.
    In detail,

            outputs[i].shape = [sum(partitions == i)] + data.shape[partitions.ndim:]

            outputs[i] = pack([data[js, ...] for js if partitions[js] == i])

    `data.shape` must start with `partitions.shape`.

    For example:

            # Scalar partitions
            partitions = 1
            num_partitions = 2
            data = [10, 20]
            outputs[0] = []  # Empty with shape [0, 2]
            outputs[1] = [[10, 20]]

            # Vector partitions
            partitions = [0, 0, 1, 1, 0]
            num_partitions = 2
            data = [10, 20, 30, 40, 50]
            outputs[0] = [10, 20, 50]
            outputs[1] = [30, 40]

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/DynamicPartition.png" alt>
    </div>

    Args:
        data: A `Tensor`.
        partitions: A `Tensor` of type `int32`.
                    Any shape. Indices in the range `[0, num_partitions)`.
        num_partitions: An `int` that is `>= 1`.
                        The number of partitions to output.
        name: A name for the operation (optional).

    Returns:
        A list of `num_partitions` `Tensor` objects of the same type as data.
    """
    pass


def dynamic_stitch(indices, data, name=None):
    """
    [TensorFlow Docs]
    Interleave the values from the `data` tensors into a single tensor.

    Builds a merged tensor such that

            merged[indices[m][i, ..., j], ...] = data[m][i, ..., j, ...]

    For example, if each `indices[m]` is scalar or vector, we have

            # Scalar indices
            merged[indices[m], ...] = data[m][...]

            # Vector indices
            merged[indices[m][i], ...] = data[m][i, ...]

    Each `data[i].shape` must start with the corresponding `indices[i].shape`,
    and the rest of `data[i].shape` must be constant w.r.t. `i`. That is, we
    must have `data[i].shape = indices[i].shape + constant`. In terms of this
    `constant`, the output shape is

            merged.shape = [max(indices)] + constant

    Values are merged in order, so if an index appears in both `indices[m][i]` and
    `indices[n][j]` for `(m,i) < (n,j)` the slice `data[n][j]` will appear in the
    merged result.

    For example:

            indices[0] = 6
            indices[1] = [4, 1]
            indices[2] = [[5, 2], [0, 3]]
            data[0] = [61, 62]
            data[1] = [[41, 42], [11, 12]]
            data[2] = [[[51, 52], [21, 22]], [[1, 2], [31, 32]]]
            merged = [[1, 2], [11, 12], [21, 22], [31, 32], [41, 42],
                [51, 52], [61, 62]]

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/DynamicStitch.png" alt>
    </div>

    Args:
        indices: A list of at least 2 `Tensor` objects of type `int32`.
        data: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
    """
    pass


def edit_distance(hypothesis, truth, normalize=True, name="edit_distance"):
    """
    [TensorFlow Docs]
    Computes the Levenshtein distance between sequences.

    This operation takes variable-length sequences (`hypothesis` and `truth`),
    each provided as a `SparseTensor`, and computes the Levenshtein distance.
    You can normalize the edit distance by length of `truth` by setting
    `normalize` to true.

    For example, given the following input:

    ```python
    # 'hypothesis' is a tensor of shape `[2, 1]` with variable-length values:
    #   (0,0) = ["a"]
    #   (1,0) = ["b"]
    hypothesis = tf.SparseTensor(
            [[0, 0, 0],
       [1, 0, 0]],
            ["a", "b"]
            (2, 1, 1))

    # 'truth' is a tensor of shape `[2, 2]` with variable-length values:
    #   (0,0) = []
    #   (0,1) = ["a"]
    #   (1,0) = ["b", "c"]
    #   (1,1) = ["a"]
    truth = tf.SparseTensor(
            [[0, 1, 0],
       [1, 0, 0],
       [1, 0, 1],
       [1, 1, 0]]
            ["a", "b", "c", "a"],
            (2, 2, 2))

    normalize = True
    ```

    This operation would return the following:

    ```python
    # 'output' is a tensor of shape `[2, 2]` with edit distances normalized
    # by 'truth' lengths.
    output ==> [[inf, 1.0], # (0,0): no truth, (0,1): no hypothesis
             [0.5, 1.0]]  # (1,0): addition, (1,1): no hypothesis
    ```

    Args:
        hypothesis: A `SparseTensor` containing hypothesis sequences.
        truth: A `SparseTensor` containing truth sequences.
        normalize: A `bool`. If `True`, normalizes the Levenshtein distance by
                   length of `truth.`
        name: A name for the operation (optional).

    Returns:
        A dense `Tensor` with rank `R - 1`, where R is the rank of the
        `SparseTensor` inputs `hypothesis` and `truth`.

    Raises:
        TypeError: If either `hypothesis` or `truth` are not a `SparseTensor`.
    """
    pass


def equal(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of (x == y) element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def erf(x, name=None):
    """
    [TensorFlow Docs]
    Computes the Gauss error function of `x` element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def erfc(x, name=None):
    """
    [TensorFlow Docs]
    Computes the complementary error function of `x` element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def exp(x, name=None):
    """
    [TensorFlow Docs]
    Computes exponential of x element-wise. \\(y = e^x\\).

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def expand_dims(input, dim, name=None):
    """
    [TensorFlow Docs]
    Inserts a dimension of 1 into a tensor's shape.

    Given a tensor `input`, this operation inserts a dimension of 1 at the
    dimension index `dim` of `input`'s shape. The dimension index `dim` starts at
    zero; if you specify a negative number for `dim` it is counted backward from
    the end.

    This operation is useful if you want to add a batch dimension to a single
    element. For example, if you have a single image of shape `[height, width,
    channels]`, you can make it a batch of 1 image with `expand_dims(image, 0)`,
    which will make the shape `[1, height, width, channels]`.

    Other examples:

    ```prettyprint
    # 't' is a tensor of shape [2]
    shape(expand_dims(t, 0)) ==> [1, 2]
    shape(expand_dims(t, 1)) ==> [2, 1]
    shape(expand_dims(t, -1)) ==> [2, 1]

    # 't2' is a tensor of shape [2, 3, 5]
    shape(expand_dims(t2, 0)) ==> [1, 2, 3, 5]
    shape(expand_dims(t2, 2)) ==> [2, 3, 1, 5]
    shape(expand_dims(t2, 3)) ==> [2, 3, 5, 1]
    ```

    This operation requires that:

    `-1-input.dims() <= dim <= input.dims()`

    This operation is related to `squeeze()`, which removes dimensions of
    size 1.

    Args:
        input: A `Tensor`.
        dim: A `Tensor` of type `int32`.
             0-D (scalar). Specifies the dimension index at which to
             expand the shape of `input`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        Contains the same data as `input`, but its shape has an additional
        dimension of size 1 added.
    """
    pass


def extract_image_patches(images, padding, ksizes=None, strides=None,
                          rates=None, name=None):
    """
    [TensorFlow Docs]
    Extract `patches` from `images` and puth them in the "depth" output dimension.

    Args:
        images: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
                4-D Tensor with shape `[batch, in_rows, in_cols, depth]`.
        padding: A `string` from: `"SAME", "VALID"`.
                 The type of padding algorithm to use.

            We specify the size-related attributes as:

            ksizes = [1, ksize_rows, ksize_cols, 1]
            strides = [1, strides_rows, strides_cols, 1]
            rates = [1, rates_rows, rates_cols, 1]
        ksizes: An optional list of `ints`. Defaults to `[]`.
                The size of the sliding window for each dimension of `images`.
                strides: An optional list of `ints`. Defaults to `[]`.
                1-D of length 4. How far the centers of two consecutive patches are in
                the images. Must be: `[1, stride_rows, stride_cols, 1]`.
                rates: An optional list of `ints`. Defaults to `[]`.
                1-D of length 4. Must be: `[1, rate_rows, rate_cols, 1]`. This is the
                input stride, specifying how far two consecutive patch samples are in the
                input. Equivalent to extracting patches with
                `patch_sizes_eff = patch_sizes + (patch_sizes - 1) * (rates - 1), followed by
                subsampling them spatially by a factor of `rates`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `images`.
        4-D Tensor with shape `[batch, out_rows, out_cols, ksize_rows *
        ksize_cols * depth]` containing image patches with size
        `ksize_rows x ksize_cols x depth` vectorized in the "depth" dimension.
    """
    pass


def fft(input, name=None):
    """
    [TensorFlow Docs]
    Compute the 1-dimensional discrete Fourier Transform.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 vector.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`. The 1D Fourier Transform of `input`.
    """
    pass


def fft2d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the 2-dimensional discrete Fourier Transform.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 matrix.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`. The 2D Fourier Transform of `input`.
    """
    pass


def fft3d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the 3-dimensional discrete Fourier Transform.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 3-D tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`. The 3D Fourier Transform of `input`.
    """
    pass


def fill(dims, value, name=None):
    """
    [TensorFlow Docs]
    Creates a tensor filled with a scalar value.

    This operation creates a tensor of shape `dims` and fills it with `value`.

    For example:

    ```prettyprint
    # Output tensor has shape [2, 3].
    fill([2, 3], 9) ==> [[9, 9, 9]
                       [9, 9, 9]]
    ```

    Args:
        dims: A `Tensor` of type `int32`.
              1-D. Represents the shape of the output tensor.
        value: A `Tensor`. 0-D (scalar). Value to fill the returned tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `value`.
    """
    pass


def floor(x, name=None):
    """
    [TensorFlow Docs]
    Returns element-wise largest integer not greater than x.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def floordiv(x, y, name=None):
    """
    [TensorFlow Docs]
    Divides `x / y` elementwise, rounding down for floating point.

    The same as `tf.div(x,y)` for integers, but uses `tf.floor(tf.div(x,y))` for
    floating point arguments so that the result is always an integer (though
    possibly an integer represented as floating point). This op is generated by
    `x // y` floor division in Python 3 and in Python 2.7 with
    `from __future__ import division`.

    Note that for efficiency, `floordiv` uses C semantics for negative numbers
    (unlike Python and Numpy).

    `x` and `y` must have the same type, and the result will have the same type
    as well.

    Args:
        x: `Tensor` numerator of real numeric type.
        y: `Tensor` denominator of real numeric type.
        name: A name for the operation (optional).

    Returns:
        `x / y` rounded down (except possibly towards zero for negative integers).

    Raises:
        TypeError: If the inputs are complex.
    """
    pass


def foldl(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
    """
    [TensorFlow Docs]
    foldl on the list of tensors unpacked from `elems` on dimension 0.

    This foldl operator repeatedly applies the callable `fn` to a sequence
    of elements from first to last. The elements are made of the tensors
    unpacked from `elems` on dimension 0. The callable fn takes two tensors as
    arguments. The first argument is the accumulated value computed from the
    preceding invocation of fn. If `initializer` is None, `elems` must contain
    at least one element, and its first element is used as the initializer.

    Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
    of the result tensor is fn(initializer, values[0]).shape`.

    Args:
        fn: The callable to be performed.
        elems: A tensor to be unpacked on dimension 0.
        initializer: (optional) The initial value for the accumulator.
        parallel_iterations: (optional) The number of iterations allowed to run
                             in parallel.
                             back_prop: (optional) True enables back propagation.
                             swap_memory: (optional) True enables GPU-CPU memory swapping.
        name: (optional) Name prefix for the returned tensors.

    Returns:
        A tensor resulting from applying `fn` consecutively to the list of tensors
        unpacked from `elems`, from first to last.

    Raises:
        TypeError: if `fn` is not callable.

    Example:
        ```python
        elems = [1, 2, 3, 4, 5, 6]
        sum = foldl(lambda a, x: a + x, elems)
        # sum == 21
        ```
    """
    pass


def foldr(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
    """
    [TensorFlow Docs]
    foldr on the list of tensors unpacked from `elems` on dimension 0.

    This foldr operator repeatedly applies the callable `fn` to a sequence
    of elements from last to first. The elements are made of the tensors
    unpacked from `elems`. The callable fn takes two tensors as arguments.
    The first argument is the accumulated value computed from the preceding
    invocation of fn. If `initializer` is None, `elems` must contain at least
    one element, and its first element is used as the initializer.

    Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
    of the result tensor is `fn(initializer, values[0]).shape`.

    Args:
        fn: The callable to be performed.
        elems: A tensor that is unpacked into a sequence of tensors to apply `fn`.
        initializer: (optional) The initial value for the accumulator.
        parallel_iterations: (optional) The number of iterations allowed to run
                             in parallel.
                             back_prop: (optional) True enables back propagation.
                             swap_memory: (optional) True enables GPU-CPU memory swapping.
        name: (optional) Name prefix for the returned tensors.

    Returns:
        A tensor resulting from applying `fn` consecutively to the list of tensors
        unpacked from `elems`, from last to first.

    Raises:
        TypeError: if `fn` is not callable.

    Example:
        ```python
        elems = [1, 2, 3, 4, 5, 6]
        sum = foldr(lambda a, x: a + x, elems)
        # sum == 21
        ```
    """
    pass


def gather(params, indices, validate_indices=None, name=None):
    """
    [TensorFlow Docs]
    Gather slices from `params` according to `indices`.

    `indices` must be an integer tensor of any dimension (usually 0-D or 1-D).
    Produces an output tensor with shape `indices.shape + params.shape[1:]` where:

            # Scalar indices
            output[:, ..., :] = params[indices, :, ... :]

            # Vector indices
            output[i, :, ..., :] = params[indices[i], :, ... :]

            # Higher rank indices
            output[i, ..., j, :, ... :] = params[indices[i, ..., j], :, ..., :]

    If `indices` is a permutation and `len(indices) == params.shape[0]` then
    this operation will permute `params` accordingly.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/Gather.png" alt>
    </div>

    Args:
        params: A `Tensor`.
        indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
        validate_indices: An optional `bool`. Defaults to `True`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `params`.
    """
    pass


def gather_nd(params, indices, name=None):
    """
    [TensorFlow Docs]
    Gather values from `params` according to `indices`.

    `indices` must be integer tensor, containing indices into `params`.
    It must be shape `[d_0, ..., d_N, R]` where `R` is the rank of `params`.
    The innermost dimension of `indices` (with length `R`) corresponds to the
    indices of `params`.

    Produces an output tensor with shape `[d_0, ..., d_{n-1}]` where:

            output[i, j, k, ...] = params[indices[i, j, k, ..., :]]

    e.g. for `indices` a matrix:

            output[i] = params[indices[i, :]]

    Args:
        params: A `Tensor`. R-D. The tensor from which to gather values.
        indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                 (N+1)-D. Index tensor having shape `[d_0, ..., d_N, R]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `params`.
        N-D. Values from `params` gathered from indices given by `indices`.
    """
    pass


def get_collection(key, scope=None):
    """
    [TensorFlow Docs]
    Wrapper for `Graph.get_collection()` using the default graph.

    See [`Graph.get_collection()`](../../api_docs/python/framework.md#Graph.get_collection)
    for more details.

    Args:
        key: The key for the collection. For example, the `GraphKeys` class
             contains many standard names for collections.
        scope: (Optional.) If supplied, the resulting list is filtered to include
               only items whose `name` attribute matches using `re.match`. Items
               without a `name` attribute are never returned if a scope is supplied and
               the choice or `re.match` means that a `scope` without special tokens
               filters by prefix.

    Returns:
        The list of values in the collection with the given `name`, or
        an empty list if no value has been added to that collection. The
        list contains the values in the order under which they were
        collected.
    """
    pass


def get_collection_ref(key):
    """
    [TensorFlow Docs]
    Wrapper for `Graph.get_collection_ref()` using the default graph.

    See [`Graph.get_collection_ref()`](../../api_docs/python/framework.md#Graph.get_collection_ref)
    for more details.

    Args:
        key: The key for the collection. For example, the `GraphKeys` class
             contains many standard names for collections.

    Returns:
        The list of values in the collection with the given `name`, or an empty
        list if no value has been added to that collection. Note that this returns
        the collection list itself, which can be modified in place to change the
        collection.
    """
    pass


def get_default_graph():
    """
    [TensorFlow Docs]
    Returns the default graph for the current thread.

    The returned graph will be the innermost graph on which a
    `Graph.as_default()` context has been entered, or a global default
    graph if none has been explicitly created.

    NOTE: The default graph is a property of the current thread. If you
    create a new thread, and wish to use the default graph in that
    thread, you must explicitly add a `with g.as_default():` in that
    thread's function.

    Returns:
        The default `Graph` being used in the current thread.
    """
    pass


def get_default_session():
    """
    [TensorFlow Docs]
    Returns the default session for the current thread.

    The returned `Session` will be the innermost session on which a
    `Session` or `Session.as_default()` context has been entered.

    NOTE: The default session is a property of the current thread. If you
    create a new thread, and wish to use the default session in that
    thread, you must explicitly add a `with sess.as_default():` in that
    thread's function.

    Returns:
        The default `Session` being used in the current thread.
    """
    pass


def get_seed(op_seed):
    """
    [TensorFlow Docs]
    Returns the local seeds an operation should use given an op-specific seed.

    Given operation-specific seed, `op_seed`, this helper function returns two
    seeds derived from graph-level and op-level seeds. Many random operations
    internally use the two seeds to allow user to change the seed globally for a
    graph, or for only specific operations.

    For details on how the graph-level seed interacts with op seeds, see
    [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed).

    Args:
        op_seed: integer.

    Returns:
        A tuple of two integers that should be used for the local seed of this
        operation.
    """
    pass


def get_session_handle(data, name=None):
    """
    [TensorFlow Docs]
    Return the handle of `data`.

    This is EXPERIMENTAL and subject to change.

    Keep `data` "in-place" in the runtime and create a handle that can be
    used to retrieve `data` in a subsequent run().

    Combined with `get_session_tensor`, we can keep a tensor produced in
    one run call in place, and use it as the input in a future run call.
    Below is a simple example:

    ```python
    c = tf.mul(a, b)
    h = tf.get_session_handle(c)
    h = sess.run(h)

    p, a = tf.get_session_tensor(tf.float32)
    b = tf.mul(a, 10)
    c = sess.run(b, feed_dict={p: h.handle})
    ```

    Args:
        data: A tensor to be stored in the session.
        name: Optional name prefix for the return tensor.

    Returns:
        A scalar string tensor representing a unique handle for `data`.

    Raises:
        TypeError: if `data` is not a Tensor.
    """
    pass


def get_session_tensor(dtype, name=None):
    """
    [TensorFlow Docs]
    Get the tensor of type `dtype` by feeding a tensor handle.

    This is EXPERIMENTAL and subject to change.

    Get the value of the tensor from a tensor handle. The tensor
    is produced in a previous run() and stored in the state of the
    session.

    Args:
        dtype: The type of the output tensor.
        name: Optional name prefix for the return tensor.

    Returns:
        A pair of tensors. The first is a placeholder for feeding a
        tensor handle and the second is the tensor in the session state
        keyed by the tensor handle.
    """
    pass


def get_variable(name, shape=None, dtype=dtypes.float32, initializer=None,
                 regularizer=None, trainable=True, collections=None,
                 caching_device=None, partitioner=None, validate_shape=True):
    """
    [TensorFlow Docs]
    Gets an existing variable with these parameters or create a new one.

    This function prefixes the name with the current variable scope
    and performs reuse checks. See the
    [Variable Scope How To](../../how_tos/variable_scope/index.md)
    for an extensive description of how reusing works. Here is a basic example:

    ```python
    with tf.variable_scope("foo"):
            v = tf.get_variable("v", [1])  # v.name == "foo/v:0"
            w = tf.get_variable("w", [1])  # w.name == "foo/w:0"
    with tf.variable_scope("foo", reuse=True)
            v1 = tf.get_variable("v")  # The same as v above.
    ```

    If initializer is `None` (the default), the default initializer passed in
    the variable scope will be used. If that one is `None` too, a
    `UniformUnitScalingInitializer` will be used. The initializer can also be
    a Tensor, in which case the variable is initialized to this value and shape.

    Similarly, if the regularizer is `None` (the default), the default regularizer
    passed in the variable scope will be used (if that is `None` too,
    then by default no regularization is performed).

    If a partitioner is provided, first a sharded `Variable` is created
    via `_get_partitioned_variable`, and the return value is a
    `Tensor` composed of the shards concatenated along the partition axis.

    Some useful partitioners are available. See, e.g.,
    `variable_axis_size_partitioner`.

    Args:
        name: The name of the new or existing variable.
        shape: Shape of the new or existing variable.
        dtype: Type of the new or existing variable (defaults to `DT_FLOAT`).
        initializer: Initializer for the variable if one is created.
                     regularizer: A (Tensor -> Tensor or None) function; the result of
                     applying it on a newly created variable will be added to the collection
                     GraphKeys.REGULARIZATION_LOSSES and can be used for regularization.
                     trainable: If `True` also add the variable to the graph collection
                     `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
        collections: List of graph collections keys to add the Variable to.
                     Defaults to `[GraphKeys.VARIABLES]` (see tf.Variable).
                     caching_device: Optional device string or function describing where the
                     Variable should be cached for reading. Defaults to the Variable's
                     device. If not `None`, caches on another device. Typical use is to
                     cache on the device where the Ops using the Variable reside, to
                     deduplicate copying through `Switch` and other conditional statements.
                     partitioner: Optional callable that accepts a fully defined `TensorShape`
                     and `dtype` of the Variable to be created, and returns a list of
                     partitions for each axis (currently only one axis can be partitioned).
        validate_shape: If False, allows the variable to be initialized with a
                        value of unknown shape. If True, the default, the shape of initial_value
                        must be known.

    Returns:
        The created or existing variable.

    Raises:
        ValueError: when creating a new variable and shape is not declared,
            or when violating reuse during variable creation. Reuse is set inside
            `variable_scope`.
    """
    pass


def get_variable_scope():
    """
    [TensorFlow Docs]
    Returns the current variable scope."""
    pass


def global_norm(t_list, name=None):
    """
    [TensorFlow Docs]
    Computes the global norm of multiple tensors.

    Given a tuple or list of tensors `t_list`, this operation returns the
    global norm of the elements in all tensors in `t_list`. The global norm is
    computed as:

    `global_norm = sqrt(sum([l2norm(t)**2 for t in t_list]))`

    Any entries in `t_list` that are of type None are ignored.

    Args:
        t_list: A tuple or list of mixed `Tensors`, `IndexedSlices`, or None.
        name: A name for the operation (optional).

    Returns:
        A 0-D (scalar) `Tensor` of type `float`.

    Raises:
        TypeError: If `t_list` is not a sequence.
    """
    pass


def gradients(ys,
              xs,
              grad_ys=None,
              name="gradients",
              colocate_gradients_with_ops=False,
              gate_gradients=False,
              aggregation_method=None):
    """
    [TensorFlow Docs]
    Constructs symbolic partial derivatives of sum of `ys` w.r.t. x in `xs`.

    `ys` and `xs` are each a `Tensor` or a list of tensors. `grad_ys`
    is a list of `Tensor`, holding the gradients received by the
    `ys`. The list must be the same length as `ys`.

    `gradients()` adds ops to the graph to output the partial
    derivatives of `ys` with respect to `xs`. It returns a list of
    `Tensor` of length `len(xs)` where each tensor is the `sum(dy/dx)`
    for y in `ys`.

    `grad_ys` is a list of tensors of the same length as `ys` that holds
    the initial gradients for each y in `ys`. When `grad_ys` is None,
    we fill in a tensor of '1's of the shape of y for each y in `ys`. A
    user can provide their own initial `grad_ys` to compute the
    derivatives using a different initial gradient for each y (e.g., if
    one wanted to weight the gradient differently for each value in
    each y).

    Args:
        ys: A `Tensor` or list of tensors to be differentiated.
        xs: A `Tensor` or list of tensors to be used for differentiation.
        grad_ys: Optional. A `Tensor` or list of tensors the same size as
                 `ys` and holding the gradients computed for each y in `ys`.
        name: Optional name to use for grouping all the gradient ops together.
              defaults to 'gradients'.
              colocate_gradients_with_ops: If True, try colocating gradients with
              the corresponding op.
              gate_gradients: If True, add a tuple around the gradients returned
              for an operations. This avoids some race conditions.
              aggregation_method: Specifies the method used to combine gradient terms.
              Accepted values are constants defined in the class `AggregationMethod`.

    Returns:
        A list of `sum(dy/dx)` for each x in `xs`.

    Raises:
        LookupError: if one of the operations between `x` and `y` does not
            have a registered gradient function.
        ValueError: if the arguments are invalid.

    """
    pass


def greater(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of (x > y) element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def greater_equal(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of (x >= y) element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def group(*inputs, **kwargs):
    """
    [TensorFlow Docs]
    Create an op that groups multiple operations.

    When this op finishes, all ops in `input` have finished. This op has no
    output.

    See also `tuple` and `with_dependencies`.

    Args:
        *inputs: Zero or more tensors to group.
        **kwargs: Optional parameters to pass when constructing the NodeDef.
        name: A name for this operation (optional).

    Returns:
        An Operation that executes all its inputs.

    Raises:
        ValueError: If an unknown keyword argument is provided.
    """
    pass


def histogram_fixed_width(values,
                          value_range,
                          nbins=100,
                          dtype=dtypes.int32,
                          name=None):
    """
    [TensorFlow Docs]
    Return histogram of values.

    Given the tensor `values`, this operation returns a rank 1 histogram counting
    the number of entries in `values` that fell into every bin. The bins are
    equal width and determined by the arguments `value_range` and `nbins`.

    Args:
        values: Numeric `Tensor`.
        value_range: Shape [2] `Tensor`. new_values <= value_range[0] will be
                     mapped to hist[0], values >= value_range[1] will be mapped to hist[-1].
                     Must be same dtype as new_values.
        nbins: Scalar `int32 Tensor`. Number of histogram bins.
        dtype: dtype for returned histogram.
        name: A name for this operation (defaults to 'histogram_fixed_width').

    Returns:
        A 1-D `Tensor` holding histogram of values.

    Examples:

    ```python
    # Bins will be: (-inf, 1), [1, 2), [2, 3), [3, 4), [4, inf)
    nbins = 5
    value_range = [0.0, 5.0]
    new_values = [-1.0, 0.0, 1.5, 2.0, 5.0, 15]

    with tf.default_session() as sess:
        hist = tf.histogram_fixed_width(new_values, value_range, nbins=5)
        variables.initialize_all_variables().run()
        sess.run(hist) => [2, 1, 1, 0, 2]
    ```
    """
    pass


def histogram_summary(tag, values, collections=None, name=None):
    """
    [TensorFlow Docs]
    Outputs a `Summary` protocol buffer with a histogram.

    The generated
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    has one summary value containing a histogram for `values`.

    This op reports an `InvalidArgument` error if any value is not finite.

    Args:
        tag: A `string` `Tensor`. 0-D. Tag to use for the summary value.
        values: A real numeric `Tensor`. Any shape. Values to use to
                build the histogram.
        collections: Optional list of graph collections keys. The new summary op is
                     added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
        name: A name for the operation (optional).

    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol
        buffer.
    """
    pass


def identity(input, name=None):
    """
    [TensorFlow Docs]
    Return a tensor with the same shape and contents as the input tensor or value.

    Args:
        input: A `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def ifft(input, name=None):
    """
    [TensorFlow Docs]
    Compute the inverse 1-dimensional discrete Fourier Transform.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 vector.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        The inverse 1D Fourier Transform of `input`.
    """
    pass


def ifft2d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the inverse 2-dimensional discrete Fourier Transform.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 matrix.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        The inverse 2D Fourier Transform of `input`.
    """
    pass


def ifft3d(input, name=None):
    """
    [TensorFlow Docs]
    Compute the inverse 3-dimensional discrete Fourier Transform.

    Args:
        input: A `Tensor` of type `complex64`. A complex64 3-D tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `complex64`.
        The inverse 3D Fourier Transform of `input`.
    """
    pass


def igamma(a, x, name=None):
    """
    [TensorFlow Docs]
    Compute the lower regularized incomplete Gamma function `Q(a, x)`.

    The lower regularized incomplete Gamma function is defined as:

    ```
    P(a, x) = gamma(a, x) / Gamma(x) = 1 - Q(a, x)
    ```
    where
    ```
    gamma(a, x) = int_{0}^{x} t^{a-1} exp(-t) dt
    ```
    is the lower incomplete Gamma function.

    Note, above `Q(a, x)` (`Igammac`) is the upper regularized complete
    Gamma function.

    Args:
        a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        x: A `Tensor`. Must have the same type as `a`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `a`.
    """
    pass


def igammac(a, x, name=None):
    """
    [TensorFlow Docs]
    Compute the upper regularized incomplete Gamma function `Q(a, x)`.

    The upper regularized incomplete Gamma function is defined as:

    ```
    Q(a, x) = Gamma(a, x) / Gamma(x) = 1 - P(a, x)
    ```
    where
    ```
    Gamma(a, x) = int_{x}^{\infty} t^{a-1} exp(-t) dt
    ```
    is the upper incomplete Gama function.

    Note, above `P(a, x)` (`Igamma`) is the lower regularized complete
    Gamma function.

    Args:
        a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        x: A `Tensor`. Must have the same type as `a`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `a`.
    """
    pass


def imag(input, name=None):
    """
    [TensorFlow Docs]
    Returns the imaginary part of a complex number.

    Given a tensor `input` of complex numbers, this operation returns a tensor of
    type `float` or `double` that is the imaginary part of each element in
    `input`. All elements in `input` must be complex numbers of the form \\(a +
    bj\\), where *a* is the real part and *b* is the imaginary part returned by
    this operation.

    For example:

    ```
    # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    tf.imag(input) ==> [4.75, 5.75]
    ```

    Args:
        input: A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `float` or `double`.
    """
    pass


def image_summary(tag, tensor, max_images=3, collections=None, name=None):
    """
    [TensorFlow Docs]
    Outputs a `Summary` protocol buffer with images.

    The summary has up to `max_images` summary values containing images. The
    images are built from `tensor` which must be 4-D with shape `[batch_size,
    height, width, channels]` and where `channels` can be:

    *  1: `tensor` is interpreted as Grayscale.
    *  3: `tensor` is interpreted as RGB.
    *  4: `tensor` is interpreted as RGBA.

    The images have the same number of channels as the input tensor. For float
    input, the values are normalized one image at a time to fit in the range
    `[0, 255]`. `uint8` values are unchanged. The op uses two different
    normalization algorithms:

    *  If the input values are all positive, they are rescaled so the largest one
     is 255.

    *  If any input value is negative, the values are shifted so input value 0.0
     is at 127. They are then rescaled so that either the smallest value is 0,
     or the largest one is 255.

    The `tag` argument is a scalar `Tensor` of type `string`. It is used to
    build the `tag` of the summary values:

    *  If `max_images` is 1, the summary value tag is '*tag*/image'.
    *  If `max_images` is greater than 1, the summary value tags are
     generated sequentially as '*tag*/image/0', '*tag*/image/1', etc.

    Args:
        tag: A scalar `Tensor` of type `string`. Used to build the `tag`
             of the summary values.
        tensor: A 4-D `uint8` or `float32` `Tensor` of shape `[batch_size, height,
                width, channels]` where `channels` is 1, 3, or 4.
        max_images: Max number of batch elements to generate images for.
        collections: Optional list of ops.GraphKeys. The collections to add the
                     summary to. Defaults to [ops.GraphKeys.SUMMARIES]
        name: A name for the operation (optional).

    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol
        buffer.
    """
    pass


def import_graph_def(graph_def, input_map=None, return_elements=None,
                     name=None, op_dict=None, producer_op_list=None):
    """
    [TensorFlow Docs]
    Imports the TensorFlow graph in `graph_def` into the Python `Graph`.

    This function provides a way to import a serialized TensorFlow
    [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
    protocol buffer, and extract individual objects in the `GraphDef` as
    [`Tensor`](#Tensor) and [`Operation`](#Operation) objects. See
    [`Graph.as_graph_def()`](#Graph.as_graph_def) for a way to create a
    `GraphDef` proto.

    Args:
        graph_def: A `GraphDef` proto containing operations to be imported into
                   the default graph.
        input_map: A dictionary mapping input names (as strings) in `graph_def`
                   to `Tensor` objects. The values of the named input tensors in the
                   imported graph will be re-mapped to the respective `Tensor` values.
                   return_elements: A list of strings containing operation names in
                   `graph_def` that will be returned as `Operation` objects; and/or
                   tensor names in `graph_def` that will be returned as `Tensor` objects.
        name: (Optional.) A prefix that will be prepended to the names in
              `graph_def`. Defaults to `"import"`.
              op_dict: (Optional.) A dictionary mapping op type names to `OpDef` protos.
              Must contain an `OpDef` proto for each op type named in `graph_def`.
              If omitted, uses the `OpDef` protos registered in the global registry.
              producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
              list of `OpDef`s used by the producer of the graph. If provided, attrs
              for ops in `graph_def` that are not in `op_dict` that have their default
              value according to `producer_op_list` will be removed. This will allow
              some more `GraphDef`s produced by later binaries to be accepted by
              earlier binaries.

    Returns:
        A list of `Operation` and/or `Tensor` objects from the imported graph,
        corresponding to the names in `return_elements`.

    Raises:
        TypeError: If `graph_def` is not a `GraphDef` proto,
            `input_map` is not a dictionary mapping strings to `Tensor` objects,
            or `return_elements` is not a list of strings.
        ValueError: If `input_map`, or `return_elements` contains names that
            do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
            it refers to an unknown tensor).
    """
    pass


def initialize_all_tables(name="init_all_tables"):
    """
    [TensorFlow Docs]
    Returns an Op that initializes all tables of the default graph.

    Args:
        name: Optional name for the initialization op.

    Returns:
        An Op that initializes all tables. Note that if there are
        not tables the returned Op is a NoOp.
    """
    pass


def initialize_all_variables():
    """
    [TensorFlow Docs]
    Returns an Op that initializes all variables.

    This is just a shortcut for `initialize_variables(all_variables())`

    Returns:
        An Op that initializes all variables in the graph.
    """
    pass


def initialize_local_variables():
    """
    [TensorFlow Docs]
    Returns an Op that initializes all local variables.

    This is just a shortcut for `initialize_variables(local_variables())`

    Returns:
        An Op that initializes all local variables in the graph.
    """
    pass


def initialize_variables(var_list, name="init"):
    """
    [TensorFlow Docs]
    Returns an Op that initializes a list of variables.

    After you launch the graph in a session, you can run the returned Op to
    initialize all the variables in `var_list`. This Op runs all the
    initializers of the variables in `var_list` in parallel.

    Calling `initialize_variables()` is equivalent to passing the list of
    initializers to `Group()`.

    If `var_list` is empty, however, the function still returns an Op that can
    be run. That Op just has no effect.

    Args:
        var_list: List of `Variable` objects to initialize.
        name: Optional name for the returned operation.

    Returns:
        An Op that run the initializers of all the specified variables.
    """
    pass


def inv(x, name=None):
    """
    [TensorFlow Docs]
    Computes the reciprocal of x element-wise.

    I.e., \\(y = 1 / x\\).

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def invert_permutation(x, name=None):
    """
    [TensorFlow Docs]
    Computes the inverse permutation of a tensor.

    This operation computes the inverse of an index permutation. It takes a 1-D
    integer tensor `x`, which represents the indices of a zero-based array, and
    swaps each value with its index position. In other words, for an output tensor
    `y` and an input tensor `x`, this operation computes the following:

    `y[x[i]] = i for i in [0, 1, ..., len(x) - 1]`

    The values must include 0. There can be no duplicate values or negative values.

    For example:

    ```prettyprint
    # tensor `x` is [3, 4, 0, 2, 1]
    invert_permutation(x) ==> [2, 4, 3, 0, 1]
    ```

    Args:
        x: A `Tensor` of type `int32`. 1-D.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int32`. 1-D.
    """
    pass


def is_finite(x, name=None):
    """
    [TensorFlow Docs]
    Returns which elements of x are finite.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def is_inf(x, name=None):
    """
    [TensorFlow Docs]
    Returns which elements of x are Inf.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def is_nan(x, name=None):
    """
    [TensorFlow Docs]
    Returns which elements of x are NaN.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def is_non_decreasing(x, name=None):
    """
    [TensorFlow Docs]
    Returns `True` if `x` is non-decreasing.

    Elements of `x` are compared in row-major order. The tensor `[x[0],...]`
    is non-decreasing if for every adjacent pair we have `x[i] <= x[i+1]`.
    If `x` has less than two elements, it is trivially non-decreasing.

    See also: `is_strictly_increasing`

    Args:
        x: Numeric `Tensor`.
        name: A name for this operation (optional). Defaults to "is_non_decreasing"

    Returns:
        Boolean `Tensor`, equal to `True` iff `x` is non-decreasing.

    Raises:
        TypeError: if `x` is not a numeric tensor.
    """
    pass


def is_strictly_increasing(x, name=None):
    """
    [TensorFlow Docs]
    Returns `True` if `x` is strictly increasing.

    Elements of `x` are compared in row-major order. The tensor `[x[0],...]`
    is strictly increasing if for every adjacent pair we have `x[i] < x[i+1]`.
    If `x` has less than two elements, it is trivially strictly increasing.

    See also: `is_non_decreasing`

    Args:
        x: Numeric `Tensor`.
        name: A name for this operation (optional).
              Defaults to "is_strictly_increasing"

    Returns:
        Boolean `Tensor`, equal to `True` iff `x` is strictly increasing.

    Raises:
        TypeError: if `x` is not a numeric tensor.
    """
    pass


def is_variable_initialized(variable):
    """
    [TensorFlow Docs]
    Tests if a variable has been initialized.

    Args:
        variable: A `Variable`.

    Returns:
        Returns a scalar boolean Tensor, `True` if the variable has been
        initialized, `False` otherwise.
    """
    pass


def lbeta(x, name='lbeta'):
    """
    [TensorFlow Docs]
    Computes `ln(|Beta(x)|)`, reducing along the last dimension.

    Given one-dimensional `z = [z_0,...,z_{K-1}]`, we define

    ```Beta(z) = \prod_j Gamma(z_j) / Gamma(\sum_j z_j)```

    And for `n + 1` dimensional `x` with shape `[N1, ..., Nn, K]`, we define
    `lbeta(x)[i1, ..., in] = Log(|Beta(x[i1, ..., in, :])|)`. In other words,
    the last dimension is treated as the `z` vector.

    Note that if `z = [u, v]`, then
    `Beta(z) = int_0^1 t^{u-1} (1 - t)^{v-1} dt`, which defines the traditional
    bivariate beta function.

    Args:
        x: A rank `n + 1` `Tensor` with type `float`, or `double`.
        name: A name for the operation (optional).

    Returns:
        The logarithm of `|Beta(x)|` reducing along the last dimension.

    Raises:
        ValueError: If `x` is empty with rank one or less.
    """
    pass


def less(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of (x < y) element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def less_equal(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of (x <= y) element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def lgamma(x, name=None):
    """
    [TensorFlow Docs]
    Computes the log of the absolute value of `Gamma(x)` element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def lin_space(start, stop, num, name=None):
    """
    [TensorFlow Docs]
    Generates values in an interval.

    A sequence of `num` evenly-spaced values are generated beginning at `start`.
    If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
    so that the last one is exactly `stop`.

    For example:

    ```
    tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
    ```

    Args:
        start: A `Tensor`. Must be one of the following types: `float32`, `float64`.
               First entry in the range.
        stop: A `Tensor`. Must have the same type as `start`.
              Last entry in the range.
        num: A `Tensor` of type `int32`. Number of values to generate.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `start`. 1-D. The generated values.
    """
    pass


def lin_space(start, stop, num, name=None):
    """
    [TensorFlow Docs]
    Generates values in an interval.

    A sequence of `num` evenly-spaced values are generated beginning at `start`.
    If `num > 1`, the values in the sequence increase by `stop - start / num - 1`,
    so that the last one is exactly `stop`.

    For example:

    ```
    tf.linspace(10.0, 12.0, 3, name="linspace") => [ 10.0  11.0  12.0]
    ```

    Args:
        start: A `Tensor`. Must be one of the following types: `float32`, `float64`.
               First entry in the range.
        stop: A `Tensor`. Must have the same type as `start`.
              Last entry in the range.
        num: A `Tensor` of type `int32`. Number of values to generate.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `start`. 1-D. The generated values.
    """
    pass


def list_diff(x, y, name=None):
    """
    [TensorFlow Docs]
    Computes the difference between two lists of numbers or strings.

    Given a list `x` and a list `y`, this operation returns a list `out` that
    represents all values that are in `x` but not in `y`. The returned list `out`
    is sorted in the same order that the numbers appear in `x` (duplicates are
    preserved). This operation also returns a list `idx` that represents the
    position of each `out` element in `x`. In other words:

    `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

    For example, given this input:

    ```prettyprint
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 3, 5]
    ```

    This operation would return:

    ```prettyprint
    out ==> [2, 4, 6]
    idx ==> [1, 3, 5]
    ```

    Args:
        x: A `Tensor`. 1-D. Values to keep.
        y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
        name: A name for the operation (optional).

    Returns:
        A tuple of `Tensor` objects (out, idx).
        out: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
        idx: A `Tensor` of type `int32`. 1-D. Positions of `x` values preserved in `out`.
    """
    pass


def list_diff(x, y, name=None):
    """
    [TensorFlow Docs]
    Computes the difference between two lists of numbers or strings.

    Given a list `x` and a list `y`, this operation returns a list `out` that
    represents all values that are in `x` but not in `y`. The returned list `out`
    is sorted in the same order that the numbers appear in `x` (duplicates are
    preserved). This operation also returns a list `idx` that represents the
    position of each `out` element in `x`. In other words:

    `out[i] = x[idx[i]] for i in [0, 1, ..., len(out) - 1]`

    For example, given this input:

    ```prettyprint
    x = [1, 2, 3, 4, 5, 6]
    y = [1, 3, 5]
    ```

    This operation would return:

    ```prettyprint
    out ==> [2, 4, 6]
    idx ==> [1, 3, 5]
    ```

    Args:
        x: A `Tensor`. 1-D. Values to keep.
        y: A `Tensor`. Must have the same type as `x`. 1-D. Values to remove.
        name: A name for the operation (optional).

    Returns:
        A tuple of `Tensor` objects (out, idx).
        out: A `Tensor`. Has the same type as `x`. 1-D. Values present in `x` but not in `y`.
        idx: A `Tensor` of type `int32`. 1-D. Positions of `x` values preserved in `out`.
    """
    pass


def load_file_system_library(library_filename):
    """
    [TensorFlow Docs]
    Loads a TensorFlow plugin, containing file system implementation.

    Pass `library_filename` to a platform-specific mechanism for dynamically
    loading a library. The rules for determining the exact location of the
    library are platform-specific and are not documented here.

    Args:
        library_filename: Path to the plugin.
                          Relative or absolute filesystem path to a dynamic library file.

    Returns:
        None.

    Raises:
        RuntimeError: when unable to load the library.
    """
    pass


def load_op_library(library_filename):
    """
    [TensorFlow Docs]
    Loads a TensorFlow plugin, containing custom ops and kernels.

    Pass "library_filename" to a platform-specific mechanism for dynamically
    loading a library. The rules for determining the exact location of the
    library are platform-specific and are not documented here.

    Args:
        library_filename: Path to the plugin.
                          Relative or absolute filesystem path to a dynamic library file.

    Returns:
        A python module containing the Python wrappers for Ops defined in
        the plugin.

    Raises:
        RuntimeError: when unable to load the library or get the python wrappers.
    """
    pass


def local_variables():
    """
    [TensorFlow Docs]
    Returns all variables created with collection=[LOCAL_VARIABLES].

    Returns:
        A list of local Variable objects.
    """
    pass


def log(x, name=None):
    """
    [TensorFlow Docs]
    Computes natural logarithm of x element-wise.

    I.e., \\(y = \log_e x\\).

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def logical_and(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of x AND y element-wise.

    Args:
        x: A `Tensor` of type `bool`.
        y: A `Tensor` of type `bool`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def logical_not(x, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of NOT x element-wise.

    Args:
        x: A `Tensor` of type `bool`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def logical_or(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of x OR y element-wise.

    Args:
        x: A `Tensor` of type `bool`.
        y: A `Tensor` of type `bool`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def logical_xor(x, y, name="LogicalXor"):
    """
    [TensorFlow Docs]
    x ^ y = (x | y) & ~(x & y)."""
    pass


def make_template(name_, func_, create_scope_now_=False, **kwargs):
    """
    [TensorFlow Docs]
    Given an arbitrary function, wrap it so that it does variable sharing.

    This wraps `func_` in a Template and partially evaluates it. Templates are
    functions that create variables the first time they are called and reuse them
    thereafter. In order for `func_` to be compatible with a `Template` it must
    have the following properties:

    * The function should create all trainable variables and any variables that
     should be reused by calling `tf.get_variable`. If a trainable variable is
     created using `tf.Variable`, then a ValueError will be thrown. Variables
     that are intended to be locals can be created by specifying
     `tf.Variable(..., trainable=false)`.
    * The function may use variable scopes and other templates internally to
            create and reuse variables, but it shouldn't use `tf.get_variables` to
            capture variables that are defined outside of the scope of the function.
    * Internal scopes and variable names should not depend on any arguments that
            are not supplied to `make_template`. In general you will get a ValueError
            telling you that you are trying to reuse a variable that doesn't exist
            if you make a mistake.

    In the following example, both `z` and `w` will be scaled by the same `y`. It
    is important to note that if we didn't assign `scalar_name` and used a
    different name for z and w that a `ValueError` would be thrown because it
    couldn't reuse the variable.

    ```python
    def my_op(x, scalar_name):
        var1 = tf.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.constant_initializer(1))
        return x * var1

    scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')

    z = scale_by_y(input1)
    w = scale_by_y(input2)
    ```

    As a safe-guard, the returned function will raise a `ValueError` after the
    first call if trainable variables are created by calling `tf.Variable`.

    If all of these are true, then 2 properties are enforced by the template:

    1. Calling the same template multiple times will share all non-local
            variables.
    2. Two different templates are guaranteed to be unique, unless you reenter the
            same variable scope as the initial definition of a template and redefine
            it. An examples of this exception:

    ```python
    def my_op(x, scalar_name):
        var1 = tf.get_variable(scalar_name,
                           shape=[],
                           initializer=tf.constant_initializer(1))
        return x * var1

    with tf.variable_scope('scope') as vs:
        scale_by_y = tf.make_template('scale_by_y', my_op, scalar_name='y')
        z = scale_by_y(input1)
        w = scale_by_y(input2)

    # Creates a template that reuses the variables above.
    with tf.variable_scope(vs, reuse=True):
        scale_by_y2 = tf.make_template('scale_by_y', my_op, scalar_name='y')
        z2 = scale_by_y2(input1)
        w2 = scale_by_y2(input2)
    ```

    Depending on the value of `create_scope_now_`, the full variable scope may be
    captured either at the time of first call or at the time of construction. If
    this option is set to True, then all Tensors created by repeated calls to the
    template will have an extra trailing _N+1 to their name, as the first time the
    scope is entered in the Template constructor no Tensors are created.

    Note: `name_`, `func_` and `create_scope_now_` have a trailing underscore to
    reduce the likelihood of collisions with kwargs.

    Args:
        name_: A name for the scope created by this template. If necessary, the name
               will be made unique by appending `_N` to the name.
        func_: The function to wrap.
        create_scope_now_: Boolean controlling whether the scope should be created
                           when the template is constructed or when the template is called. Default
                           is False, meaning the scope is created when the template is called.
                           **kwargs: Keyword arguments to apply to `func_`.

    Returns:
        A function to encapsulate a set of variables which should be created once
        and reused. An enclosing scope will created, either where `make_template`
        is called, or wherever the result is called, depending on the value of
        `create_scope_now_`. Regardless of the value, the first time the template
        is called it will enter the scope with no reuse, and call `func_` to create
        variables, which are guaranteed to be unique. All subsequent calls will
        re-enter the scope and reuse those variables.

    Raises:
        ValueError: if the name is None.
    """
    pass


def map_fn(fn, elems, dtype=None, parallel_iterations=10, back_prop=True,
           swap_memory=False, name=None):
    """
    [TensorFlow Docs]
    map on the list of tensors unpacked from `elems` on dimension 0.

    This map operator repeatedly applies the callable `fn` to a sequence of
    elements from first to last. The elements are made of the tensors unpacked
    from `elems`. `dtype` is the data type of the return value of `fn`. Users
    must provide `dtype` if it is different from the data type of `elems`.

    Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
    of the result tensor is `[len(values)] + fn(values[0]).shape`.

    Args:
        fn: The callable to be performed.
        elems: A tensor to be unpacked to apply `fn`.
        dtype: (optional) The output type of `fn`.
        parallel_iterations: (optional) The number of iterations allowed to run
                             in parallel.
                             back_prop: (optional) True enables back propagation.
                             swap_memory: (optional) True enables GPU-CPU memory swapping.
        name: (optional) Name prefix for the returned tensors.

    Returns:
        A tensor that packs the results of applying `fn` to the list of tensors
        unpacked from `elems`, from first to last.

    Raises:
        TypeError: if `fn` is not callable.

    Example:
        ```python
        elems = [1, 2, 3, 4, 5, 6]
        squares = map_fn(lambda x: x * x, elems)
        # squares == [1, 4, 9, 16, 25, 36]
        ```
    """
    pass


def matching_files(pattern, name=None):
    """
    [TensorFlow Docs]
    Returns the set of files matching a pattern.

    Note that this routine only supports wildcard characters in the
    basename portion of the pattern, not in the directory portion.

    Args:
        pattern: A `Tensor` of type `string`. A (scalar) shell wildcard pattern.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `string`. A vector of matching filenames.
    """
    pass


def matmul(a, b,
           transpose_a=False, transpose_b=False,
           a_is_sparse=False, b_is_sparse=False,
           name=None):
    """
    [TensorFlow Docs]
    Multiplies matrix `a` by matrix `b`, producing `a` * `b`.

    The inputs must be two-dimensional matrices, with matching inner dimensions,
    possibly after transposition.

    Both matrices must be of the same type. The supported types are:
    `float`, `double`, `int32`, `complex64`.

    Either matrix can be transposed on the fly by setting the corresponding flag
    to `True`. This is `False` by default.

    If one or both of the matrices contain a lot of zeros, a more efficient
    multiplication algorithm can be used by setting the corresponding
    `a_is_sparse` or `b_is_sparse` flag to `True`. These are `False` by default.

    For example:

    ```python
    # 2-D tensor `a`
    a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) => [[1. 2. 3.]
                                                        [4. 5. 6.]]
    # 2-D tensor `b`
    b = tf.constant([7, 8, 9, 10, 11, 12], shape=[3, 2]) => [[7. 8.]
                                                           [9. 10.]
                                                           [11. 12.]]
    c = tf.matmul(a, b) => [[58 64]
                          [139 154]]
    ```

    Args:
        a: `Tensor` of type `float`, `double`, `int32` or `complex64`.
        b: `Tensor` with same type as `a`.
        transpose_a: If `True`, `a` is transposed before multiplication.
                     transpose_b: If `True`, `b` is transposed before multiplication.
                     a_is_sparse: If `True`, `a` is treated as a sparse matrix.
                     b_is_sparse: If `True`, `b` is treated as a sparse matrix.
        name: Name for the operation (optional).

    Returns:
        A `Tensor` of the same type as `a`.
    """
    pass


def matrix_determinant(input, name=None):
    """
    [TensorFlow Docs]
    Calculates the determinant of a square matrix.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
               A tensor of shape `[M, M]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        A scalar, equal to the determinant of the input.
    """
    pass


def matrix_inverse(input, adjoint=None, name=None):
    """
    [TensorFlow Docs]
    Calculates the inverse of a square invertible matrix or its adjoint (conjugate

    transpose).

    The op uses LU decomposition with partial pivoting to compute the inverse.

    If the matrix is not invertible there is no guarantee what the op does. It
    may detect the condition and raise an exception or it may simply return a
    garbage result.

    Args:
        input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
               Shape is `[M, M]`.
        adjoint: An optional `bool`. Defaults to `False`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        Shape is `[M, M]`. If `adjoint` is `False` then `output` contains the
        matrix inverse of `input`. If `adjoint` is `True` then `output` contains the
        matrix inverse of the adjoint of `input`.
    """
    pass


def matrix_solve(matrix, rhs, adjoint=None, name=None):
    """
    [TensorFlow Docs]
    Solves a system of linear equations. Checks for invertibility.

    Args:
        matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
                Shape is `[M, M]`.
        rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
        adjoint: An optional `bool`. Defaults to `False`.
                 Boolean indicating whether to solve with `matrix` or its adjoint.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `matrix`.
        Shape is `[M, K]`. If `adjoint` is `False` then `output` that solves
        `matrix` * `output` = `rhs`. If `adjoint` is `True` then `output` that solves
        `adjoint(matrix)` * `output` = `rhs`.
    """
    pass


def matrix_solve_ls(matrix, rhs, l2_regularizer=0.0, fast=True, name=None):
    """
    [TensorFlow Docs]
    Solves a linear least-squares problem.

    Below we will use the following notation
    `matrix`=\\(A \in \Re^{m \times n}\\),
    `rhs`=\\(B  \in \Re^{m \times k}\\),
    `output`=\\(X  \in \Re^{n \times k}\\),
    `l2_regularizer`=\\(\lambda\\).

    If `fast` is `True`, then the solution is computed by solving the normal
    equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
    \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the regularized
    least-squares problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}}
    ||A Z - B||_F^2 + \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is
    computed as \\(X = A^T (A A^T + \lambda I)^{-1} B\\),
    which (for \\(\lambda = 0\\)) is the minimum-norm solution to the
    under-determined linear system, i.e.
    \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\),
    subject to \\(A Z = B\\).
    Notice that the fast path is only numerically stable when \\(A\\) is
    numerically full rank and has a condition number
    \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\)
    or \\(\lambda\\) is sufficiently large.

    If `fast` is `False` then the solution is computed using the rank revealing
    QR decomposition with column pivoting. This will always compute a
    least-squares solution that minimizes the residual norm
    \\(||A X - B||_F^2 \\), even when \\(A\\) is rank deficient or
    ill-conditioned. Notice: The current version does not compute a minimum norm
    solution. If `fast` is `False` then `l2_regularizer` is ignored.

    Args:
        matrix: 2-D `Tensor` of shape `[M, N]`.
        rhs: 2-D `Tensor` of shape is `[M, K]`.
        l2_regularizer: 0-D  `double` `Tensor`. Ignored if `fast=False`.
                        fast: bool. Defaults to `True`.
        name: string, optional name of the operation.

    Returns:
        output: Matrix of shape `[N, K]` containing the matrix that solves
            `matrix * output = rhs` in the least-squares sense.
    """
    pass


def matrix_triangular_solve(matrix, rhs, lower=None, adjoint=None, name=None):
    """
    [TensorFlow Docs]
    Solves a system of linear equations with an upper or lower triangular matrix by

    backsubstitution.

    `matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
    upper triangular part of `matrix` is assumed to be zero and not accessed.
    If `lower` is False then the strictly lower triangular part of `matrix` is
    assumed to be zero and not accessed.
    `rhs` is a matrix of shape [M, K]`.

    The output is a matrix of shape `[M, K]`. If `adjoint` is `False` the output
    satisfies the matrix equation `matrix` * `output` = `rhs`.
    If `adjoint` is `False` then `output` satisfies the matrix equation
    `matrix` * `output` = `rhs`.
    If `adjoint` is `True` then `output` satisfies the matrix equation
    `adjoint(matrix)` * `output` = `rhs`.

    Args:
        matrix: A `Tensor`. Must be one of the following types: `float64`, `float32`.
                Shape is `[M, M]`.
        rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
        lower: An optional `bool`. Defaults to `True`.
               Boolean indicating whether `matrix` is lower or upper triangular
        adjoint: An optional `bool`. Defaults to `False`.
                 Boolean indicating whether to solve with `matrix` or its adjoint.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `matrix`. Shape is `[M, K]`.
    """
    pass


def maximum(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the max of x and y (i.e. x > y ? x : y) element-wise, broadcasts.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def merge_all_summaries(key=ops.GraphKeys.SUMMARIES):
    """
    [TensorFlow Docs]
    Merges all summaries collected in the default graph.

    Args:
        key: `GraphKey` used to collect the summaries. Defaults to
             `GraphKeys.SUMMARIES`.

    Returns:
        If no summaries were collected, returns None. Otherwise returns a scalar
        `Tensor` of type `string` containing the serialized `Summary` protocol
        buffer resulting from the merging.
    """
    pass


def merge_summary(inputs, collections=None, name=None):
    """
    [TensorFlow Docs]
    Merges summaries.

    This op creates a
    [`Summary`](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
    protocol buffer that contains the union of all the values in the input
    summaries.

    When the Op is run, it reports an `InvalidArgument` error if multiple values
    in the summaries to merge use the same tag.

    Args:
        inputs: A list of `string` `Tensor` objects containing serialized `Summary`
                protocol buffers.
        collections: Optional list of graph collections keys. The new summary op is
                     added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
        name: A name for the operation (optional).

    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol
        buffer resulting from the merging.
    """
    pass


def minimum(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the min of x and y (i.e. x < y ? x : y) element-wise, broadcasts.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def mod(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns element-wise remainder of division.

    Args:
        x: A `Tensor`. Must be one of the following types: `int32`, `int64`, `float32`, `float64`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def moving_average_variables():
    """
    [TensorFlow Docs]
    Returns all variables that maintain their moving averages.

    If an `ExponentialMovingAverage` object is created and the `apply()`
    method is called on a list of variables, these variables will
    be added to the `GraphKeys.MOVING_AVERAGE_VARIABLES` collection.
    This convenience function returns the contents of that collection.

    Returns:
        A list of Variable objects.
    """
    pass


def mul(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns x * y element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `complex128`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def multinomial(logits, num_samples, seed=None, name=None):
    """
    [TensorFlow Docs]
    Draws samples from a multinomial distribution.

    Example:

        samples = tf.multinomial(tf.log([[0.5, 0.5]]), 10)
        # samples has shape [1, 10], where each value is either 0 or 1.

        samples = tf.multinomial([[1, -1, -1]], 10)
        # samples is equivalent to tf.zeros([1, 10], dtype=tf.int64).

    Args:
        logits: 2-D Tensor with shape `[batch_size, num_classes]`. Each slice
                `[i, :]` represents the unnormalized log probabilities for all classes.
        num_samples: 0-D. Number of independent samples to draw for each row slice.
        seed: A Python integer. Used to create a random seed for the distribution.
              See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        name: Optional name for the operation.

    Returns:
        The drawn samples of shape `[batch_size, num_samples]`.
    """
    pass


def name_scope(name):
    """
    [TensorFlow Docs]
    Wrapper for `Graph.name_scope()` using the default graph.

    See
    [`Graph.name_scope()`](../../api_docs/python/framework.md#Graph.name_scope)
    for more details.

    Args:
        name: A name for the scope.

    Returns:
        A context manager that installs `name` as a new name scope in the
        default graph.
    """
    pass


def neg(x, name=None):
    """
    [TensorFlow Docs]
    Computes numerical negative value element-wise.

    I.e., \\(y = -x\\).

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def no_op(name=None):
    """
    [TensorFlow Docs]
    Does nothing. Only useful as a placeholder for control edges.

    Args:
        name: A name for the operation (optional).

    Returns:
        The created Operation.
    """
    pass


def no_regularizer(_):
    """
    [TensorFlow Docs]
    Use this function to prevent regularization of variables."""
    pass


def not_equal(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns the truth value of (x != y) element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`, `complex64`, `quint8`, `qint8`, `qint32`, `string`, `bool`, `complex128`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`.
    """
    pass


def one_hot(indices, depth, on_value=None, off_value=None,
            axis=None, dtype=None, name=None):
    """
    [TensorFlow Docs]
    Returns a one-hot tensor.

    The locations represented by indices in `indices` take value `on_value`,
    while all other locations take value `off_value`.

    `on_value` and `off_value` must have matching data types. If `dtype` is also
    provided, they must be the same data type as specified by `dtype`.

    If `on_value` is not provided, it will default to the value `1` with type
    `dtype`

    If `off_value` is not provided, it will default to the value `0` with type
    `dtype`

    If the input `indices` is rank `N`, the output will have rank `N+1`. The
    new axis is created at dimension `axis` (default: the new axis is appended
    at the end).

    If `indices` is a scalar the output shape will be a vector of length `depth`

    If `indices` is a vector of length `features`, the output shape will be:
    ```
        features x depth if axis == -1
        depth x features if axis == 0
    ```

    If `indices` is a matrix (batch) with shape `[batch, features]`, the output
    shape will be:
    ```
        batch x features x depth if axis == -1
        batch x depth x features if axis == 1
        depth x batch x features if axis == 0
    ```

    If `dtype` is not provided, it will attempt to assume the data type of
    `on_value` or `off_value`, if one or both are passed in. If none of
    `on_value`, `off_value`, or `dtype` are provided, `dtype` will default to the
    value `tf.float32`

    Note: If a non-numeric data type output is desired (tf.string, tf.bool, etc.),
    both `on_value` and `off_value` _must_ be provided to `one_hot`

    Examples
    =========

    Suppose that

    ```
        indices = [0, 2, -1, 1]
        depth = 3
        on_value = 5.0
        off_value = 0.0
        axis = -1
    ```

    Then output is `[4 x 3]`:

    ```
        output =
        [5.0 0.0 0.0]  // one_hot(0)
        [0.0 0.0 5.0]  // one_hot(2)
        [0.0 0.0 0.0]  // one_hot(-1)
        [0.0 5.0 0.0]  // one_hot(1)
    ```

    Suppose that

    ```
        indices = [[0, 2], [1, -1]]
        depth = 3
        on_value = 1.0
        off_value = 0.0
        axis = -1
    ```

    Then output is `[2 x 2 x 3]`:

    ```
        output =
        [
            [1.0, 0.0, 0.0]  // one_hot(0)
            [0.0, 0.0, 1.0]  // one_hot(2)
        ][
            [0.0, 1.0, 0.0]  // one_hot(1)
            [0.0, 0.0, 0.0]  // one_hot(-1)
        ]
    ```

    Using default values for `on_value` and `off_value`:

    ```
        indices = [0, 1, 2]
        depth = 3
    ```

    The output will be

    ```
        output =
        [[1., 0., 0.],
     [0., 1., 0.],
     [0., 0., 1.]]
    ```

    Args:
        indices: A `Tensor` of indices.
        depth: A scalar defining the depth of the one hot dimension.
        on_value: A scalar defining the value to fill in output when `indices[j]
                  = i`. (default: 1)
                  off_value: A scalar defining the value to fill in output when `indices[j]
                  != i`. (default: 0)
        axis: The axis to fill (default: -1, a new inner-most axis).
        dtype: The data type of the output tensor.

    Returns:
        output: The one-hot tensor.

    Raises:
        TypeError: If dtype of either `on_value` or `off_value` don't match `dtype`
        TypeError: If dtype of `on_value` and `off_value` don't match one another
    """
    pass


def ones(shape, dtype=dtypes.float32, name=None):
    """
    [TensorFlow Docs]
    Creates a tensor with all elements set to 1.

    This operation returns a tensor of type `dtype` with shape `shape` and all
    elements set to 1.

    For example:

    ```python
    tf.ones([2, 3], int32) ==> [[1, 1, 1], [1, 1, 1]]
    ```

    Args:
        shape: Either a list of integers, or a 1-D `Tensor` of type `int32`.
        dtype: The type of an element in the resulting `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with all elements set to 1.
    """
    pass


def ones_initializer(shape, dtype=dtypes.float32):
    """
    [TensorFlow Docs]
    An adaptor for ones() to match the Initializer spec."""
    pass


def ones_like(tensor, dtype=None, name=None):
    """
    [TensorFlow Docs]
    Creates a tensor with all elements set to 1.

    Given a single tensor (`tensor`), this operation returns a tensor of the same
    type and shape as `tensor` with all elements set to 1. Optionally, you can
    specify a new type (`dtype`) for the returned tensor.

    For example:

    ```python
    # 'tensor' is [[1, 2, 3], [4, 5, 6]]
    tf.ones_like(tensor) ==> [[1, 1, 1], [1, 1, 1]]
    ```

    Args:
        tensor: A `Tensor`.
        dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
               `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with all elements set to 1.
    """
    pass


def pack(values, name="pack"):
    """
    [TensorFlow Docs]
    Packs a list of rank-`R` tensors into one rank-`(R+1)` tensor.

    Packs tensors in `values` into a tensor with rank one higher than each tensor
    in `values` and shape `[len(values)] + values[0].shape`. The output satisfies
    `output[i, ...] = values[i][...]`.

    This is the opposite of unpack. The numpy equivalent is

            tf.pack([x, y, z]) = np.asarray([x, y, z])

    Args:
        values: A list of `Tensor` objects with the same shape and type.
        name: A name for this operation (optional).

    Returns:
        output: A packed `Tensor` with the same type as `values`.
    """
    pass


def pad(tensor, paddings, mode="CONSTANT",
        name=None):  # pylint: disable=invalid-name
    """Pads a tensor.

    This operation pads a `tensor` according to the `paddings` you specify.
    `paddings` is an integer tensor with shape `[n, 2]`, where n is the rank of
    `tensor`. For each dimension D of `input`, `paddings[D, 0]` indicates how
    many values to add before the contents of `tensor` in that dimension, and
    `paddings[D, 1]` indicates how many values to add after the contents of
    `tensor` in that dimension. If `mode` is "REFLECT" then both `paddings[D, 0]`
    and `paddings[D, 1]` must be no greater than `tensor.dim_size(D) - 1`. If
    `mode` is "SYMMETRIC" then both `paddings[D, 0]` and `paddings[D, 1]` must be
    no greater than `tensor.dim_size(D)`.

    The padded size of each dimension D of the output is:

    `paddings[D, 0] + tensor.dim_size(D) + paddings[D, 1]`

    For example:

    ```python
    # 't' is [[1, 2, 3], [4, 5, 6]].
    # 'paddings' is [[1, 1,], [2, 2]].
    # rank of 't' is 2.
    pad(t, paddings, "CONSTANT") ==> [[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 2, 3, 0, 0],
                                    [0, 0, 4, 5, 6, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]]

    pad(t, paddings, "REFLECT") ==> [[6, 5, 4, 5, 6, 5, 4],
                                   [3, 2, 1, 2, 3, 2, 1],
                                   [6, 5, 4, 5, 6, 5, 4],
                                   [3, 2, 1, 2, 3, 2, 1]]

    pad(t, paddings, "SYMMETRIC") ==> [[2, 1, 1, 2, 3, 3, 2],
                                     [2, 1, 1, 2, 3, 3, 2],
                                     [5, 4, 4, 5, 6, 6, 5],
                                     [5, 4, 4, 5, 6, 6, 5]]
    ```

    Args:
        tensor: A `Tensor`.
        paddings: A `Tensor` of type `int32`.
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC".
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `tensor`.

    Raises:
        ValueError: When mode is not one of "CONSTANT", "REFLECT", or "SYMMETRIC".
    """
    pass


def parse_example(serialized, features, name=None, example_names=None):
    # pylint: disable=line-too-long
    """Parses `Example` protos into a `dict` of tensors.

    Parses a number of serialized [`Example`]
    (https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
    protos given in `serialized`.

    `example_names` may contain descriptive names for the corresponding serialized
    protos. These may be useful for debugging purposes, but they have no effect on
    the output. If not `None`, `example_names` must be the same length as `serialized`.

    This op parses serialized examples into a dictionary mapping keys to `Tensor`
    and `SparseTensor` objects. `features` is a dict from keys to `VarLenFeature`
    and `FixedLenFeature` objects. Each `VarLenFeature` is mapped to a
    `SparseTensor`, and each `FixedLenFeature` is mapped to a `Tensor`.

    Each `VarLenFeature` maps to a `SparseTensor` of the specified type
    representing a ragged matrix. Its indices are `[batch, index]` where `batch`
    is the batch entry the value is from in `serialized`, and `index` is the
    value's index in the list of values associated with that feature and example.

    Each `FixedLenFeature` `df` maps to a `Tensor` of the specified type (or
    `tf.float32` if not specified) and shape `(serialized.size(),) + df.shape`.

    `FixedLenFeature` entries with a `default_value` are optional. With no default
    value, we will fail if that `Feature` is missing from any example in
    `serialized`.

    Examples:

    For example, if one expects a `tf.float32` sparse feature `ft` and three
    serialized `Example`s are provided:

    ```
    serialized = [
        features
            { feature { key: "ft" value { float_list { value: [1.0, 2.0] } } } },
        features
            { feature []},
        features
            { feature { key: "ft" value { float_list { value: [3.0] } } }
    ]
    ```

    then the output will look like:

    ```
    {"ft": SparseTensor(indices=[[0, 0], [0, 1], [2, 0]],
                      values=[1.0, 2.0, 3.0],
                      shape=(3, 2)) }
    ```

    Given two `Example` input protos in `serialized`:

    ```
    [
        features {
            feature { key: "kw" value { bytes_list { value: [ "knit", "big" ] } } }
            feature { key: "gps" value { float_list { value: [] } } }
        },
        features {
            feature { key: "kw" value { bytes_list { value: [ "emmy" ] } } }
            feature { key: "dank" value { int64_list { value: [ 42 ] } } }
            feature { key: "gps" value { } }
        }
    ]
    ```

    And arguments

    ```
    example_names: ["input0", "input1"],
    features: {
              "kw": VarLenFeature(tf.string),
              "dank": VarLenFeature(tf.int64),
              "gps": VarLenFeature(tf.float32),
              }
              ```

    Then the output is a dictionary:

    ```python
    {
        "kw": SparseTensor(
                indices=[[0, 0], [0, 1], [1, 0]],
                values=["knit", "big", "emmy"]
                shape=[2, 2]),
        "dank": SparseTensor(
                indices=[[1, 0]],
                values=[42],
                shape=[2, 1]),
        "gps": SparseTensor(
                indices=[],
                values=[],
                shape=[2, 0]),
    }
    ```

    For dense results in two serialized `Example`s:

    ```
    [
        features {
            feature { key: "age" value { int64_list { value: [ 0 ] } } }
            feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
     },
     features {
            feature { key: "age" value { int64_list { value: [] } } }
            feature { key: "gender" value { bytes_list { value: [ "f" ] } } }
        }
    ]
    ```

    We can use arguments:

    ```
    example_names: ["input0", "input1"],
    features: {
              "age": FixedLenFeature([], dtype=tf.int64, default_value=-1),
              "gender": FixedLenFeature([], dtype=tf.string),
              }
              ```

    And the expected output is:

    ```python
    {
        "age": [[0], [-1]],
        "gender": [["f"], ["f"]],
    }
    ```

    Args:
        serialized: A vector (1-D Tensor) of strings, a batch of binary
                    serialized `Example` protos.
        features: A `dict` mapping feature keys to `FixedLenFeature` or
                  `VarLenFeature` values.
        name: A name for this operation (optional).
              example_names: A vector (1-D Tensor) of strings (optional), the names of
              the serialized protos in the batch.

    Returns:
        A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

    Raises:
        ValueError: if any feature is invalid.
    """
    pass


def parse_single_example(serialized, features, name=None, example_names=None):
    """
    [TensorFlow Docs]
    Parses a single `Example` proto.

    Similar to `parse_example`, except:

    For dense tensors, the returned `Tensor` is identical to the output of
    `parse_example`, except there is no batch dimension, the output shape is the
    same as the shape given in `dense_shape`.

    For `SparseTensor`s, the first (batch) column of the indices matrix is removed
    (the indices matrix is a column vector), the values vector is unchanged, and
    the first (`batch_size`) entry of the shape vector is removed (it is now a
    single element vector).

    Args:
        serialized: A scalar string Tensor, a single serialized Example.
                    See `_parse_single_example_raw` documentation for more details.
        features: A `dict` mapping feature keys to `FixedLenFeature` or
                  `VarLenFeature` values.
        name: A name for this operation (optional).
              example_names: (Optional) A scalar string Tensor, the associated name.
              See `_parse_single_example_raw` documentation for more details.

    Returns:
        A `dict` mapping feature keys to `Tensor` and `SparseTensor` values.

    Raises:
        ValueError: if any feature is invalid.
    """
    pass


def parse_single_sequence_example(
        serialized, context_features=None, sequence_features=None,
        example_name=None, name=None):
    # pylint: disable=line-too-long
    """Parses a single `SequenceExample` proto.

    Parses a single serialized [`SequenceExample`]
    (https://www.tensorflow.org/code/tensorflow/core/example/example.proto)
    proto given in `serialized`.

    This op parses a serialize sequence example into a tuple of dictionaries
    mapping keys to `Tensor` and `SparseTensor` objects respectively.
    The first dictionary contains mappings for keys appearing in
    `context_features`, and the second dictionary contains mappings for keys
    appearing in `sequence_features`.

    At least one of `context_features` and `sequence_features` must be provided
    and non-empty.

    The `context_features` keys are associated with a `SequenceExample` as a
    whole, independent of time / frame. In contrast, the `sequence_features` keys
    provide a way to access variable-length data within the `FeatureList` section
    of the `SequenceExample` proto. While the shapes of `context_features` values
    are fixed with respect to frame, the frame dimension (the first dimension)
    of `sequence_features` values may vary between `SequenceExample` protos,
    and even between `feature_list` keys within the same `SequenceExample`.

    `context_features` contains `VarLenFeature` and `FixedLenFeature` objects.
    Each `VarLenFeature` is mapped to a `SparseTensor`, and each `FixedLenFeature`
    is mapped to a `Tensor`, of the specified type, shape, and default value.

    `sequence_features` contains `VarLenFeature` and `FixedLenSequenceFeature`
    objects. Each `VarLenFeature` is mapped to a `SparseTensor`, and each
    `FixedLenSequenceFeature` is mapped to a `Tensor`, each of the specified type.
    The shape will be `(T,) + df.shape` for `FixedLenSequenceFeature` `df`, where
    `T` is the length of the associated `FeatureList` in the `SequenceExample`.

    Each `SparseTensor` corresponding to `sequence_features` represents a ragged
    vector. Its indices are `[time, index]`, where `time` is the `FeatureList`
    entry and `index` is the value's index in the list of values associated with
    that time.

    `FixedLenFeature` entries with a `default_value` and `FixedLenSequenceFeature`
    entries with `allow_missing=True` are optional; otherwise, we will fail if
    that `Feature` or `FeatureList` is missing from any example in `serialized`.

    `example_name` may contain a descriptive name for the corresponding serialized
    proto. This may be useful for debugging purposes, but it has no effect on the
    output. If not `None`, `example_name` must be a scalar.

    Args:
        serialized: A scalar (0-D Tensor) of type string, a single binary
                    serialized `SequenceExample` proto.
        context_features: A `dict` mapping feature keys to `FixedLenFeature` or
                          `VarLenFeature` values. These features are associated with a
                          `SequenceExample` as a whole.
                          sequence_features: A `dict` mapping feature keys to
                          `FixedLenSequenceFeature` or `VarLenFeature` values. These features are
                          associated with data within the `FeatureList` section of the
                          `SequenceExample` proto.
                          example_name: A scalar (0-D Tensor) of strings (optional), the name of
                          the serialized proto.
        name: A name for this operation (optional).

    Returns:
        A tuple of two `dict`s, each mapping keys to `Tensor`s and `SparseTensor`s.
        The first dict contains the context key/values.
        The second dict contains the feature_list key/values.

    Raises:
        ValueError: if any feature is invalid.
    """
    pass


def placeholder(dtype, shape=None, name=None):
    """
    [TensorFlow Docs]
    Inserts a placeholder for a tensor that will be always fed.

    **Important**: This tensor will produce an error if evaluated. Its value must
    be fed using the `feed_dict` optional argument to `Session.run()`,
    `Tensor.eval()`, or `Operation.run()`.

    For example:

    ```python
    x = tf.placeholder(tf.float32, shape=(1024, 1024))
    y = tf.matmul(x, x)

    with tf.Session() as sess:
        print(sess.run(y))  # ERROR: will fail because x was not fed.

        rand_array = np.random.rand(1024, 1024)
        print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
    ```

    Args:
        dtype: The type of elements in the tensor to be fed.
        shape: The shape of the tensor to be fed (optional). If the shape is not
               specified, you can feed a tensor of any shape.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` that may be used as a handle for feeding a value, but not
        evaluated directly.
    """
    pass


def placeholder_with_default(input, shape, name=None):
    """
    [TensorFlow Docs]
    A placeholder op that passes though `input` when its output is not fed.

    Args:
        input: A `Tensor`. The default value to produce when `output` is not fed.
        shape: A `tf.TensorShape` or list of `ints`.
               The (possibly partial) shape of the tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        A placeholder tensor that defaults to `input` if it is not fed.
    """
    pass


def polygamma(a, x, name=None):
    """
    [TensorFlow Docs]
    Compute the polygamma function \\(\psi^{(n)}(x)\\).

    The polygamma function is defined as:

    ```
    \psi^{(n)}(x) = \frac{d^n}{dx^n} \psi(x)
    ```
    where \\(\psi(x)\\) is the digamma function.

    Args:
        a: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        x: A `Tensor`. Must have the same type as `a`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `a`.
    """
    pass


def pow(x, y, name=None):
    """
    [TensorFlow Docs]
    Computes the power of one value to another.

    Given a tensor `x` and a tensor `y`, this operation computes \\\\(x^y\\\\) for
    corresponding elements in `x` and `y`. For example:

    ```
    # tensor 'x' is [[2, 2], [3, 3]]
    # tensor 'y' is [[8, 16], [2, 3]]
    tf.pow(x, y) ==> [[256, 65536], [9, 27]]
    ```

    Args:
        x: A `Tensor` of type `float`, `double`, `int32`, `int64`, `complex64`, or
           `complex128`.
        y: A `Tensor` of type `float`, `double`, `int32`, `int64`, `complex64`, or
           `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`.
    """
    pass


def py_func(func, inp, Tout, name=None):
    """
    [TensorFlow Docs]
    Wraps a python function and uses it as a tensorflow op.

    Given a python function `func`, which takes numpy arrays as its
    inputs and returns numpy arrays as its outputs. E.g.,

    ```python
    def my_func(x):
        # x will be a numpy array with the contents of the placeholder below
        return np.sinh(x)
    inp = tf.placeholder(tf.float32, [...])
    y = py_func(my_func, [inp], [tf.float32])
    ```

    The above snippet constructs a tf graph which invokes a numpy
    sinh(x) as an op in the graph.

    Args:
        func: A python function.
        inp: A list of `Tensor`.
        Tout: A list of tensorflow data types indicating what `func`
              returns.
        name: A name for the operation (optional).

    Returns:
        A list of `Tensor` which `func` computes.
    """
    pass


def random_crop(value, size, seed=None, name=None):
    """
    [TensorFlow Docs]
    Randomly crops a tensor to a given size.

    Slices a shape `size` portion out of `value` at a uniformly chosen offset.
    Requires `value.shape >= size`.

    If a dimension should not be cropped, pass the full size of that dimension.
    For example, RGB images can be cropped with
    `size = [crop_height, crop_width, 3]`.

    Args:
        value: Input tensor to crop.
        size: 1-D tensor with size the rank of `value`.
        seed: Python integer. Used to create a random seed. See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        name: A name for this operation (optional).

    Returns:
        A cropped tensor of the same rank as `value` and shape `size`.
    """
    pass


def random_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
                  seed=None, name=None):
    """
    [TensorFlow Docs]
    Outputs random values from a normal distribution.

    Args:
        shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
        mean: A 0-D Tensor or Python value of type `dtype`. The mean of the normal
              distribution.
              stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
              of the normal distribution.
        dtype: The type of the output.
        seed: A Python integer. Used to create a random seed for the distribution.
              See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape filled with random normal values.
    """
    pass


def random_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                              dtype=dtypes.float32):
    """
    [TensorFlow Docs]
    Returns an initializer that generates tensors with a normal distribution.

    Args:
        mean: a python scalar or a scalar tensor. Mean of the random values
              to generate.
              stddev: a python scalar or a scalar tensor. Standard deviation of the
              random values to generate.
        seed: A Python integer. Used to create random seeds. See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer that generates tensors with a normal distribution.

    Raises:
        ValueError: if `dtype` is not a floating point type.
    """
    pass


def random_shuffle(value, seed=None, name=None):
    """
    [TensorFlow Docs]
    Randomly shuffles a tensor along its first dimension.

    The tensor is shuffled along dimension 0, such that each `value[j]` is mapped
    to one and only one `output[i]`. For example, a mapping that might occur for a
    3x2 tensor is:

    ```python
    [[1, 2],     [[5, 6],
   [3, 4], ==>   [1, 2],
   [5, 6]]        [3, 4]]
    ```

    Args:
        value: A Tensor to be shuffled.
        seed: A Python integer. Used to create a random seed for the distribution.
              See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        name: A name for the operation (optional).

    Returns:
        A tensor of same shape and type as `value`, shuffled along its first
        dimension.
    """
    pass


def random_uniform(shape, minval=0, maxval=None,
                   dtype=dtypes.float32, seed=None,
                   name=None):
    """
    [TensorFlow Docs]
    Outputs random values from a uniform distribution.

    The generated values follow a uniform distribution in the range
    `[minval, maxval)`. The lower bound `minval` is included in the range, while
    the upper bound `maxval` is excluded.

    For floats, the default range is `[0, 1)`. For ints, at least `maxval` must
    be specified explicitly.

    In the integer case, the random integers are slightly biased unless
    `maxval - minval` is an exact power of two. The bias is small for values of
    `maxval - minval` significantly smaller than the range of the output (either
    `2**32` or `2**64`).

    Args:
        shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
        minval: A 0-D Tensor or Python value of type `dtype`. The lower bound on the
                range of random values to generate. Defaults to 0.
                maxval: A 0-D Tensor or Python value of type `dtype`. The upper bound on
                the range of random values to generate. Defaults to 1 if `dtype` is
                floating point.
        dtype: The type of the output: `float32`, `float64`, `int32`, or `int64`.
        seed: A Python integer. Used to create a random seed for the distribution.
              See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape filled with random uniform values.

    Raises:
        ValueError: If `dtype` is integral and `maxval` is not specified.
    """
    pass


def random_uniform_initializer(minval=0.0, maxval=1.0, seed=None,
                               dtype=dtypes.float32):
    """
    [TensorFlow Docs]
    Returns an initializer that generates tensors with a uniform distribution.

    Args:
        minval: a python scalar or a scalar tensor. lower bound of the range
                of random values to generate.
                maxval: a python scalar or a scalar tensor. upper bound of the range
                of random values to generate.
        seed: A Python integer. Used to create random seeds. See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer that generates tensors with a uniform distribution.

    Raises:
        ValueError: if `dtype` is not a floating point type.
    """
    pass


def range(start, limit=None, delta=1, name="range"):
    """
    [TensorFlow Docs]
    Creates a sequence of integers.

    Creates a sequence of integers that begins at `start` and extends by
    increments of `delta` up to but not including `limit`.

    Like the Python builtin `range`, `start` defaults to 0, so that
    `range(n) = range(0, n)`.

    For example:

    ```
    # 'start' is 3
    # 'limit' is 18
    # 'delta' is 3
    tf.range(start, limit, delta) ==> [3, 6, 9, 12, 15]

    # 'limit' is 5
    tf.range(limit) ==> [0, 1, 2, 3, 4]
    ```

    Args:
        start: A 0-D (scalar) of type `int32`. First entry in sequence.
               Defaults to 0.
        limit: A 0-D (scalar) of type `int32`. Upper limit of sequence,
               exclusive.
               delta: A 0-D `Tensor` (scalar) of type `int32`. Optional. Default is 1.
               Number that increments `start`.
        name: A name for the operation (optional).

    Returns:
        An 1-D `int32` `Tensor`.
    """
    pass


def rank(input, name=None):
    """
    [TensorFlow Docs]
    Returns the rank of a tensor.

    This operation returns an integer representing the rank of `input`.

    For example:

    ```python
    # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    # shape of tensor 't' is [2, 2, 3]
    rank(t) ==> 3
    ```

    **Note**: The rank of a tensor is not the same as the rank of a matrix. The
    rank of a tensor is the number of indices required to uniquely select each
    element of the tensor. Rank is also known as "order", "degree", or "ndims."

    Args:
        input: A `Tensor` or `SparseTensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int32`.
    """
    pass


def read_file(filename, name=None):
    """
    [TensorFlow Docs]
    Reads and outputs the entire contents of the input filename.

    Args:
        filename: A `Tensor` of type `string`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `string`.
    """
    pass


def real(input, name=None):
    """
    [TensorFlow Docs]
    Returns the real part of a complex number.

    Given a tensor `input` of complex numbers, this operation returns a tensor of
    type `float` or `double` that is the real part of each element in `input`.
    All elements in `input` must be complex numbers of the form \\(a + bj\\),
    where *a* is the real part returned by this operation and *b* is the
    imaginary part.

    For example:

    ```
    # tensor 'input' is [-2.25 + 4.75j, 3.25 + 5.75j]
    tf.real(input) ==> [-2.25, 3.25]
    ```

    Args:
        input: A `Tensor`. Must be one of the following types: `complex64`,
               `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `float` or `double`.
    """
    pass


def reduce_all(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
    """
    [TensorFlow Docs]
    Computes the "logical and" of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.

    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    For example:

    ```python
    # 'x' is [[True, True]
    #         [False, False]]
    tf.reduce_all(x) ==> False
    tf.reduce_all(x, 0) ==> [False, False]
    tf.reduce_all(x, 1) ==> [True, False]
    ```

    Args:
        input_tensor: The boolean tensor to reduce.
        reduction_indices: The dimensions to reduce. If `None` (the default),
                           reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).

    Returns:
        The reduced tensor.
    """
    pass


def reduce_any(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
    """
    [TensorFlow Docs]
    Computes the "logical or" of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.

    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    For example:

    ```python
    # 'x' is [[True, True]
    #         [False, False]]
    tf.reduce_any(x) ==> True
    tf.reduce_any(x, 0) ==> [True, True]
    tf.reduce_any(x, 1) ==> [True, False]
    ```

    Args:
        input_tensor: The boolean tensor to reduce.
        reduction_indices: The dimensions to reduce. If `None` (the default),
                           reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).

    Returns:
        The reduced tensor.
    """
    pass


def reduce_join(inputs, reduction_indices, keep_dims=None, separator=None,
                name=None):
    """
    [TensorFlow Docs]
    Joins a string Tensor across the given dimensions.

    Computes the string join across dimensions in the given string Tensor of shape
    `[d_0, d_1, ..., d_n-1]`. Returns a new Tensor created by joining the input
    strings with the given separator (default: empty string). Negative indices are
    counted backwards from the end, with `-1` being equivalent to `n - 1`. Passing
    an empty `reduction_indices` joins all strings in linear index order and outputs
    a scalar string.


    For example:
    ```
    # tensor `a` is [["a", "b"], ["c", "d"]]
    tf.reduce_join(a, 0) ==> ["ac", "bd"]
    tf.reduce_join(a, 1) ==> ["ab", "cd"]
    tf.reduce_join(a, -2) = tf.reduce_join(a, 0) ==> ["ac", "bd"]
    tf.reduce_join(a, -1) = tf.reduce_join(a, 1) ==> ["ab", "cd"]
    tf.reduce_join(a, 0, keep_dims=True) ==> [["ac", "bd"]]
    tf.reduce_join(a, 1, keep_dims=True) ==> [["ab"], ["cd"]]
    tf.reduce_join(a, 0, separator=".") ==> ["a.c", "b.d"]
    tf.reduce_join(a, [0, 1]) ==> ["acbd"]
    tf.reduce_join(a, [1, 0]) ==> ["abcd"]
    tf.reduce_join(a, []) ==> ["abcd"]
    ```

    Args:
        inputs: A `Tensor` of type `string`.
                The input to be joined. All reduced indices must have non-zero size.
        reduction_indices: A `Tensor` of type `int32`.
                           The dimensions to reduce over. Dimensions are reduced in the
                           order specified. If `reduction_indices` has higher rank than `1`, it is
                           flattened. Omitting `reduction_indices` is equivalent to passing
                           `[n-1, n-2, ..., 0]`. Negative indices from `-n` to `-1` are supported.
        keep_dims: An optional `bool`. Defaults to `False`.
                   If `True`, retain reduced dimensions with length `1`.
                   separator: An optional `string`. Defaults to `""`.
                   The separator to use when joining.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `string`.
        Has shape equal to that of the input with reduced dimensions removed or
        set to `1` depending on `keep_dims`.
    """
    pass


def reduce_max(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
    """
    [TensorFlow Docs]
    Computes the maximum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.

    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
        input_tensor: The tensor to reduce. Should have numeric type.
        reduction_indices: The dimensions to reduce. If `None` (the default),
                           reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).

    Returns:
        The reduced tensor.
    """
    pass


def reduce_mean(input_tensor, reduction_indices=None, keep_dims=False,
                name=None):
    """
    [TensorFlow Docs]
    Computes the mean of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.

    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    For example:

    ```python
    # 'x' is [[1., 1.]
    #         [2., 2.]]
    tf.reduce_mean(x) ==> 1.5
    tf.reduce_mean(x, 0) ==> [1.5, 1.5]
    tf.reduce_mean(x, 1) ==> [1., 2.]
    ```

    Args:
        input_tensor: The tensor to reduce. Should have numeric type.
        reduction_indices: The dimensions to reduce. If `None` (the default),
                           reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).

    Returns:
        The reduced tensor.
    """
    pass


def reduce_min(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
    """
    [TensorFlow Docs]
    Computes the minimum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.

    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
        input_tensor: The tensor to reduce. Should have numeric type.
        reduction_indices: The dimensions to reduce. If `None` (the default),
                           reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).

    Returns:
        The reduced tensor.
    """
    pass


def reduce_prod(input_tensor, reduction_indices=None, keep_dims=False,
                name=None):
    """
    [TensorFlow Docs]
    Computes the product of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.

    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    Args:
        input_tensor: The tensor to reduce. Should have numeric type.
        reduction_indices: The dimensions to reduce. If `None` (the default),
                           reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).

    Returns:
        The reduced tensor.
    """
    pass


def reduce_sum(input_tensor, reduction_indices=None, keep_dims=False,
               name=None):
    """
    [TensorFlow Docs]
    Computes the sum of elements across dimensions of a tensor.

    Reduces `input_tensor` along the dimensions given in `reduction_indices`.
    Unless `keep_dims` is true, the rank of the tensor is reduced by 1 for each
    entry in `reduction_indices`. If `keep_dims` is true, the reduced dimensions
    are retained with length 1.

    If `reduction_indices` has no entries, all dimensions are reduced, and a
    tensor with a single element is returned.

    For example:

    ```python
    # 'x' is [[1, 1, 1]
    #         [1, 1, 1]]
    tf.reduce_sum(x) ==> 6
    tf.reduce_sum(x, 0) ==> [2, 2, 2]
    tf.reduce_sum(x, 1) ==> [3, 3]
    tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
    tf.reduce_sum(x, [0, 1]) ==> 6
    ```

    Args:
        input_tensor: The tensor to reduce. Should have numeric type.
        reduction_indices: The dimensions to reduce. If `None` (the default),
                           reduces all dimensions.
        keep_dims: If true, retains reduced dimensions with length 1.
        name: A name for the operation (optional).

    Returns:
        The reduced tensor.
    """
    pass


def register_tensor_conversion_function(base_type, conversion_func,
                                        priority=100):
    """
    [TensorFlow Docs]
    Registers a function for converting objects of `base_type` to `Tensor`.

    The conversion function must have the following signature:

            def conversion_func(value, dtype=None, name=None, as_ref=False):
                # ...

    It must return a `Tensor` with the given `dtype` if specified. If the
    conversion function creates a new `Tensor`, it should use the given
    `name` if specified. All exceptions will be propagated to the caller.

    The conversion function may return `NotImplemented` for some
    inputs. In this case, the conversion process will continue to try
    subsequent conversion functions.

    If `as_ref` is true, the function must return a `Tensor` reference,
    such as a `Variable`.

    NOTE: The conversion functions will execute in order of priority,
    followed by order of registration. To ensure that a conversion function
    `F` runs before another conversion function `G`, ensure that `F` is
    registered with a smaller priority than `G`.

    Args:
        base_type: The base type or tuple of base types for all objects that
                   `conversion_func` accepts.
        conversion_func: A function that converts instances of `base_type` to
                         `Tensor`.
        priority: Optional integer that indicates the priority for applying this
                  conversion function. Conversion functions with smaller priority values
                  run earlier than conversion functions with larger priority values.
                  Defaults to 100.

    Raises:
        TypeError: If the arguments do not have the appropriate type.

    """
    pass


def report_uninitialized_variables(var_list=None,
                                   name="report_uninitialized_variables"):
    """
    [TensorFlow Docs]
    Adds ops to list the names of uninitialized variables.

    When run, it returns a 1-D tensor containing the names of uninitialized
    variables if there are any, or an empty array if there are none.

    Args:
        var_list: List of `Variable` objects to check. Defaults to the
                  value of `all_variables() + local_variables()`
        name: Optional name of the `Operation`.

    Returns:
        A 1-D tensor containing names of the unintialized variables, or an empty 1-D
        tensor if there are no variables or no uninitialized variables.
    """
    pass


def reset_default_graph():
    """
    [TensorFlow Docs]
    Clears the default graph stack and resets the global default graph.

    NOTE: The default graph is a property of the current thread. This
    function applies only to the current thread. Calling this function while
    a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
    behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
    after calling this function will result in undefined behavior.
    """
    pass


def reshape(tensor, shape, name=None):
    """
    [TensorFlow Docs]
    Reshapes a tensor.

    Given `tensor`, this operation returns a tensor that has the same values
    as `tensor` with shape `shape`.

    If one component of `shape` is the special value -1, the size of that dimension
    is computed so that the total size remains constant. In particular, a `shape`
    of `[-1]` flattens into 1-D. At most one component of `shape` can be -1.

    If `shape` is 1-D or higher, then the operation returns a tensor with shape
    `shape` filled with the values of `tensor`. In this case, the number of elements
    implied by `shape` must be the same as the number of elements in `tensor`.

    For example:

    ```prettyprint
    # tensor 't' is [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # tensor 't' has shape [9]
    reshape(t, [3, 3]) ==> [[1, 2, 3],
                          [4, 5, 6],
                          [7, 8, 9]]

    # tensor 't' is [[[1, 1], [2, 2]],
    #                [[3, 3], [4, 4]]]
    # tensor 't' has shape [2, 2, 2]
    reshape(t, [2, 4]) ==> [[1, 1, 2, 2],
                          [3, 3, 4, 4]]

    # tensor 't' is [[[1, 1, 1],
    #                 [2, 2, 2]],
    #                [[3, 3, 3],
    #                 [4, 4, 4]],
    #                [[5, 5, 5],
    #                 [6, 6, 6]]]
    # tensor 't' has shape [3, 2, 3]
    # pass '[-1]' to flatten 't'
    reshape(t, [-1]) ==> [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]

    # -1 can also be used to infer the shape

    # -1 is inferred to be 9:
    reshape(t, [2, -1]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    # -1 is inferred to be 2:
    reshape(t, [-1, 9]) ==> [[1, 1, 1, 2, 2, 2, 3, 3, 3],
                           [4, 4, 4, 5, 5, 5, 6, 6, 6]]
    # -1 is inferred to be 3:
    reshape(t, [ 2, -1, 3]) ==> [[[1, 1, 1],
                                [2, 2, 2],
                                [3, 3, 3]],
                               [[4, 4, 4],
                                [5, 5, 5],
                                [6, 6, 6]]]

    # tensor 't' is [7]
    # shape `[]` reshapes to a scalar
    reshape(t, []) ==> 7
    ```

    Args:
        tensor: A `Tensor`.
        shape: A `Tensor` of type `int32`. Defines the shape of the output tensor.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `tensor`.
    """
    pass


def reverse(tensor, dims, name=None):
    """
    [TensorFlow Docs]
    Reverses specific dimensions of a tensor.

    Given a `tensor`, and a `bool` tensor `dims` representing the dimensions
    of `tensor`, this operation reverses each dimension i of `tensor` where
    `dims[i]` is `True`.

    `tensor` can have up to 8 dimensions. The number of dimensions
    of `tensor` must equal the number of elements in `dims`. In other words:

    `rank(tensor) = size(dims)`

    For example:

    ```prettyprint
    # tensor 't' is [[[[ 0, 1, 2, 3],
    #                  [ 4, 5, 6, 7],
    #                  [ 8, 9, 10, 11]],
    #                 [[12, 13, 14, 15],
    #                  [16, 17, 18, 19],
    #                  [20, 21, 22, 23]]]]
    # tensor 't' shape is [1, 2, 3, 4]

    # 'dims' is [False, False, False, True]
    reverse(t, dims) ==> [[[[ 3, 2, 1, 0],
                          [ 7, 6, 5, 4],
                          [ 11, 10, 9, 8]],
                         [[15, 14, 13, 12],
                          [19, 18, 17, 16],
                          [23, 22, 21, 20]]]]

    # 'dims' is [False, True, False, False]
    reverse(t, dims) ==> [[[[12, 13, 14, 15],
                          [16, 17, 18, 19],
                          [20, 21, 22, 23]
                         [[ 0, 1, 2, 3],
                          [ 4, 5, 6, 7],
                          [ 8, 9, 10, 11]]]]

    # 'dims' is [False, False, True, False]
    reverse(t, dims) ==> [[[[8, 9, 10, 11],
                          [4, 5, 6, 7],
                          [0, 1, 2, 3]]
                         [[20, 21, 22, 23],
                          [16, 17, 18, 19],
                          [12, 13, 14, 15]]]]
    ```

    Args:
        tensor: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `bool`, `half`, `float32`, `float64`.
                Up to 8-D.
        dims: A `Tensor` of type `bool`. 1-D. The dimensions to reverse.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `tensor`. The same shape as `tensor`.
    """
    pass


def reverse_sequence(input, seq_lengths, seq_dim, batch_dim=None, name=None):
    """
    [TensorFlow Docs]
    Reverses variable length slices.

    This op first slices `input` along the dimension `batch_dim`, and for each
    slice `i`, reverses the first `seq_lengths[i]` elements along
    the dimension `seq_dim`.

    The elements of `seq_lengths` must obey `seq_lengths[i] < input.dims[seq_dim]`,
    and `seq_lengths` must be a vector of length `input.dims[batch_dim]`.

    The output slice `i` along dimension `batch_dim` is then given by input
    slice `i`, with the first `seq_lengths[i]` slices along dimension
    `seq_dim` reversed.

    For example:

    ```prettyprint
    # Given this:
    batch_dim = 0
    seq_dim = 1
    input.dims = (4, 8, ...)
    seq_lengths = [7, 2, 3, 5]

    # then slices of input are reversed on seq_dim, but only up to seq_lengths:
    output[0, 0:7, :, ...] = input[0, 7:0:-1, :, ...]
    output[1, 0:2, :, ...] = input[1, 2:0:-1, :, ...]
    output[2, 0:3, :, ...] = input[2, 3:0:-1, :, ...]
    output[3, 0:5, :, ...] = input[3, 5:0:-1, :, ...]

    # while entries past seq_lens are copied through:
    output[0, 7:, :, ...] = input[0, 7:, :, ...]
    output[1, 2:, :, ...] = input[1, 2:, :, ...]
    output[2, 3:, :, ...] = input[2, 3:, :, ...]
    output[3, 2:, :, ...] = input[3, 2:, :, ...]
    ```

    In contrast, if:

    ```prettyprint
    # Given this:
    batch_dim = 2
    seq_dim = 0
    input.dims = (8, ?, 4, ...)
    seq_lengths = [7, 2, 3, 5]

    # then slices of input are reversed on seq_dim, but only up to seq_lengths:
    output[0:7, :, 0, :, ...] = input[7:0:-1, :, 0, :, ...]
    output[0:2, :, 1, :, ...] = input[2:0:-1, :, 1, :, ...]
    output[0:3, :, 2, :, ...] = input[3:0:-1, :, 2, :, ...]
    output[0:5, :, 3, :, ...] = input[5:0:-1, :, 3, :, ...]

    # while entries past seq_lens are copied through:
    output[7:, :, 0, :, ...] = input[7:, :, 0, :, ...]
    output[2:, :, 1, :, ...] = input[2:, :, 1, :, ...]
    output[3:, :, 2, :, ...] = input[3:, :, 2, :, ...]
    output[2:, :, 3, :, ...] = input[2:, :, 3, :, ...]
    ```

    Args:
        input: A `Tensor`. The input to reverse.
        seq_lengths: A `Tensor` of type `int64`.
                     1-D with length `input.dims(batch_dim)` and
                     `max(seq_lengths) < input.dims(seq_dim)`
        seq_dim: An `int`. The dimension which is partially reversed.
        batch_dim: An optional `int`. Defaults to `0`.
                   The dimension along which reversal is performed.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        The partially reversed input. It has the same shape as `input`.
    """
    pass


def round(x, name=None):
    """
    [TensorFlow Docs]
    Rounds the values of a tensor to the nearest integer, element-wise.

    For example:

    ```python
    # 'a' is [0.9, 2.5, 2.3, -4.4]
    tf.round(a) ==> [ 1.0, 3.0, 2.0, -4.0 ]
    ```

    Args:
        x: A `Tensor` of type `float` or `double`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of same shape and type as `x`.
    """
    pass


def rsqrt(x, name=None):
    """
    [TensorFlow Docs]
    Computes reciprocal of square root of x element-wise.

    I.e., \\(y = 1 / \sqrt{x}\\).

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def saturate_cast(value, dtype, name=None):
    """
    [TensorFlow Docs]
    Performs a safe saturating cast of `value` to `dtype`.

    This function casts the input to `dtype` without applying any scaling. If
    there is a danger that values would over or underflow in the cast, this op
    applies the appropriate clamping before the cast.

    Args:
        value: A `Tensor`.
        dtype: The desired output `DType`.
        name: A name for the operation (optional).

    Returns:
        `value` safely cast to `dtype`.
    """
    pass


def scalar_mul(scalar, x):
    """
    [TensorFlow Docs]
    Multiplies a scalar times a `Tensor` or `IndexedSlices` object.

    Intended for use in gradient code which might deal with `IndexedSlices`
    objects, which are easy to multiply by a scalar but more expensive to
    multiply with arbitrary tensors.

    Args:
        scalar: A 0-D scalar `Tensor`. Must have known shape.
        x: A `Tensor` or `IndexedSlices` to be scaled.

    Returns:
        `scalar * x` of the same type (`Tensor` or `IndexedSlices`) as `x`.

    Raises:
        ValueError: if scalar is not a 0-D `scalar`.
    """
    pass


def scalar_summary(tags, values, collections=None, name=None):
    """
    [TensorFlow Docs]
    Outputs a `Summary` protocol buffer with scalar values.

    The input `tags` and `values` must have the same shape. The generated
    summary has a summary value for each tag-value pair in `tags` and `values`.

    Args:
        tags: A `string` `Tensor`. Tags for the summaries.
        values: A real numeric Tensor. Values for the summaries.
        collections: Optional list of graph collections keys. The new summary op is
                     added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
        name: A name for the operation (optional).

    Returns:
        A scalar `Tensor` of type `string`. The serialized `Summary` protocol
        buffer.
    """
    pass


def scan(fn, elems, initializer=None, parallel_iterations=10, back_prop=True,
         swap_memory=False, name=None):
    """
    [TensorFlow Docs]
    scan on the list of tensors unpacked from `elems` on dimension 0.

    This scan operator repeatedly applies the callable `fn` to a sequence
    of elements from first to last. The elements are made of the tensors
    unpacked from `elems` on dimension 0. The callable fn takes two tensors as
    arguments. The first argument is the accumulated value computed from the
    preceding invocation of fn. If `initializer` is None, `elems` must contain
    at least one element, and its first element is used as the initializer.

    Suppose that `elems` is unpacked into `values`, a list of tensors. The shape
    of the result tensor is `[len(values)] + fn(initializer, values[0]).shape`.

    Args:
        fn: The callable to be performed.
        elems: A tensor to be unpacked on dimension 0.
        initializer: (optional) The initial value for the accumulator.
        parallel_iterations: (optional) The number of iterations allowed to run
                             in parallel.
                             back_prop: (optional) True enables back propagation.
                             swap_memory: (optional) True enables GPU-CPU memory swapping.
        name: (optional) Name prefix for the returned tensors.

    Returns:
        A tensor that packs the results of applying `fn` to the list of tensors
        unpacked from `elems`, from first to last.

    Raises:
        TypeError: if `fn` is not callable.

    Example:
        ```python
        elems = [1, 2, 3, 4, 5, 6]
        sum = scan(lambda a, x: a + x, elems)
        # sum == [1, 3, 6, 10, 15, 21]
        ```
    """
    pass


def scatter_add(ref, indices, updates, use_locking=None, name=None):
    """
    [TensorFlow Docs]
    Adds sparse updates to a variable reference.

    This operation computes

            # Scalar indices
            ref[indices, ...] += updates[...]

            # Vector indices (for each i)
            ref[indices[i], ...] += updates[i, ...]

            # High rank indices (for each i, ..., j)
            ref[indices[i, ..., j], ...] += updates[i, ..., j, ...]

    This operation outputs `ref` after the update is done.
    This makes it easier to chain operations that need to use the reset value.

    Duplicate entries are handled correctly: if multiple `indices` reference
    the same location, their contributions add.

    Requires `updates.shape = indices.shape + ref.shape[1:]`.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/ScatterAdd.png" alt>
    </div>

    Args:
        ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
             Should be from a `Variable` node.
        indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                 A tensor of indices into the first dimension of `ref`.
        updates: A `Tensor`. Must have the same type as `ref`.
                 A tensor of updated values to add to `ref`.
        use_locking: An optional `bool`. Defaults to `False`.
                     If True, the addition will be protected by a lock;
                     otherwise the behavior is undefined, but may exhibit less contention.
        name: A name for the operation (optional).

    Returns:
        Same as `ref`. Returned as a convenience for operations that want
        to use the updated values after the update is done.
    """
    pass


def scatter_sub(ref, indices, updates, use_locking=None, name=None):
    """
    [TensorFlow Docs]
    Subtracts sparse updates to a variable reference.

            # Scalar indices
            ref[indices, ...] -= updates[...]

            # Vector indices (for each i)
            ref[indices[i], ...] -= updates[i, ...]

            # High rank indices (for each i, ..., j)
            ref[indices[i, ..., j], ...] -= updates[i, ..., j, ...]

    This operation outputs `ref` after the update is done.
    This makes it easier to chain operations that need to use the reset value.

    Duplicate entries are handled correctly: if multiple `indices` reference
    the same location, their (negated) contributions add.

    Requires `updates.shape = indices.shape + ref.shape[1:]`.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/ScatterSub.png" alt>
    </div>

    Args:
        ref: A mutable `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
             Should be from a `Variable` node.
        indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                 A tensor of indices into the first dimension of `ref`.
        updates: A `Tensor`. Must have the same type as `ref`.
                 A tensor of updated values to subtract from `ref`.
        use_locking: An optional `bool`. Defaults to `False`.
                     If True, the subtraction will be protected by a lock;
                     otherwise the behavior is undefined, but may exhibit less contention.
        name: A name for the operation (optional).

    Returns:
        Same as `ref`. Returned as a convenience for operations that want
        to use the updated values after the update is done.
    """
    pass


def scatter_update(ref, indices, updates, use_locking=None, name=None):
    """
    [TensorFlow Docs]
    Applies sparse updates to a variable reference.

    This operation computes

            # Scalar indices
            ref[indices, ...] = updates[...]

            # Vector indices (for each i)
            ref[indices[i], ...] = updates[i, ...]

            # High rank indices (for each i, ..., j)
            ref[indices[i, ..., j], ...] = updates[i, ..., j, ...]

    This operation outputs `ref` after the update is done.
    This makes it easier to chain operations that need to use the reset value.

    If values in `ref` is to be updated more than once, because there are
    duplicate entires in `indices`, the order at which the updates happen
    for each value is undefined.

    Requires `updates.shape = indices.shape + ref.shape[1:]`.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/ScatterUpdate.png" alt>
    </div>

    Args:
        ref: A mutable `Tensor`. Should be from a `Variable` node.
        indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                 A tensor of indices into the first dimension of `ref`.
        updates: A `Tensor`. Must have the same type as `ref`.
                 A tensor of updated values to store in `ref`.
        use_locking: An optional `bool`. Defaults to `True`.
                     If True, the assignment will be protected by a lock;
                     otherwise the behavior is undefined, but may exhibit less contention.
        name: A name for the operation (optional).

    Returns:
        Same as `ref`. Returned as a convenience for operations that want
        to use the updated values after the update is done.
    """
    pass


def segment_max(data, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the maximum along segments of a tensor.

    Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
    for an explanation of segments.

    Computes a tensor such that
    \\(output_i = \max_j(data_j)\\) where `max` is over `j` such
    that `segment_ids[j] == i`.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/SegmentMax.png" alt>
    </div>

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                     A 1-D tensor whose rank is equal to the rank of `data`'s
                     first dimension. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def segment_mean(data, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the mean along segments of a tensor.

    Read [the section on
    Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
    of segments.

    Computes a tensor such that
    \\(output_i = \frac{\sum_j data_j}{N}\\) where `mean` is
    over `j` such that `segment_ids[j] == i` and `N` is the total number of
    values summed.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/SegmentMean.png" alt>
    </div>

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                     A 1-D tensor whose rank is equal to the rank of `data`'s
                     first dimension. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def segment_min(data, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the minimum along segments of a tensor.

    Read [the section on
    Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
    of segments.

    Computes a tensor such that
    \\(output_i = \min_j(data_j)\\) where `min` is over `j` such
    that `segment_ids[j] == i`.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/SegmentMin.png" alt>
    </div>

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                     A 1-D tensor whose rank is equal to the rank of `data`'s
                     first dimension. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def segment_prod(data, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the product along segments of a tensor.

    Read [the section on
    Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
    of segments.

    Computes a tensor such that
    \\(output_i = \prod_j data_j\\) where the product is over `j` such
    that `segment_ids[j] == i`.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/SegmentProd.png" alt>
    </div>

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                     A 1-D tensor whose rank is equal to the rank of `data`'s
                     first dimension. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def segment_sum(data, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the sum along segments of a tensor.

    Read [the section on Segmentation](../../api_docs/python/math_ops.md#segmentation)
    for an explanation of segments.

    Computes a tensor such that
    \\(output_i = \sum_j data_j\\) where sum is over `j` such
    that `segment_ids[j] == i`.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/SegmentSum.png" alt>
    </div>

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                     A 1-D tensor whose rank is equal to the rank of `data`'s
                     first dimension. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def select(condition, t, e, name=None):
    """
    [TensorFlow Docs]
    Selects elements from `t` or `e`, depending on `condition`.

    The `t`, and `e` tensors must all have the same shape,
    and the output will also have that shape. The `condition` tensor
    must be a scalar if `t` and `e` are scalars. If `t` and `e` are vectors
    or higher rank, then `condition` must be either a vector with size
    matching the first dimension of `t`, or must have the same shape as `t`.

    The `condition` tensor acts as a mask that chooses, based on the value at each
    element, whether the corresponding element / row in the output should be
    taken from `t` (if true) or `e` (if false).

    If `condition` is a vector and `t` and `e` are higher rank matrices, then
    it chooses which row (outer dimension) to copy from `t` and `e`.
    If `condition` has the same shape as `t` and `e`, then it chooses which
    element to copy from `t` and `e`.

    For example:

    ```prettyprint
    # 'condition' tensor is [[True, False]
    #                        [False, True]]
    # 't' is [[1, 2],
    #         [3, 4]]
    # 'e' is [[5, 6],
    #         [7, 8]]
    select(condition, t, e) ==> [[1, 6],
                               [7, 4]]


    # 'condition' tensor is [True, False]
    # 't' is [[1, 2],
    #         [3, 4]]
    # 'e' is [[5, 6],
    #         [7, 8]]
    select(condition, t, e) ==> [[1, 2],
                               [7, 8]]

    ```

    Args:
        condition: A `Tensor` of type `bool`.
        t: A `Tensor` which may have the same shape as `condition`.
           If `condition` is rank 1, `t` may have higher rank,
           but its first dimension must match the size of `condition`.
        e: A `Tensor` with the same type and shape as `t`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with the same type and shape as `t` and `e`.
    """
    pass


def self_adjoint_eig(input, name=None):
    """
    [TensorFlow Docs]
    Calculates the Eigen Decomposition of a square Self-Adjoint matrix.

    Only the lower-triangular part of the input will be used in this case. The
    upper-triangular part will not be read.

    The result is a M+1 x M matrix whose first row is the eigenvalues, and
    subsequent rows are eigenvectors.

    Args:
        input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
               Shape is `[M, M]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. Shape is `[M+1, M]`.
    """
    pass


def serialize_many_sparse(sp_input, name=None):
    """
    [TensorFlow Docs]
    Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

    The `SparseTensor` must have rank `R` greater than 1, and the first dimension
    is treated as the minibatch dimension. Elements of the `SparseTensor`
    must be sorted in increasing order of this first dimension. The serialized
    `SparseTensor` objects going into each row of the output `Tensor` will have
    rank `R-1`.

    The minibatch size `N` is extracted from `sparse_shape[0]`.

    Args:
        sp_input: The input rank `R` `SparseTensor`.
        name: A name prefix for the returned tensors (optional).

    Returns:
        A string matrix (2-D `Tensor`) with `N` rows and `3` columns.
        Each column represents serialized `SparseTensor`'s indices, values, and
        shape (respectively).

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def serialize_sparse(sp_input, name=None):
    """
    [TensorFlow Docs]
    Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.

    Args:
        sp_input: The input `SparseTensor`.
        name: A name prefix for the returned tensors (optional).

    Returns:
        A string 3-vector (1D `Tensor`), with each column representing the
        serialized `SparseTensor`'s indices, values, and shape (respectively).

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def set_random_seed(seed):
    """
    [TensorFlow Docs]
    Sets the graph-level random seed.

    Operations that rely on a random seed actually derive it from two seeds:
    the graph-level and operation-level seeds. This sets the graph-level seed.

    Its interactions with operation-level seeds is as follows:

        1. If neither the graph-level nor the operation seed is set:
            A random seed is used for this op.
        2. If the graph-level seed is set, but the operation seed is not:
            The system deterministically picks an operation seed in conjunction
            with the graph-level seed so that it gets a unique random sequence.
        3. If the graph-level seed is not set, but the operation seed is set:
            A default graph-level seed and the specified operation seed are used to
            determine the random sequence.
        4. If both the graph-level and the operation seed are set:
            Both seeds are used in conjunction to determine the random sequence.

    To illustrate the user-visible effects, consider these examples:

    To generate different sequences across sessions, set neither
    graph-level nor op-level seeds:

    ```python
    a = tf.random_uniform([1])
    b = tf.random_normal([1])

    print("Session 1")
    with tf.Session() as sess1:
        print(sess1.run(a))  # generates 'A1'
        print(sess1.run(a))  # generates 'A2'
        print(sess1.run(b))  # generates 'B1'
        print(sess1.run(b))  # generates 'B2'

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(a))  # generates 'A3'
        print(sess2.run(a))  # generates 'A4'
        print(sess2.run(b))  # generates 'B3'
        print(sess2.run(b))  # generates 'B4'
    ```

    To generate the same repeatable sequence for an op across sessions, set the
    seed for the op:

    ```python
    a = tf.random_uniform([1], seed=1)
    b = tf.random_normal([1])

    # Repeatedly running this block with the same graph will generate the same
    # sequence of values for 'a', but different sequences of values for 'b'.
    print("Session 1")
    with tf.Session() as sess1:
        print(sess1.run(a))  # generates 'A1'
        print(sess1.run(a))  # generates 'A2'
        print(sess1.run(b))  # generates 'B1'
        print(sess1.run(b))  # generates 'B2'

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(a))  # generates 'A1'
        print(sess2.run(a))  # generates 'A2'
        print(sess2.run(b))  # generates 'B3'
        print(sess2.run(b))  # generates 'B4'
    ```

    To make the random sequences generated by all ops be repeatable across
    sessions, set a graph-level seed:

    ```python
    tf.set_random_seed(1234)
    a = tf.random_uniform([1])
    b = tf.random_normal([1])

    # Repeatedly running this block with the same graph will generate different
    # sequences of 'a' and 'b'.
    print("Session 1")
    with tf.Session() as sess1:
        print(sess1.run(a))  # generates 'A1'
        print(sess1.run(a))  # generates 'A2'
        print(sess1.run(b))  # generates 'B1'
        print(sess1.run(b))  # generates 'B2'

    print("Session 2")
    with tf.Session() as sess2:
        print(sess2.run(a))  # generates 'A1'
        print(sess2.run(a))  # generates 'A2'
        print(sess2.run(b))  # generates 'B1'
        print(sess2.run(b))  # generates 'B2'
    ```

    Args:
        seed: integer.
              """
    pass


def shape(input, name=None):
    """
    [TensorFlow Docs]
    Returns the shape of a tensor.

    This operation returns a 1-D integer tensor representing the shape of `input`.

    For example:

    ```prettyprint
    # 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]
    shape(t) ==> [2, 2, 3]
    ```

    Args:
        input: A `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int32`.
    """
    pass


def shape_n(input, name=None):
    """
    [TensorFlow Docs]
    Returns shape of tensors.

    This operation returns N 1-D integer tensors representing shape of `input[i]s`.

    Args:
        input: A list of at least 1 `Tensor` objects of the same type.
        name: A name for the operation (optional).

    Returns:
        A list with the same number of `Tensor` objects as `input` of `Tensor` objects of type `int32`.
    """
    pass


def sigmoid(x, name=None):
    """
    [TensorFlow Docs]
    Computes sigmoid of `x` element-wise.

    Specifically, `y = 1 / (1 + exp(-x))`.

    Args:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
           or `qint32`.
        name: A name for the operation (optional).

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32`
            otherwise the return type is `quint8`.
    """
    pass


def sign(x, name=None):
    """
    [TensorFlow Docs]
    Returns an element-wise indication of the sign of a number.

    `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`.

    For complex numbers, `y = sign(x) = x / |x|` if `x != 0`, otherwise `y = 0`.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def sin(x, name=None):
    """
    [TensorFlow Docs]
    Computes sin of x element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def size(input, name=None):
    """
    [TensorFlow Docs]
    Returns the size of a tensor.

    This operation returns an integer representing the number of elements in
    `input`.

    For example:

    ```prettyprint
    # 't' is [[[1, 1,, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]
    size(t) ==> 12
    ```

    Args:
        input: A `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int32`.
    """
    pass


def slice(input_, begin, size, name=None):
    """
    [TensorFlow Docs]
    Extracts a slice from a tensor.

    This operation extracts a slice of size `size` from a tensor `input` starting
    at the location specified by `begin`. The slice `size` is represented as a
    tensor shape, where `size[i]` is the number of elements of the 'i'th dimension
    of `input` that you want to slice. The starting location (`begin`) for the
    slice is represented as an offset in each dimension of `input`. In other
    words, `begin[i]` is the offset into the 'i'th dimension of `input` that you
    want to slice from.

    `begin` is zero-based; `size` is one-based. If `size[i]` is -1,
    all remaining elements in dimension i are included in the
    slice. In other words, this is equivalent to setting:

    `size[i] = input.dim_size(i) - begin[i]`

    This operation requires that:

    `0 <= begin[i] <= begin[i] + size[i] <= Di  for i in [0, n]`

    For example:

    ```
    # 'input' is [[[1, 1, 1], [2, 2, 2]],
    #             [[3, 3, 3], [4, 4, 4]],
    #             [[5, 5, 5], [6, 6, 6]]]
    tf.slice(input, [1, 0, 0], [1, 1, 3]) ==> [[[3, 3, 3]]]
    tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
                                              [4, 4, 4]]]
    tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                             [[5, 5, 5]]]
    ```

    Args:
        input_: A `Tensor`.
        begin: An `int32` or `int64` `Tensor`.
        size: An `int32` or `int64` `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` the same type as `input`.
    """
    pass


def space_to_batch(input, paddings, block_size, name=None):
    """
    [TensorFlow Docs]
    SpaceToBatch for 4-D tensors of type T.

    Zero-pads and then rearranges (permutes) blocks of spatial data into batch.
    More specifically, this op outputs a copy of the input tensor where values from
    the `height` and `width` dimensions are moved to the `batch` dimension. After
    the zero-padding, both `height` and `width` of the input must be divisible by the
    block size.

    Args:
        input: A `Tensor`. 4-D with shape `[batch, height, width, depth]`.
        paddings: A `Tensor` of type `int32`.
                  2-D tensor of non-negative integers with shape `[2, 2]`. It specifies
                  the padding of the input with zeros across the spatial dimensions as follows:

            paddings = [[pad_top, pad_bottom], [pad_left, pad_right]]

                The effective spatial dimensions of the zero-padded input tensor will be:

            height_pad = pad_top + height + pad_bottom
            width_pad = pad_left + width + pad_right

            The attr `block_size` must be greater than one. It indicates the block size.

                * Non-overlapping blocks of size `block_size x block size` in the height and
          width dimensions are rearranged into the batch dimension at each location.
                * The batch of the output tensor is `batch * block_size * block_size`.
                * Both height_pad and width_pad must be divisible by block_size.

            The shape of the output will be:

          [batch*block_size*block_size, height_pad/block_size, width_pad/block_size,
           depth]
        block_size: An `int`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def space_to_depth(input, block_size, name=None):
    """
    [TensorFlow Docs]
    SpaceToDepth for tensors of type T.

    Rearranges blocks of spatial data, into depth. More specifically,
    this op outputs a copy of the input tensor where values from the `height`
    and `width` dimensions are moved to the `depth` dimension.
    The attr `block_size` indicates the input block size and how the data is moved.

        * Non-overlapping blocks of size `block_size x block size` are rearranged
            into depth at each location.
        * The depth of the output tensor is `input_depth * block_size * block_size`.
        * The input tensor's height and width must be divisible by block_size.

    That is, assuming the input is in the shape:
    `[batch, height, width, depth]`,
    the shape of the output will be:
    `[batch, height/block_size, width/block_size, depth*block_size*block_size]`

    This operation requires that the input tensor be of rank 4, and that
    `block_size` be >=1 and a divisor of both the input `height` and `width`.

    This operation is useful for resizing the activations between convolutions
    (but keeping all data), e.g. instead of pooling. It is also useful for training
    purely convolutional models.

    For example, given this input of shape `[1, 2, 2, 1]`, and block_size of 2:

    ```prettyprint
    x = [[[[1], [2]],
                [[3], [4]]]]
    ```

    This operation will output a tensor of shape `[1, 1, 1, 4]`:

    ```prettyprint
    [[[[1, 2, 3, 4]]]]
    ```

    Here, the input has a batch of 1 and each batch element has shape `[2, 2, 1]`,
    the corresponding output will have a single element (i.e. width and height are
    both 1) and will have a depth of 4 channels (1 * block_size * block_size).
    The output element shape is `[1, 1, 4]`.

    For an input tensor with larger depth, here of shape `[1, 2, 2, 3]`, e.g.

    ```prettyprint
    x = [[[[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]]]]
    ```

    This operation, for block_size of 2, will return the following tensor of shape
    `[1, 1, 1, 12]`

    ```prettyprint
    [[[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]]]
    ```

    Similarly, for the following input of shape `[1 4 4 1]`, and a block size of 2:

    ```prettyprint
    x = [[[[1], [2], [5], [6]],
                [[3], [4], [7], [8]],
                [[9], [10], [13], [14]],
                [[11], [12], [15], [16]]]]
    ```

    the operator will return the following tensor of shape `[1 2 2 4]`:

    ```prettyprint
    x = [[[[1, 2, 3, 4],
         [5, 6, 7, 8]],
                [[9, 10, 11, 12],
         [13, 14, 15, 16]]]]
    ```

    Args:
        input: A `Tensor`.
        block_size: An `int`. The size of the spatial block.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def sparse_add(a, b, thresh=0):
    """
    [TensorFlow Docs]
    Adds two tensors, at least one of each is a `SparseTensor`.

    If one `SparseTensor` and one `Tensor` are passed in, returns a `Tensor`. If
    both arguments are `SparseTensor`s, this returns a `SparseTensor`. The order
    of arguments does not matter. Use vanilla `tf.add()` for adding two dense
    `Tensor`s.

    The indices of any input `SparseTensor` are assumed ordered in standard
    lexicographic order. If this is not the case, before this step run
    `SparseReorder` to restore index ordering.

    If both arguments are sparse, we perform "clipping" as follows. By default,
    if two values sum to zero at some index, the output `SparseTensor` would still
    include that particular location in its index, storing a zero in the
    corresponding value slot. To override this, callers can specify `thresh`,
    indicating that if the sum has a magnitude strictly smaller than `thresh`, its
    corresponding value and index would then not be included. In particular,
    `thresh == 0.0` (default) means everything is kept and actual thresholding
    happens only for a positive value.

    For example, suppose the logical sum of two sparse operands is (densified):

            [       2]
            [.1     0]
            [ 6   -.2]

    Then,

            - thresh == 0 (the default): all 5 index/value pairs will be returned.
            - thresh == 0.11: only .1 and 0  will vanish, and the remaining three
          index/value pairs will be returned.
            - thresh == 0.21: .1, 0, and -.2 will vanish.

    Args:
        a: The first operand; `SparseTensor` or `Tensor`.
        b: The second operand; `SparseTensor` or `Tensor`. At least one operand
           must be sparse.
        thresh: A 0-D `Tensor`. The magnitude threshold that determines if an
                output value/index pair takes space. Its dtype should match that of the
                values if they are real; if the latter are complex64/complex128, then the
                dtype should be float32/float64, correspondingly.

    Returns:
        A `SparseTensor` or a `Tensor`, representing the sum.

    Raises:
        TypeError: If both `a` and `b` are `Tensor`s. Use `tf.add()` instead.
    """
    pass


def sparse_concat(concat_dim, sp_inputs, name=None, expand_nonconcat_dim=False):
    """
    [TensorFlow Docs]
    Concatenates a list of `SparseTensor` along the specified dimension.

    Concatenation is with respect to the dense versions of each sparse input.
    It is assumed that each inputs is a `SparseTensor` whose elements are ordered
    along increasing dimension number.

    If expand_nonconcat_dim is False, all inputs' shapes must match, except for
    the concat dimension. If expand_nonconcat_dim is True, then inputs' shapes are
    allowd to vary among all inputs.

    The `indices`, `values`, and `shapes` lists must have the same length.

    If expand_nonconcat_dim is False, then the output shape is identical to the
    inputs', except along the concat dimension, where it is the sum of the inputs'
    sizes along that dimension.

    If expand_nonconcat_dim is True, then the output shape along the non-concat
    dimensions will be expand to be the largest among all inputs, and it is the
    sum of the inputs sizes along the concat dimension.

    The output elements will be resorted to preserve the sort order along
    increasing dimension number.

    This op runs in `O(M log M)` time, where `M` is the total number of non-empty
    values across all inputs. This is due to the need for an internal sort in
    order to concatenate efficiently across an arbitrary dimension.

    For example, if `concat_dim = 1` and the inputs are

            sp_inputs[0]: shape = [2, 3]
            [0, 2]: "a"
            [1, 0]: "b"
            [1, 1]: "c"

            sp_inputs[1]: shape = [2, 4]
            [0, 1]: "d"
            [0, 2]: "e"

    then the output will be

            shape = [2, 7]
            [0, 2]: "a"
            [0, 4]: "d"
            [0, 5]: "e"
            [1, 0]: "b"
            [1, 1]: "c"

    Graphically this is equivalent to doing

            [    a] concat [  d e  ] = [    a   d e  ]
            [b c  ]        [       ]   [b c          ]

    Another example, if 'concat_dim = 1' and the inputs are

            sp_inputs[0]: shape = [3, 3]
            [0, 2]: "a"
            [1, 0]: "b"
            [2, 1]: "c"

            sp_inputs[1]: shape = [2, 4]
            [0, 1]: "d"
            [0, 2]: "e"

    if expand_nonconcat_dim = False, this will result in an error. But if
    expand_nonconcat_dim = True, this will result in:

            shape = [3, 7]
            [0, 2]: "a"
            [0, 4]: "d"
            [0, 5]: "e"
            [1, 0]: "b"
            [2, 1]: "c"

    Graphically this is equivalent to doing

            [    a] concat [  d e  ] = [    a   d e  ]
            [b    ]        [       ]   [b            ]
            [  c  ]                    [  c          ]


    Args:
        concat_dim: Dimension to concatenate along.
        sp_inputs: List of `SparseTensor` to concatenate.
        name: A name prefix for the returned tensors (optional).
              expand_nonconcat_dim: Whether to allow the expansion in the non-concat
              dimensions. Defaulted to False.

    Returns:
        A `SparseTensor` with the concatenated output.

    Raises:
        TypeError: If `sp_inputs` is not a list of `SparseTensor`.
    """
    pass


def sparse_fill_empty_rows(sp_input, default_value, name=None):
    """
    [TensorFlow Docs]
    Fills empty rows in the input 2-D `SparseTensor` with a default value.

    This op adds entries with the specified `default_value` at index
    `[row, 0]` for any row in the input that does not already have a value.

    For example, suppose `sp_input` has shape `[5, 6]` and non-empty values:

            [0, 1]: a
            [0, 3]: b
            [2, 0]: c
            [3, 1]: d

    Rows 1 and 4 are empty, so the output will be of shape `[5, 6]` with values:

            [0, 1]: a
            [0, 3]: b
            [1, 0]: default_value
            [2, 0]: c
            [3, 1]: d
            [4, 0]: default_value

    Note that the input may have empty columns at the end, with no effect on
    this op.

    The output `SparseTensor` will be in row-major order and will have the
    same shape as the input.

    This op also returns an indicator vector such that

            empty_row_indicator[i] = True iff row i was an empty row.

    Args:
        sp_input: A `SparseTensor` with shape `[N, M]`.
        default_value: The value to fill for empty rows, with the same type as
                       `sp_input.`
        name: A name prefix for the returned tensors (optional)

    Returns:
        sp_ordered_output: A `SparseTensor` with shape `[N, M]`, and with all empty
            rows filled in with `default_value`.
        empty_row_indicator: A bool vector of length `N` indicating whether each
            input row was empty.

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def sparse_mask(a, mask_indices, name=None):
    """
    [TensorFlow Docs]
    Masks elements of `IndexedSlices`.

    Given an `IndexedSlices` instance `a`, returns another `IndexedSlices` that
    contains a subset of the slices of `a`. Only the slices at indices specified
    in `mask_indices` are returned.

    This is useful when you need to extract a subset of slices in an
    `IndexedSlices` object.

    For example:

    ```python
    # `a` contains slices at indices [12, 26, 37, 45] from a large tensor
    # with shape [1000, 10]
    a.indices => [12, 26, 37, 45]
    tf.shape(a.values) => [4, 10]

    # `b` will be the subset of `a` slices at its second and third indices, so
    # we want to mask of its first and last indices (which are at absolute
    # indices 12, 45)
    b = tf.sparse_mask(a, [12, 45])

    b.indices => [26, 37]
    tf.shape(b.values) => [2, 10]

    ```

    Args:
        * `a`: An `IndexedSlices` instance.
        * `mask_indices`: Indices of elements to mask.
        * `name`: A name for the operation (optional).

    Returns:
        The masked `IndexedSlices` instance.
    """
    pass


def _sparse_mat_mul(a, b, transpose_a=None, transpose_b=None,
                    a_is_sparse=None, b_is_sparse=None, name=None):
    """
    [TensorFlow Docs]
    Multiply matrix "a" by matrix "b".

    The inputs must be two-dimensional matrices and the inner dimension of "a" must
    match the outer dimension of "b". This op is optimized for the case where at
    least one of "a" or "b" is sparse. The breakeven for using this versus a dense
    matrix multiply on one platform was 30% zero values in the sparse matrix.

    Args:
        a: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
        b: A `Tensor`. Must be one of the following types: `float32`, `bfloat16`.
        transpose_a: An optional `bool`. Defaults to `False`.
                     transpose_b: An optional `bool`. Defaults to `False`.
                     a_is_sparse: An optional `bool`. Defaults to `False`.
                     b_is_sparse: An optional `bool`. Defaults to `False`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `float32`.
    """
    pass


def sparse_merge(sp_ids, sp_values, vocab_size, name=None):
    """
    [TensorFlow Docs]
    Combines a batch of feature ids and values into a single `SparseTensor`.

    The most common use case for this function occurs when feature ids and
    their corresponding values are stored in `Example` protos on disk.
    `parse_example` will return a batch of ids and a batch of values, and this
    function joins them into a single logical `SparseTensor` for use in
    functions such as `sparse_tensor_dense_matmul`, `sparse_to_dense`, etc.

    The `SparseTensor` returned by this function has the following properties:

        - `indices` is equivalent to `sp_ids.indices` with the last
            dimension discarded and replaced with `sp_ids.values`.
        - `values` is simply `sp_values.values`.
        - If `sp_ids.shape = [D0, D1, ..., Dn, K]`, then
            `output.shape = [D0, D1, ..., Dn, vocab_size]`.

    For example, consider the following feature vectors:

        vector1 = [-3, 0, 0, 0, 0, 0]
        vector2 = [ 0, 1, 0, 4, 1, 0]
        vector3 = [ 5, 0, 0, 9, 0, 0]

    These might be stored sparsely in the following Example protos by storing
    only the feature ids (column number if the vectors are treated as a matrix)
    of the non-zero elements and the corresponding values:

        examples = [Example(features={
                    "ids": Feature(int64_list=Int64List(value=[0])),
                    "values": Feature(float_list=FloatList(value=[-3]))}),
                Example(features={
                    "ids": Feature(int64_list=Int64List(value=[1, 4, 3])),
                    "values": Feature(float_list=FloatList(value=[1, 1, 4]))}),
                Example(features={
                    "ids": Feature(int64_list=Int64List(value=[0, 3])),
                    "values": Feature(float_list=FloatList(value=[5, 9]))})]

    The result of calling parse_example on these examples will produce a
    dictionary with entries for "ids" and "values". Passing those two objects
    to this function along with vocab_size=6, will produce a `SparseTensor` that
    sparsely represents all three instances. Namely, the `indices` property will
    contain the coordinates of the non-zero entries in the feature matrix (the
    first dimension is the row number in the matrix, i.e., the index within the
    batch, and the second dimension is the column number, i.e., the feature id);
    `values` will contain the actual values. `shape` will be the shape of the
    original matrix, i.e., (3, 6). For our example above, the output will be
    equal to:

        SparseTensor(indices=[[0, 0], [1, 1], [1, 3], [1, 4], [2, 0], [2, 3]],
                 values=[-3, 1, 4, 1, 5, 9],
                 shape=[3, 6])

    Args:
        sp_ids: A `SparseTensor` with `values` property of type `int32`
                or `int64`.
        sp_values: A`SparseTensor` of any type.
        vocab_size: A scalar `int64` Tensor (or Python int) containing the new size
                    of the last dimension, `all(0 <= sp_ids.values < vocab_size)`.
        name: A name prefix for the returned tensors (optional)

    Returns:
        A `SparseTensor` compactly representing a batch of feature ids and values,
        useful for passing to functions that expect such a `SparseTensor`.

    Raises:
        TypeError: If `sp_ids` or `sp_values` are not a `SparseTensor`.
    """
    pass


def sparse_placeholder(dtype, shape=None, name=None):
    """
    [TensorFlow Docs]
    Inserts a placeholder for a sparse tensor that will be always fed.

    **Important**: This sparse tensor will produce an error if evaluated.
    Its value must be fed using the `feed_dict` optional argument to
    `Session.run()`, `Tensor.eval()`, or `Operation.run()`.

    For example:

    ```python
    x = tf.sparse_placeholder(tf.float32)
    y = tf.sparse_reduce_sum(x)

    with tf.Session() as sess:
        print(sess.run(y))  # ERROR: will fail because x was not fed.

        indices = np.array([[3, 2, 0], [4, 5, 1]], dtype=np.int64)
        values = np.array([1.0, 2.0], dtype=np.float32)
        shape = np.array([7, 9, 2], dtype=np.int64)
        print(sess.run(y, feed_dict={
            x: tf.SparseTensorValue(indices, values, shape)}))  # Will succeed.
               print(sess.run(y, feed_dict={
            x: (indices, values, shape)}))  # Will succeed.

        sp = tf.SparseTensor(indices=indices, values=values, shape=shape)
        sp_value = sp.eval(session)
        print(sess.run(y, feed_dict={x: sp_value}))  # Will succeed.
    ```

    Args:
        dtype: The type of `values` elements in the tensor to be fed.
        shape: The shape of the tensor to be fed (optional). If the shape is not
               specified, you can feed a sparse tensor of any shape.
        name: A name for prefixing the operations (optional).

    Returns:
        A `SparseTensor` that may be used as a handle for feeding a value, but not
        evaluated directly.
    """
    pass


def sparse_reduce_sum(sp_input, reduction_axes=None, keep_dims=False):
    """
    [TensorFlow Docs]
    Computes the sum of elements across dimensions of a SparseTensor.

    This Op takes a SparseTensor and is the sparse counterpart to
    `tf.reduce_sum()`. In particular, this Op also returns a dense `Tensor`
    instead of a sparse one.

    Reduces `sp_input` along the dimensions given in `reduction_axes`. Unless
    `keep_dims` is true, the rank of the tensor is reduced by 1 for each entry in
    `reduction_axes`. If `keep_dims` is true, the reduced dimensions are retained
    with length 1.

    If `reduction_axes` has no entries, all dimensions are reduced, and a tensor
    with a single element is returned. Additionally, the axes can be negative,
    similar to the indexing rules in Python.

    For example:

    ```python
    # 'x' represents [[1, ?, 1]
    #                 [?, 1, ?]]
    # where ? is implictly-zero.
    tf.sparse_reduce_sum(x) ==> 3
    tf.sparse_reduce_sum(x, 0) ==> [1, 1, 1]
    tf.sparse_reduce_sum(x, 1) ==> [2, 1]  # Can also use -1 as the axis.
    tf.sparse_reduce_sum(x, 1, keep_dims=True) ==> [[2], [1]]
    tf.sparse_reduce_sum(x, [0, 1]) ==> 3
    ```

    Args:
        sp_input: The SparseTensor to reduce. Should have numeric type.
        reduction_axes: The dimensions to reduce; list or scalar. If `None` (the
                        default), reduces all dimensions.
        keep_dims: If true, retain reduced dimensions with length 1.

    Returns:
        The reduced Tensor.
    """
    pass


def sparse_reorder(sp_input, name=None):
    """
    [TensorFlow Docs]
    Reorders a `SparseTensor` into the canonical, row-major ordering.

    Note that by convention, all sparse ops preserve the canonical ordering
    along increasing dimension number. The only time ordering can be violated
    is during manual manipulation of the indices and values to add entries.

    Reordering does not affect the shape of the `SparseTensor`.

    For example, if `sp_input` has shape `[4, 5]` and `indices` / `values`:

            [0, 3]: b
            [0, 1]: a
            [3, 1]: d
            [2, 0]: c

    then the output will be a `SparseTensor` of shape `[4, 5]` and
    `indices` / `values`:

            [0, 1]: a
            [0, 3]: b
            [2, 0]: c
            [3, 1]: d

    Args:
        sp_input: The input `SparseTensor`.
        name: A name prefix for the returned tensors (optional)

    Returns:
        A `SparseTensor` with the same shape and non-empty values, but in
        canonical ordering.

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def sparse_reset_shape(sp_input, new_shape=None):
    """
    [TensorFlow Docs]
    Resets the shape of a `SparseTensor` with indices and values unchanged.

    If `new_shape` is None, returns a copy of `sp_input` with its shape reset
    to the tight bounding box of `sp_input`.

    If `new_shape` is provided, then it must be larger or equal in all dimensions
    compared to the shape of `sp_input`. When this condition is met, the returned
    SparseTensor will have its shape reset to `new_shape` and its indices and
    values unchanged from that of `sp_input.`

    For example:

        Consider a `sp_input` with shape [2, 3, 5]:

            [0, 0, 1]: a
            [0, 1, 0]: b
            [0, 2, 2]: c
            [1, 0, 3]: d

        - It is an error to set `new_shape` as [3, 7] since this represents a
            rank-2 tensor while `sp_input` is rank-3. This is either a ValueError
            during graph construction (if both shapes are known) or an OpError during
            run time.

        - Setting `new_shape` as [2, 3, 6] will be fine as this shape is larger or
            eqaul in every dimension compared to the original shape [2, 3, 5].

        - On the other hand, setting new_shape as [2, 3, 4] is also an error: The
            third dimension is smaller than the original shape [2, 3, 5] (and an
            `InvalidArgumentError` will be raised).

        - If `new_shape` is None, the returned SparseTensor will have a shape
            [2, 3, 4], which is the tight bounding box of `sp_input`.

    Args:
        sp_input: The input `SparseTensor`.
        new_shape: None or a vector representing the new shape for the returned
                   `SpraseTensor`.

    Returns:
        A `SparseTensor` indices and values unchanged from `input_sp`. Its shape is
            `new_shape` if that is set. Otherwise it is  the tight bounding box of
       `input_sp`

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
        ValueError: If `new_shape` represents a tensor with a different rank from
            that of `sp_input` (if shapes are known when graph is constructed).
        OpError:
            - If `new_shape` has dimension sizes that are too small.
            - If shapes are not known during graph construction time, and during run
                time it is found out that the ranks do not match.
    """
    pass


def sparse_retain(sp_input, to_retain):
    """
    [TensorFlow Docs]
    Retains specified non-empty values within a `SparseTensor`.

    For example, if `sp_input` has shape `[4, 5]` and 4 non-empty string values:

            [0, 1]: a
            [0, 3]: b
            [2, 0]: c
            [3, 1]: d

    and `to_retain = [True, False, False, True]`, then the output will
    be a `SparseTensor` of shape `[4, 5]` with 2 non-empty values:

            [0, 1]: a
            [3, 1]: d

    Args:
        sp_input: The input `SparseTensor` with `N` non-empty elements.
        to_retain: A bool vector of length `N` with `M` true values.

    Returns:
        A `SparseTensor` with the same shape as the input and `M` non-empty
        elements corresponding to the true positions in `to_retain`.

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def sparse_segment_mean(data, indices, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the mean along sparse segments of a tensor.

    Read [the section on
    Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
    of segments.

    Like `SegmentMean`, but `segment_ids` can have rank less than `data`'s first
    dimension, selecting a subset of dimension 0, specified by `indices`.

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        indices: A `Tensor` of type `int32`.
                 A 1-D tensor. Has same rank as `segment_ids`.
        segment_ids: A `Tensor` of type `int32`.
                     A 1-D tensor. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def sparse_segment_mean_grad(grad, indices, segment_ids, output_dim0,
                             name=None):
    """
    [TensorFlow Docs]
    Computes gradients for SparseSegmentMean.

    Returns tensor "output" with same shape as grad, except for dimension 0 whose
    value is output_dim0.

    Args:
        grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
              gradient propagated to the SparseSegmentMean op.
        indices: A `Tensor` of type `int32`.
                 indices passed to the corresponding SparseSegmentMean op.
        segment_ids: A `Tensor` of type `int32`.
                     segment_ids passed to the corresponding SparseSegmentMean op.
        output_dim0: A `Tensor` of type `int32`.
                     dimension 0 of "data" passed to SparseSegmentMean op.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `grad`.
    """
    pass


def sparse_segment_sqrt_n(data, indices, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the sum along sparse segments of a tensor divided by the sqrt of N.

    N is the size of the segment being reduced.

    Read [the section on
    Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
    of segments.

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        indices: A `Tensor` of type `int32`.
                 A 1-D tensor. Has same rank as `segment_ids`.
        segment_ids: A `Tensor` of type `int32`.
                     A 1-D tensor. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def sparse_segment_sqrt_n_grad(grad, indices, segment_ids, output_dim0,
                               name=None):
    """
    [TensorFlow Docs]
    Computes gradients for SparseSegmentSqrtN.

    Returns tensor "output" with same shape as grad, except for dimension 0 whose
    value is output_dim0.

    Args:
        grad: A `Tensor`. Must be one of the following types: `float32`, `float64`.
              gradient propagated to the SparseSegmentSqrtN op.
        indices: A `Tensor` of type `int32`.
                 indices passed to the corresponding SparseSegmentSqrtN op.
        segment_ids: A `Tensor` of type `int32`.
                     segment_ids passed to the corresponding SparseSegmentSqrtN op.
        output_dim0: A `Tensor` of type `int32`.
                     dimension 0 of "data" passed to SparseSegmentSqrtN op.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `grad`.
    """
    pass


def sparse_segment_sum(data, indices, segment_ids, name=None):
    """
    [TensorFlow Docs]
    Computes the sum along sparse segments of a tensor.

    Read [the section on
    Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
    of segments.

    Like `SegmentSum`, but `segment_ids` can have rank less than `data`'s first
    dimension, selecting a subset of dimension 0, specified by `indices`.

    For example:

    ```prettyprint
    c = tf.constant([[1,2,3,4], [-1,-2,-3,-4], [5,6,7,8]])

    # Select two rows, one segment.
    tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 0]))
        ==> [[0 0 0 0]]

    # Select two rows, two segment.
    tf.sparse_segment_sum(c, tf.constant([0, 1]), tf.constant([0, 1]))
        ==> [[ 1  2  3  4]
         [-1 -2 -3 -4]]

    # Select all rows, two segments.
    tf.sparse_segment_sum(c, tf.constant([0, 1, 2]), tf.constant([0, 0, 1]))
        ==> [[0 0 0 0]
         [5 6 7 8]]

    # Which is equivalent to:
    tf.segment_sum(c, tf.constant([0, 0, 1]))
    ```

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        indices: A `Tensor` of type `int32`.
                 A 1-D tensor. Has same rank as `segment_ids`.
        segment_ids: A `Tensor` of type `int32`.
                     A 1-D tensor. Values should be sorted and can be repeated.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `k`, the number of segments.
    """
    pass


def sparse_softmax(sp_input, name=None):
    """
    [TensorFlow Docs]
    Applies softmax to a batched N-D `SparseTensor`.

    The inputs represent an N-D SparseTensor  with logical shape `[..., B, C]`
    (where `N >= 2`), and with indices sorted in the canonical lexicographic
    order.

    This op is equivalent to applying the normal `tf.nn.softmax()` to each
    innermost logical submatrix with shape `[B, C]`, but with the catch that *the
    implicitly zero elements do not participate*. Specifically, the algorithm is
    equivalent to:

        (1) Applies `tf.nn.softmax()` to a densified view of each innermost
                submatrix with shape `[B, C]`, along the size-C dimension;
        (2) Masks out the original implicitly-zero locations;
        (3) Renormalizes the remaining elements.

    Hence, the `SparseTensor` result has exactly the same non-zero indices and
    shape.

    Example:

    ```python
    # First batch:
    # [?   e.]
    # [1. ? ]
    # Second batch:
    # [e   ? ]
    # [e   e ]
    shape = [2, 2, 2]  # 3-D SparseTensor
    values = np.asarray([[[0., np.e], [1., 0.]], [[np.e, 0.], [np.e, np.e]]])
    indices = np.vstack(np.where(values)).astype(np.int64).T

    result = tf.sparse_softmax(tf.SparseTensor(indices, values, shape))
    # ...returning a 3-D SparseTensor, equivalent to:
    # [?   1.]     [1    ?]
    # [1. ? ] and [.5  .5]
    # where ? means implicitly zero.
    ```

    Args:
        sp_input: N-D `SparseTensor`, where `N >= 2`.
        name: optional name of the operation.
              Returns:
              output: N-D `SparseTensor` representing the results.
              """
    pass


def sparse_split(split_dim, num_split, sp_input, name=None):
    """
    [TensorFlow Docs]
    Split a `SparseTensor` into `num_split` tensors along `split_dim`.

    If the `sp_input.shape[split_dim]` is not an integer multiple of `num_split`
    each slice starting from 0:`shape[split_dim] % num_split` gets extra one
    dimension. For example, if `split_dim = 1` and `num_split = 2` and the
    input is:

            input_tensor = shape = [2, 7]
            [    a   d e  ]
            [b c          ]

    Graphically the output tensors are:

            output_tensor[0] =
            [    a ]
            [b c   ]

            output_tensor[1] =
            [ d e  ]
            [      ]

    Args:
        split_dim: A 0-D `int32` `Tensor`. The dimension along which to split.
        num_split: A Python integer. The number of ways to split.
        sp_input: The `SparseTensor` to split.
        name: A name for the operation (optional).

    Returns:
        `num_split` `SparseTensor` objects resulting from splitting `value`.

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def sparse_tensor_dense_matmul(sp_a, b, adjoint_a=False, adjoint_b=False,
                               name=None):
    # pylint: disable=line-too-long
    """Multiply SparseTensor (of rank 2) "A" by dense matrix "B".

    No validity checking is performed on the indices of A. However, the following
    input format is recommended for optimal behavior:

    if adjoint_a == false:
        A should be sorted in lexicographically increasing order. Use
        sparse_reorder if you're not sure.
    if adjoint_a == true:
        A should be sorted in order of increasing dimension 1 (i.e., "column major"
        order instead of "row major" order).

    Deciding when to use sparse_tensor_dense_matmul vs. matmul(sp_a=True):

    There are a number of questions to ask in the decision process, including:

    * Will the SparseTensor A fit in memory if densified?
    * Is the column count of the product large (>> 1)?
    * Is the density of A larger than approximately 15%?

    If the answer to several of these questions is yes, consider
    converting the SparseTensor to a dense one and using tf.matmul with sp_a=True.

    This operation tends to perform well when A is more sparse, if the column size
    of the product is small (e.g. matrix-vector multiplication), if sp_a.shape
    takes on large values.

    Below is a rough speed comparison between sparse_tensor_dense_matmul,
    labelled 'sparse', and matmul(sp_a=True), labelled 'dense'. For purposes of
    the comparison, the time spent converting from a SparseTensor to a dense
    Tensor is not included, so it is overly conservative with respect to
    the time ratio.

    Benchmark system:
    CPU: Intel Ivybridge with HyperThreading (6 cores) dL1:32KB dL2:256KB dL3:12MB
    GPU: NVidia Tesla k40c

    Compiled with:
    -c opt --config=cuda --copt=-mavx

    ```tensorflow/python/sparse_tensor_dense_matmul_op_test --benchmarks
    A sparse [m, k] with % nonzero values between 1% and 80%
    B dense [k, n]

    % nnz    n       gpu     m       k       dt(dense)       dt(sparse)      dt(sparse)/dt(dense)
    0.01     1       True    100     100     0.000221166     0.00010154      0.459112
    0.01     1       True    100     1000    0.00033858      0.000109275     0.322745
    0.01     1       True    1000    100     0.000310557     9.85661e-05     0.317385
    0.01     1       True    1000    1000    0.0008721       0.000100875     0.115669
    0.01     1       False   100     100     0.000208085     0.000107603     0.51711
    0.01     1       False   100     1000    0.000327112     9.51118e-05     0.290762
    0.01     1       False   1000    100     0.000308222     0.00010345      0.335635
    0.01     1       False   1000    1000    0.000865721     0.000101397     0.117124
    0.01     10      True    100     100     0.000218522     0.000105537     0.482958
    0.01     10      True    100     1000    0.000340882     0.000111641     0.327506
    0.01     10      True    1000    100     0.000315472     0.000117376     0.372064
    0.01     10      True    1000    1000    0.000905493     0.000123263     0.136128
    0.01     10      False   100     100     0.000221529     9.82571e-05     0.44354
    0.01     10      False   100     1000    0.000330552     0.000112615     0.340687
    0.01     10      False   1000    100     0.000341277     0.000114097     0.334324
    0.01     10      False   1000    1000    0.000819944     0.000120982     0.147549
    0.01     25      True    100     100     0.000207806     0.000105977     0.509981
    0.01     25      True    100     1000    0.000322879     0.00012921      0.400181
    0.01     25      True    1000    100     0.00038262      0.000141583     0.370035
    0.01     25      True    1000    1000    0.000865438     0.000202083     0.233504
    0.01     25      False   100     100     0.000209401     0.000104696     0.499979
    0.01     25      False   100     1000    0.000321161     0.000130737     0.407076
    0.01     25      False   1000    100     0.000377012     0.000136801     0.362856
    0.01     25      False   1000    1000    0.000861125     0.00020272      0.235413
    0.2      1       True    100     100     0.000206952     9.69219e-05     0.46833
    0.2      1       True    100     1000    0.000348674     0.000147475     0.422959
    0.2      1       True    1000    100     0.000336908     0.00010122      0.300439
    0.2      1       True    1000    1000    0.001022        0.000203274     0.198898
    0.2      1       False   100     100     0.000207532     9.5412e-05      0.459746
    0.2      1       False   100     1000    0.000356127     0.000146824     0.41228
    0.2      1       False   1000    100     0.000322664     0.000100918     0.312764
    0.2      1       False   1000    1000    0.000998987     0.000203442     0.203648
    0.2      10      True    100     100     0.000211692     0.000109903     0.519165
    0.2      10      True    100     1000    0.000372819     0.000164321     0.440753
    0.2      10      True    1000    100     0.000338651     0.000144806     0.427596
    0.2      10      True    1000    1000    0.00108312      0.000758876     0.70064
    0.2      10      False   100     100     0.000215727     0.000110502     0.512231
    0.2      10      False   100     1000    0.000375419     0.0001613       0.429653
    0.2      10      False   1000    100     0.000336999     0.000145628     0.432132
    0.2      10      False   1000    1000    0.00110502      0.000762043     0.689618
    0.2      25      True    100     100     0.000218705     0.000129913     0.594009
    0.2      25      True    100     1000    0.000394794     0.00029428      0.745402
    0.2      25      True    1000    100     0.000404483     0.0002693       0.665788
    0.2      25      True    1000    1000    0.0012002       0.00194494      1.62052
    0.2      25      False   100     100     0.000221494     0.0001306       0.589632
    0.2      25      False   100     1000    0.000396436     0.000297204     0.74969
    0.2      25      False   1000    100     0.000409346     0.000270068     0.659754
    0.2      25      False   1000    1000    0.00121051      0.00193737      1.60046
    0.5      1       True    100     100     0.000214981     9.82111e-05     0.456836
    0.5      1       True    100     1000    0.000415328     0.000223073     0.537101
    0.5      1       True    1000    100     0.000358324     0.00011269      0.314492
    0.5      1       True    1000    1000    0.00137612      0.000437401     0.317851
    0.5      1       False   100     100     0.000224196     0.000101423     0.452386
    0.5      1       False   100     1000    0.000400987     0.000223286     0.556841
    0.5      1       False   1000    100     0.000368825     0.00011224      0.304318
    0.5      1       False   1000    1000    0.00136036      0.000429369     0.31563
    0.5      10      True    100     100     0.000222125     0.000112308     0.505608
    0.5      10      True    100     1000    0.000461088     0.00032357      0.701753
    0.5      10      True    1000    100     0.000394624     0.000225497     0.571422
    0.5      10      True    1000    1000    0.00158027      0.00190898      1.20801
    0.5      10      False   100     100     0.000232083     0.000114978     0.495418
    0.5      10      False   100     1000    0.000454574     0.000324632     0.714146
    0.5      10      False   1000    100     0.000379097     0.000227768     0.600817
    0.5      10      False   1000    1000    0.00160292      0.00190168      1.18638
    0.5      25      True    100     100     0.00023429      0.000151703     0.647501
    0.5      25      True    100     1000    0.000497462     0.000598873     1.20386
    0.5      25      True    1000    100     0.000460778     0.000557038     1.20891
    0.5      25      True    1000    1000    0.00170036      0.00467336      2.74845
    0.5      25      False   100     100     0.000228981     0.000155334     0.678371
    0.5      25      False   100     1000    0.000496139     0.000620789     1.25124
    0.5      25      False   1000    100     0.00045473      0.000551528     1.21287
    0.5      25      False   1000    1000    0.00171793      0.00467152      2.71927
    0.8      1       True    100     100     0.000222037     0.000105301     0.47425
    0.8      1       True    100     1000    0.000410804     0.000329327     0.801664
    0.8      1       True    1000    100     0.000349735     0.000131225     0.375212
    0.8      1       True    1000    1000    0.00139219      0.000677065     0.48633
    0.8      1       False   100     100     0.000214079     0.000107486     0.502085
    0.8      1       False   100     1000    0.000413746     0.000323244     0.781261
    0.8      1       False   1000    100     0.000348983     0.000131983     0.378193
    0.8      1       False   1000    1000    0.00136296      0.000685325     0.50282
    0.8      10      True    100     100     0.000229159     0.00011825      0.516017
    0.8      10      True    100     1000    0.000498845     0.000532618     1.0677
    0.8      10      True    1000    100     0.000383126     0.00029935      0.781336
    0.8      10      True    1000    1000    0.00162866      0.00307312      1.88689
    0.8      10      False   100     100     0.000230783     0.000124958     0.541452
    0.8      10      False   100     1000    0.000493393     0.000550654     1.11606
    0.8      10      False   1000    100     0.000377167     0.000298581     0.791642
    0.8      10      False   1000    1000    0.00165795      0.00305103      1.84024
    0.8      25      True    100     100     0.000233496     0.000175241     0.75051
    0.8      25      True    100     1000    0.00055654      0.00102658      1.84458
    0.8      25      True    1000    100     0.000463814     0.000783267     1.68875
    0.8      25      True    1000    1000    0.00186905      0.00755344      4.04132
    0.8      25      False   100     100     0.000240243     0.000175047     0.728625
    0.8      25      False   100     1000    0.000578102     0.00104499      1.80763
    0.8      25      False   1000    100     0.000485113     0.000776849     1.60138
    0.8      25      False   1000    1000    0.00211448      0.00752736      3.55992
    ```

    Args:
        sp_a: SparseTensor A, of rank 2.
        b: A dense Matrix with the same dtype as sp_a.
        adjoint_a: Use the adjoint of A in the matrix multiply. If A is complex,
                   this is transpose(conj(A)). Otherwise it's transpose(A).
                   adjoint_b: Use the adjoint of B in the matrix multiply. If B is complex,
                   this is transpose(conj(B)). Otherwise it's transpose(B).
        name: A name prefix for the returned tensors (optional)

    Returns:
        A dense matrix (pseudo-code in dense np.matrix notation):
            A = A.H if adjoint_a else A
            B = B.H if adjoint_b else B
            return A*B
    """
    pass


def sparse_tensor_to_dense(sp_input,
                           default_value=0,
                           validate_indices=True,
                           name=None):
    """
    [TensorFlow Docs]
    Converts a `SparseTensor` into a dense tensor.

    This op is a convenience wrapper around `sparse_to_dense` for `SparseTensor`s.

    For example, if `sp_input` has shape `[3, 5]` and non-empty string values:

            [0, 1]: a
            [0, 3]: b
            [2, 0]: c

    and `default_value` is `x`, then the output will be a dense `[3, 5]`
    string tensor with values:

            [[x a x b x]
       [x x x x x]
       [c x x x x]]

    Indices must be without repeats. This is only
    tested if validate_indices is True.

    Args:
        sp_input: The input `SparseTensor`.
        default_value: Scalar value to set for indices not specified in
                       `sp_input`. Defaults to zero.
        validate_indices: A boolean value. If `True`, indices are checked to make
                          sure they are sorted in lexicographic order and that there are no repeats.
        name: A name prefix for the returned tensors (optional).

    Returns:
        A dense tensor with shape `sp_input.shape` and values specified by
        the non-empty values in `sp_input`. Indices not in `sp_input` are assigned
        `default_value`.

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def sparse_to_dense(sparse_indices,
                    output_shape,
                    sparse_values,
                    default_value=0,
                    validate_indices=True,
                    name=None):
    """
    [TensorFlow Docs]
    Converts a sparse representation into a dense tensor.

    Builds an array `dense` with shape `output_shape` such that

    ```python
    # If sparse_indices is scalar
    dense[i] = (i == sparse_indices ? sparse_values : default_value)

    # If sparse_indices is a vector, then for each i
    dense[sparse_indices[i]] = sparse_values[i]

    # If sparse_indices is an n by d matrix, then for each i in [0, n)
    dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
    ```

    All other values in `dense` are set to `default_value`. If `sparse_values`
    is a scalar, all sparse indices are set to this single value.

    Indices should be sorted in lexicographic order, and indices must not
    contain any repeats. If `validate_indices` is True, these properties
    are checked during execution.

    Args:
        sparse_indices: A 0-D, 1-D, or 2-D `Tensor` of type `int32` or `int64`.
                        `sparse_indices[i]` contains the complete index where `sparse_values[i]`
                        will be placed.
        output_shape: A 1-D `Tensor` of the same type as `sparse_indices`. Shape
                      of the dense output tensor.
        sparse_values: A 0-D or 1-D `Tensor`. Values corresponding to each row of
                       `sparse_indices`, or a scalar value to be used for all sparse indices.
        default_value: A 0-D `Tensor` of the same type as `sparse_values`. Value
                       to set for indices not specified in `sparse_indices`. Defaults to zero.
        validate_indices: A boolean value. If True, indices are checked to make
                          sure they are sorted in lexicographic order and that there are no repeats.
        name: A name for the operation (optional).

    Returns:
        Dense `Tensor` of shape `output_shape`. Has the same type as
        `sparse_values`.
    """
    pass


def sparse_to_indicator(sp_input, vocab_size, name=None):
    """
    [TensorFlow Docs]
    Converts a `SparseTensor` of ids into a dense bool indicator tensor.

    The last dimension of `sp_input.indices` is discarded and replaced with
    the values of `sp_input`. If `sp_input.shape = [D0, D1, ..., Dn, K]`, then
    `output.shape = [D0, D1, ..., Dn, vocab_size]`, where

            output[d_0, d_1, ..., d_n, sp_input[d_0, d_1, ..., d_n, k]] = True

    and False elsewhere in `output`.

    For example, if `sp_input.shape = [2, 3, 4]` with non-empty values:

            [0, 0, 0]: 0
            [0, 1, 0]: 10
            [1, 0, 3]: 103
            [1, 1, 2]: 150
            [1, 1, 3]: 149
            [1, 1, 4]: 150
            [1, 2, 1]: 121

    and `vocab_size = 200`, then the output will be a `[2, 3, 200]` dense bool
    tensor with False everywhere except at positions

            (0, 0, 0), (0, 1, 10), (1, 0, 103), (1, 1, 149), (1, 1, 150),
            (1, 2, 121).

    Note that repeats are allowed in the input SparseTensor.
    This op is useful for converting `SparseTensor`s into dense formats for
    compatibility with ops that expect dense tensors.

    The input `SparseTensor` must be in row-major order.

    Args:
        sp_input: A `SparseTensor` with `values` property of type `int32` or
                  `int64`.
        vocab_size: A scalar int64 Tensor (or Python int) containing the new size
                    of the last dimension, `all(0 <= sp_input.values < vocab_size)`.
        name: A name prefix for the returned tensors (optional)

    Returns:
        A dense bool indicator tensor representing the indices with specified value.

    Raises:
        TypeError: If `sp_input` is not a `SparseTensor`.
    """
    pass


def split(split_dim, num_split, value, name="split"):
    """
    [TensorFlow Docs]
    Splits a tensor into `num_split` tensors along one dimension.

    Splits `value` along dimension `split_dim` into `num_split` smaller tensors.
    Requires that `num_split` evenly divide `value.shape[split_dim]`.

    For example:

    ```python
    # 'value' is a tensor with shape [5, 30]
    # Split 'value' into 3 tensors along dimension 1
    split0, split1, split2 = tf.split(1, 3, value)
    tf.shape(split0) ==> [5, 10]
    ```

    Args:
        split_dim: A 0-D `int32` `Tensor`. The dimension along which to split.
                   Must be in the range `[0, rank(value))`.
        num_split: A Python integer. The number of ways to split.
        value: The `Tensor` to split.
        name: A name for the operation (optional).

    Returns:
        `num_split` `Tensor` objects resulting from splitting `value`.
    """
    pass


def sqrt(x, name=None):
    """
    [TensorFlow Docs]
    Computes square root of x element-wise.

    I.e., \\(y = \sqrt{x} = x^{1/2}\\).

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def square(x, name=None):
    """
    [TensorFlow Docs]
    Computes square of x element-wise.

    I.e., \\(y = x * x = x^2\\).

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def squared_difference(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns (x - y)(x - y) element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def squeeze(input, squeeze_dims=None, name=None):
    """
    [TensorFlow Docs]
    Removes dimensions of size 1 from the shape of a tensor.

    Given a tensor `input`, this operation returns a tensor of the same type with
    all dimensions of size 1 removed. If you don't want to remove all size 1
    dimensions, you can remove specific size 1 dimensions by specifying
    `squeeze_dims`.

    For example:

    ```prettyprint
    # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    shape(squeeze(t)) ==> [2, 3]
    ```

    Or, to remove specific size 1 dimensions:

    ```prettyprint
    # 't' is a tensor of shape [1, 2, 1, 3, 1, 1]
    shape(squeeze(t, [2, 4])) ==> [1, 2, 3, 1]
    ```

    Args:
        input: A `Tensor`. The `input` to squeeze.
        squeeze_dims: An optional list of `ints`. Defaults to `[]`.
                      If specified, only squeezes the dimensions listed. The dimension
                      index starts at 0. It is an error to squeeze a dimension that is not 1.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        Contains the same data as `input`, but has one or more dimensions of
        size 1 removed.
    """
    pass


def stop_gradient(input, name=None):
    """
    [TensorFlow Docs]
    Stops gradient computation.

    When executed in a graph, this op outputs its input tensor as-is.

    When building ops to compute gradients, this op prevents the contribution of
    its inputs to be taken into account. Normally, the gradient generator adds ops
    to a graph to compute the derivatives of a specified 'loss' by recursively
    finding out inputs that contributed to its computation. If you insert this op
    in the graph it inputs are masked from the gradient generator. They are not
    taken into account for computing gradients.

    This is useful any time you want to compute a value with TensorFlow but need
    to pretend that the value was a constant. Some examples include:

    *  The *EM* algorithm where the *M-step* should not involve backpropagation
     through the output of the *E-step*.
    *  Contrastive divergence training of Boltzmann machines where, when
     differentiating the energy function, the training must not backpropagate
     through the graph that generated the samples from the model.
    *  Adversarial training, where no backprop should happen through the adversarial
     example generation process.

    Args:
        input: A `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def string_to_hash_bucket(string_tensor, num_buckets, name=None):
    """
    [TensorFlow Docs]
    Converts each string in the input Tensor to its hash mod by a number of buckets.

    The hash function is deterministic on the content of the string within the
    process.

    Note that the hash function may change from time to time.

    Args:
        string_tensor: A `Tensor` of type `string`.
        num_buckets: An `int` that is `>= 1`. The number of buckets.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
        A Tensor of the same shape as the input `string_tensor`.
    """
    pass


def string_to_hash_bucket_fast(input, num_buckets, name=None):
    """
    [TensorFlow Docs]
    Converts each string in the input Tensor to its hash mod by a number of buckets.

    The hash function is deterministic on the content of the string within the
    process and will never change. However, it is not suitable for cryptography.
    This function may be used when CPU time is scarce and inputs are trusted or
    unimportant. There is a risk of adversaries constructing inputs that all hash
    to the same bucket. To prevent this problem, use a strong hash function with
    `tf.string_to_hash_bucket_strong`.

    Args:
        input: A `Tensor` of type `string`. The strings to assign a hash bucket.
        num_buckets: An `int` that is `>= 1`. The number of buckets.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
        A Tensor of the same shape as the input `string_tensor`.
    """
    pass


def string_to_hash_bucket_strong(input, num_buckets, key, name=None):
    """
    [TensorFlow Docs]
    Converts each string in the input Tensor to its hash mod by a number of buckets.

    The hash function is deterministic on the content of the string within the
    process. The hash function is a keyed hash function, where attribute `key`
    defines the key of the hash function. `key` is an array of 2 elements.

    A strong hash is important when inputs may be malicious, e.g. URLs with
    additional components. Adversaries could try to make their inputs hash to the
    same bucket for a denial-of-service attack or to skew the results. A strong
    hash prevents this by making it dificult, if not infeasible, to compute inputs
    that hash to the same bucket. This comes at a cost of roughly 4x higher compute
    time than tf.string_to_hash_bucket_fast.

    Args:
        input: A `Tensor` of type `string`. The strings to assign a hash bucket.
        num_buckets: An `int` that is `>= 1`. The number of buckets.
        key: A list of `ints`.
             The key for the keyed hash function passed as a list of two uint64
             elements.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
        A Tensor of the same shape as the input `string_tensor`.
    """
    pass


def string_to_number(string_tensor, out_type=None, name=None):
    """
    [TensorFlow Docs]
    Converts each string in the input Tensor to the specified numeric type.

    (Note that int32 overflow results in an error while float overflow
    results in a rounded value.)

    Args:
        string_tensor: A `Tensor` of type `string`.
        out_type: An optional `tf.DType` from: `tf.float32, tf.int32`. Defaults to `tf.float32`.
                  The numeric type to interpret each string in string_tensor as.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `out_type`.
        A Tensor of the same shape as the input `string_tensor`.
    """
    pass


def sub(x, y, name=None):
    """
    [TensorFlow Docs]
    Returns x - y element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        y: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def tan(x, name=None):
    """
    [TensorFlow Docs]
    Computes tan of x element-wise.

    Args:
        x: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`, `int32`, `int64`, `complex64`, `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def tanh(x, name=None):
    """
    [TensorFlow Docs]
    Computes hyperbolic tangent of `x` element-wise.

    Args:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
           or `qint32`.
        name: A name for the operation (optional).

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
            the return type is `quint8`.
    """
    pass


def tile(input, multiples, name=None):
    """
    [TensorFlow Docs]
    Constructs a tensor by tiling a given tensor.

    This operation creates a new tensor by replicating `input` `multiples` times.
    The output tensor's i'th dimension has `input.dims(i) * multiples[i]` elements,
    and the values of `input` are replicated `multiples[i]` times along the 'i'th
    dimension. For example, tiling `[a b c d]` by `[2]` produces
    `[a b c d a b c d]`.

    Args:
        input: A `Tensor`. 1-D or higher.
        multiples: A `Tensor` of type `int32`.
                   1-D. Length must be the same as the number of dimensions in `input`
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def to_bfloat16(x, name="ToBFloat16"):
    """
    [TensorFlow Docs]
    Casts a tensor to type `bfloat16`.

    Args:
        x: A `Tensor` or `SparseTensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` or `SparseTensor` with same shape as `x` with type `bfloat16`.

    Raises:
        TypeError: If `x` cannot be cast to the `bfloat16`.
    """
    pass


def to_double(x, name="ToDouble"):
    """
    [TensorFlow Docs]
    Casts a tensor to type `float64`.

    Args:
        x: A `Tensor` or `SparseTensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` or `SparseTensor` with same shape as `x` with type `float64`.

    Raises:
        TypeError: If `x` cannot be cast to the `float64`.
    """
    pass


def to_float(x, name="ToFloat"):
    """
    [TensorFlow Docs]
    Casts a tensor to type `float32`.

    Args:
        x: A `Tensor` or `SparseTensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` or `SparseTensor` with same shape as `x` with type `float32`.

    Raises:
        TypeError: If `x` cannot be cast to the `float32`.
    """
    pass


def to_int32(x, name="ToInt32"):
    """
    [TensorFlow Docs]
    Casts a tensor to type `int32`.

    Args:
        x: A `Tensor` or `SparseTensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` or `SparseTensor` with same shape as `x` with type `int32`.

    Raises:
        TypeError: If `x` cannot be cast to the `int32`.
    """
    pass


def to_int64(x, name="ToInt64"):
    """
    [TensorFlow Docs]
    Casts a tensor to type `int64`.

    Args:
        x: A `Tensor` or `SparseTensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` or `SparseTensor` with same shape as `x` with type `int64`.

    Raises:
        TypeError: If `x` cannot be cast to the `int64`.
    """
    pass


def trace(x, name=None):
    """
    [TensorFlow Docs]
     Compute the trace of a tensor `x`.

    `trace(x)` returns the sum of along the diagonal.

    For example:

    ```python
    # 'x' is [[1, 1],
    #         [1, 1]]
    tf.trace(x) ==> 2

    # 'x' is [[1,2,3],
    #         [4,5,6],
    #         [7,8,9]]
    tf.trace(x) ==> 15
    ```

    Args:
        x: 2-D tensor.
        name: A name for the operation (optional).

    Returns:
        The trace of input tensor.
    """
    pass


def trainable_variables():
    """
    [TensorFlow Docs]
    Returns all variables created with `trainable=True`.

    When passed `trainable=True`, the `Variable()` constructor automatically
    adds new variables to the graph collection
    `GraphKeys.TRAINABLE_VARIABLES`. This convenience function returns the
    contents of that collection.

    Returns:
        A list of Variable objects.
    """
    pass


def transpose(a, perm=None, name="transpose"):
    """
    [TensorFlow Docs]
    Transposes `a`. Permutes the dimensions according to `perm`.

    The returned tensor's dimension i will correspond to the input dimension
    `perm[i]`. If `perm` is not given, it is set to (n-1...0), where n is
    the rank of the input tensor. Hence by default, this operation performs a
    regular matrix transpose on 2-D input Tensors.

    For example:

    ```python
    # 'x' is [[1 2 3]
    #         [4 5 6]]
    tf.transpose(x) ==> [[1 4]
                       [2 5]
                       [3 6]]

    # Equivalently
    tf.transpose(x, perm=[1, 0]) ==> [[1 4]
                                    [2 5]
                                    [3 6]]

    # 'perm' is more useful for n-dimensional tensors, for n > 2
    # 'x' is   [[[1  2  3]
    #            [4  5  6]]
    #           [[7  8  9]
    #            [10 11 12]]]
    # Take the transpose of the matrices in dimension-0
    tf.transpose(x, perm=[0, 2, 1]) ==> [[[1  4]
                                        [2  5]
                                        [3  6]]

                                       [[7 10]
                                        [8 11]
                                        [9 12]]]
    ```

    Args:
        a: A `Tensor`.
        perm: A permutation of the dimensions of `a`.
        name: A name for the operation (optional).

    Returns:
        A transposed `Tensor`.
    """
    pass


def truediv(x, y, name=None):
    """
    [TensorFlow Docs]
    Divides x / y elementwise, always producing floating point results.

    The same as `tf.div` for floating point arguments, but casts integer arguments
    to floating point before dividing so that the result is always floating point.
    This op is generated by normal `x / y` division in Python 3 and in Python 2.7
    with `from __future__ import division`. If you want integer division that
    rounds down, use `x // y` or `tf.floordiv`.

    `x` and `y` must have the same numeric type. If the inputs are floating
    point, the output will have the same type. If the inputs are integral, the
    inputs are cast to `float32` for `int8` and `int16` and `float64` for `int32`
    and `int64` (matching the behavior of Numpy).

    Args:
        x: `Tensor` numerator of numeric type.
        y: `Tensor` denominator of numeric type.
        name: A name for the operation (optional).

    Returns:
        `x / y` evaluated in floating point.

    Raises:
        TypeError: If `x` and `y` have different dtypes.
    """
    pass


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=dtypes.float32,
                     seed=None, name=None):
    """
    [TensorFlow Docs]
    Outputs random values from a truncated normal distribution.

    The generated values follow a normal distribution with specified mean and
    standard deviation, except that values whose magnitude is more than 2 standard
    deviations from the mean are dropped and re-picked.

    Args:
        shape: A 1-D integer Tensor or Python array. The shape of the output tensor.
        mean: A 0-D Tensor or Python value of type `dtype`. The mean of the
              truncated normal distribution.
              stddev: A 0-D Tensor or Python value of type `dtype`. The standard deviation
              of the truncated normal distribution.
        dtype: The type of the output.
        seed: A Python integer. Used to create a random seed for the distribution.
              See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        name: A name for the operation (optional).

    Returns:
        A tensor of the specified shape filled with random truncated normal values.
    """
    pass


def truncated_normal_initializer(mean=0.0, stddev=1.0, seed=None,
                                 dtype=dtypes.float32):
    """
    [TensorFlow Docs]
    Returns an initializer that generates a truncated normal distribution.

    These values are similar to values from a `random_normal_initializer`
    except that values more than two standard deviations from the mean
    are discarded and re-drawn. This is the recommended initializer for
    neural network weights and filters.

    Args:
        mean: a python scalar or a scalar tensor. Mean of the random values
              to generate.
              stddev: a python scalar or a scalar tensor. Standard deviation of the
              random values to generate.
        seed: A Python integer. Used to create random seeds. See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        dtype: The data type. Only floating point types are supported.

    Returns:
        An initializer that generates tensors with a truncated normal
        distribution.

    Raises:
        ValueError: if `dtype` is not a floating point type.
    """
    pass


def tuple(tensors, name=None, control_inputs=None):
    """
    [TensorFlow Docs]
    Group tensors together.

    This creates a tuple of tensors with the same values as the `tensors`
    argument, except that the value of each tensor is only returned after the
    values of all tensors have been computed.

    `control_inputs` contains additional ops that have to finish before this op
    finishes, but whose outputs are not returned.

    This can be used as a "join" mechanism for parallel computations: all the
    argument tensors can be computed in parallel, but the values of any tensor
    returned by `tuple` are only available after all the parallel computations
    are done.

    See also `group` and `with_dependencies`.

    Args:
        tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
        name: (optional) A name to use as a `name_scope` for the operation.
        control_inputs: List of additional ops to finish before returning.

    Returns:
        Same as `tensors`.

    Raises:
        ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
        TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
            objects.

    """
    pass


def uniform_unit_scaling_initializer(factor=1.0, seed=None,
                                     dtype=dtypes.float32, full_shape=None):
    """
    [TensorFlow Docs]
    Returns an initializer that generates tensors without scaling variance.

    When initializing a deep network, it is in principle advantageous to keep
    the scale of the input variance constant, so it does not explode or diminish
    by reaching the final layer. If the input is `x` and the operation `x * W`,
    and we want to initialize `W` uniformly at random, we need to pick `W` from

            [-sqrt(3) / sqrt(dim), sqrt(3) / sqrt(dim)]

    to keep the scale intact, where `dim = W.shape[0]` (the size of the input).
    A similar calculation for convolutional networks gives an analogous result
    with `dim` equal to the product of the first 3 dimensions. When
    nonlinearities are present, we need to multiply this by a constant `factor`.
    See [Sussillo et al., 2014](https://arxiv.org/abs/1412.6558)
    ([pdf](http://arxiv.org/pdf/1412.6558.pdf)) for deeper motivation, experiments
    and the calculation of constants. In section 2.3 there, the constants were
    numerically computed: for a linear layer it's 1.0, relu: ~1.43, tanh: ~1.15.

    If the shape tuple `full_shape` is provided, the scale will be calculated from
    this predefined shape. This is useful when a `Variable` is being partitioned
    across several shards, and each shard has a smaller shape than the whole.
    Since the shards are usually concatenated when used, the scale should be
    based on the shape of the whole.

    Args:
        factor: Float. A multiplicative factor by which the values will be scaled.
        seed: A Python integer. Used to create random seeds. See
              [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
              for behavior.
        dtype: The data type. Only floating point types are supported.
               full_shape: Tuple or list of integers. The shape used for calculating
               scale normalization (instead of the shape passed at creation time).
               Useful when creating sharded variables via partitioning.

    Returns:
        An initializer that generates tensors with unit variance.

    Raises:
        ValueError: if `dtype` is not a floating point type.
    """
    pass


def unique(x, name=None):
    """
    [TensorFlow Docs]
    Finds unique elements in a 1-D tensor.

    This operation returns a tensor `y` containing all of the unique elements of `x`
    sorted in the same order that they occur in `x`. This operation also returns a
    tensor `idx` the same size as `x` that contains the index of each value of `x`
    in the unique output `y`. In other words:

    `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

    For example:

    ```prettyprint
    # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    y, idx = unique(x)
    y ==> [1, 2, 4, 7, 8]
    idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
    ```

    Args:
        x: A `Tensor`. 1-D.
        name: A name for the operation (optional).

    Returns:
        A tuple of `Tensor` objects (y, idx).
        y: A `Tensor`. Has the same type as `x`. 1-D.
           idx: A `Tensor` of type `int32`. 1-D.
           """
    pass


def unique_with_counts(x, name=None):
    """
    [TensorFlow Docs]
    Finds unique elements in a 1-D tensor.

    This operation returns a tensor `y` containing all of the unique elements of `x`
    sorted in the same order that they occur in `x`. This operation also returns a
    tensor `idx` the same size as `x` that contains the index of each value of `x`
    in the unique output `y`. Finally, it returns a third tensor `count` that
    contains the count of each element of `y` in `x`. In other words:

    `y[idx[i]] = x[i] for i in [0, 1,...,rank(x) - 1]`

    For example:

    ```prettyprint
    # tensor 'x' is [1, 1, 2, 4, 4, 4, 7, 8, 8]
    y, idx, count = unique_with_counts(x)
    y ==> [1, 2, 4, 7, 8]
    idx ==> [0, 0, 1, 2, 2, 2, 3, 4, 4]
    count ==> [2, 1, 3, 1, 2]
    ```

    Args:
        x: A `Tensor`. 1-D.
        name: A name for the operation (optional).

    Returns:
        A tuple of `Tensor` objects (y, idx, count).
        y: A `Tensor`. Has the same type as `x`. 1-D.
           idx: A `Tensor` of type `int32`. 1-D.
           count: A `Tensor` of type `int32`. 1-D.
           """
    pass


def unpack(value, num=None, name="unpack"):
    """
    [TensorFlow Docs]
    Unpacks the outer dimension of a rank-`R` tensor into rank-`(R-1)` tensors.

    Unpacks `num` tensors from `value` along the first dimension.
    If `num` is not specified (the default), it is inferred from `value`'s shape.
    If `value.shape[0]` is not known, `ValueError` is raised.

    The ith tensor in `output` is the slice `value[i, ...]`. Each tensor in
    `output` has shape `value.shape[1:]`.

    This is the opposite of pack. The numpy equivalent is

            tf.unpack(x, n) = list(x)

    Args:
        value: A rank `R > 0` `Tensor` to be unpacked.
        num: An `int`. The first dimension of value. Automatically inferred if
             `None` (the default).
        name: A name for the operation (optional).

    Returns:
        The list of `Tensor` objects unpacked from `value`.

    Raises:
        ValueError: If `num` is unspecified and cannot be inferred.
    """
    pass


def unsorted_segment_sum(data, segment_ids, num_segments, name=None):
    """
    [TensorFlow Docs]
    Computes the sum along segments of a tensor.

    Read [the section on
    Segmentation](../../api_docs/python/math_ops.md#segmentation) for an explanation
    of segments.

    Computes a tensor such that
    \\(output_i = \sum_j data_j\\) where sum is over `j` such
    that `segment_ids[j] == i`. Unlike `SegmentSum`, `segment_ids`
    need not be sorted and need not cover all values in the full
        range of valid values.

    If the sum is empty for a given segment ID `i`, `output[i] = 0`.

    `num_segments` should equal the number of distinct segment IDs.

    <div style="width:70%; margin:auto; margin-bottom:10px; margin-top:20px;">
    <img style="width:100%" src="../../images/UnsortedSegmentSum.png" alt>
    </div>

    Args:
        data: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
        segment_ids: A `Tensor`. Must be one of the following types: `int32`, `int64`.
                     A 1-D tensor whose rank is equal to the rank of `data`'s
                     first dimension.
        num_segments: A `Tensor` of type `int32`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `data`.
        Has same shape as data, except for dimension 0 which
        has size `num_segments`.
    """
    pass


def variable_axis_size_partitioner(
        max_shard_bytes, axis=0, bytes_per_string_element=16, max_shards=None):
    """
    [TensorFlow Docs]
    Get a partitioner for VariableScope to keep shards below `max_shard_bytes`.

    This partitioner will shard a Variable along one axis, attempting to keep
    the maximum shard size below `max_shard_bytes`. In practice, this is not
    always possible when sharding along only one axis. When this happens,
    this axis is sharded as much as possible (i.e., every dimension becomes
    a separate shard).

    If the partitioner hits the `max_shards` limit, then each shard may end up
    larger than `max_shard_bytes`. By default `max_shards` equals `None` and no
    limit on the number of shards is enforced.

    One reasonable value for `max_shard_bytes` is `(64 << 20) - 1`, or almost
    `64MB`, to keep below the protobuf byte limit.

    Args:
        max_shard_bytes: The maximum size any given shard is allowed to be.
        axis: The axis to partition along. Default: outermost axis.
              bytes_per_string_element: If the `Variable` is of type string, this provides
              an estimate of how large each scalar in the `Variable` is.
              max_shards: The maximum number of shards in int created taking precedence
              over `max_shard_bytes`.

    Returns:
        A partition function usable as the `partitioner` argument to
        `variable_scope`, `get_variable`, and `get_partitioned_variable_list`.

    Raises:
        ValueError: If any of the byte counts are non-positive.
    """
    pass

    @wraps(func)
    def helper(*args, **kwds):
        return GeneratorContextManager(func(*args, **kwds))


pass


@wraps(func)
def helper(*args, **kwds):
    return GeneratorContextManager(func(*args, **kwds))


pass


def verify_tensor_all_finite(t, msg, name=None):
    """
    [TensorFlow Docs]
    Assert that the tensor does not contain any NaN's or Inf's.

    Args:
        t: Tensor to check.
        msg: Message to log on failure.
        name: A name for this operation (optional).

    Returns:
        Same tensor as `t`.
    """
    pass


def where(input, name=None):
    """
    [TensorFlow Docs]
    Returns locations of true values in a boolean tensor.

    This operation returns the coordinates of true elements in `input`. The
    coordinates are returned in a 2-D tensor where the first dimension (rows)
    represents the number of true elements, and the second dimension (columns)
    represents the coordinates of the true elements. Keep in mind, the shape of
    the output tensor can vary depending on how many true values there are in
    `input`. Indices are output in row-major order.

    For example:

    ```prettyprint
    # 'input' tensor is [[True, False]
    #                    [True, False]]
    # 'input' has two true values, so output has two coordinates.
    # 'input' has rank of 2, so coordinates have two indices.
    where(input) ==> [[0, 0],
                    [1, 0]]

    # `input` tensor is [[[True, False]
    #                     [True, False]]
    #                    [[False, True]
    #                     [False, True]]
    #                    [[False, False]
    #                     [False, True]]]
    # 'input' has 5 true values, so output has 5 coordinates.
    # 'input' has rank of 3, so coordinates have three indices.
    where(input) ==> [[0, 0, 0],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [2, 1, 1]]
    ```

    Args:
        input: A `Tensor` of type `bool`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `int64`.
    """
    pass


def while_loop(cond, body, loop_vars, parallel_iterations=10, back_prop=True,
               swap_memory=False, name=None):
    """
    [TensorFlow Docs]
    Repeat `body` while the condition `cond` is true.

    `cond` is a callable returning a boolean scalar tensor. `body` is a callable
    returning a list of tensors of the same length and with the same types as
    `loop_vars`. `loop_vars` is a list of tensors that is passed to both `cond`
    and `body`. `cond` and `body` both take as many arguments as there are
    `loop_vars`.

    In addition to regular Tensors or IndexedSlices, the body may accept and
    return TensorArray objects. The flows of the TensorArray objects will
    be appropriately forwarded between loops and during gradient calculations.

    While `cond` evaluates to true, `body` is executed.

    `while_loop` implements non-strict semantics, enabling multiple iterations
    to run in parallel. The maximum number of parallel iterations can be
    controlled by `parallel_iterations`, which gives users some control over
    memory consumption and execution order. For correct programs, `while_loop`
    should return the same result for any parallel_iterations > 0.

    For training, TensorFlow remembers the tensors that are produced in the
    forward inference but needed in back propagation. These tensors can be a
    main source of memory consumption and often cause OOM problems when training
    on GPUs. When the flag swap_memory is true, we swap out these tensors from
    GPU to CPU. This for example allows us to train RNN models with very long
    sequences and large batches.

    Args:
        cond: A callable that represents the termination condition of the loop.
        body: A callable that represents the loop body.
        loop_vars: The list of variable input tensors.
        parallel_iterations: The number of iterations allowed to run in parallel.
                             back_prop: Whether backprop is enabled for this while loop.
                             swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
        name: Optional name prefix for the returned tensors.

    Returns:
        The output tensors for the loop variables after the loop.

    Raises:
        TypeError: if `cond` or `body` is not callable.
        ValueError: if `loop_var` is empty.

    Example:

        ```python
        i = tf.constant(0)
        c = lambda i: tf.less(i, 10)
        b = lambda i: tf.add(i, 1)
        r = tf.while_loop(c, b, [i])
        ```

    """
    pass


def zeros(shape, dtype=dtypes.float32, name=None):
    """
    [TensorFlow Docs]
    Creates a tensor with all elements set to zero.

    This operation returns a tensor of type `dtype` with shape `shape` and
    all elements set to zero.

    For example:

    ```python
    tf.zeros([3, 4], int32) ==> [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    ```

    Args:
        shape: Either a list of integers, or a 1-D `Tensor` of type `int32`.
        dtype: The type of an element in the resulting `Tensor`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with all elements set to zero.
    """
    pass


def zeros_initializer(shape, dtype=dtypes.float32):
    """
    [TensorFlow Docs]
    An adaptor for zeros() to match the Initializer spec."""
    pass


def zeros_like(tensor, dtype=None, name=None):
    """
    [TensorFlow Docs]
    Creates a tensor with all elements set to zero.

    Given a single tensor (`tensor`), this operation returns a tensor of the
    same type and shape as `tensor` with all elements set to zero. Optionally,
    you can use `dtype` to specify a new type for the returned tensor.

    For example:

    ```python
    # 'tensor' is [[1, 2, 3], [4, 5, 6]]
    tf.zeros_like(tensor) ==> [[0, 0, 0], [0, 0, 0]]
    ```

    Args:
        tensor: A `Tensor`.
        dtype: A type for the returned `Tensor`. Must be `float32`, `float64`,
               `int8`, `int16`, `int32`, `int64`, `uint8`, `complex64`, or `complex128`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with all elements set to zero.
    """
    pass


def zeta(x, q, name=None):
    """
    [TensorFlow Docs]
    Compute the Hurwitz zeta function \\(\zeta(x, q)\\).

    The Hurwitz zeta function is defined as:

    ```
    \zeta(x, q) = \sum_{n=0}^{\infty} (q + n)^{-x}
    ```

    Args:
        x: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        q: A `Tensor`. Must have the same type as `x`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `x`.
    """
    pass


def _compute_sampled_logits(weights, biases, inputs, labels, num_sampled,
                            num_classes, num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None):
    """
    [TensorFlow Docs]
    Helper function for nce_loss and sampled_softmax_loss functions.

    Computes sampled output training logits and labels suitable for implementing
    e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
    sampled_softmax_loss).

    Note: In the case where num_true > 1, we assign to each target class
    the target probability 1 / num_true so that the target probabilities
    sum to 1 per-example.

    Args:
        weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
                objects whose concatenation along dimension 0 has shape
                `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
        biases: A `Tensor` of shape `[num_classes]`.  The class biases.
        inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
                activations of the input network.
        labels: A `Tensor` of type `int64` and shape `[batch_size,
                num_true]`. The target classes.  Note that this format differs from
                the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
        num_sampled: An `int`.  The number of classes to randomly sample per batch.
        num_classes: An `int`. The number of possible classes.
        num_true: An `int`.  The number of target classes per training example.
        sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
                `sampled_expected_count`) returned by a `*_candidate_sampler` function.
                (if None, we default to `log_uniform_candidate_sampler`)
        subtract_log_q: A `bool`.  whether to subtract the log expected count of
                the labels in the sample to get the logits of the true labels.
                Default is True.  Turn off for Negative Sampling.
        remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
                where a sampled class equals one of the target classes.  Default is
                False.
        partition_strategy: A string specifying the partitioning strategy, relevant
                if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
                Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
        name: A name for the operation (optional).
    Returns:
        out_logits, out_labels: `Tensor` objects each with shape
                `[batch_size, num_true + num_sampled]`, for passing to either
                `nn.sigmoid_cross_entropy_with_logits` (NCE) or
                `nn.softmax_cross_entropy_with_logits` (sampled softmax).
    """
    pass


def _sum_rows(x):
    """
    [TensorFlow Docs]
    Returns a vector summing up each row of the matrix x."""
    pass


def all_candidate_sampler(true_classes, num_true, num_sampled, unique,
                          seed=None, name=None):
    """
    [TensorFlow Docs]
    Generate the set of all classes.

    Deterministically generates and returns the set of all possible classes.
    For testing purposes.  There is no need to use this, since you might as
    well use full softmax or full logistic regression.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size,
            num_true]`. The target classes.
        num_true: An `int`.  The number of target classes per training example.
        num_sampled: An `int`.  The number of possible classes.
        unique: A `bool`. Ignored.
            unique.
        seed: An `int`. An operation-specific seed. Default is 0.
        name: A name for the operation (optional).

    Returns:
        sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
            This operation deterministically returns the entire range
            `[0, num_sampled]`.
        true_expected_count: A tensor of type `float`.  Same shape as
            `true_classes`. The expected counts under the sampling distribution
            of each of `true_classes`. All returned values are 1.0.
        sampled_expected_count: A tensor of type `float`. Same shape as
            `sampled_candidates`. The expected counts under the sampling distribution
            of each of `sampled_candidates`. All returned values are 1.0.
    """
    pass


"""## Casting

TensorFlow provides several operations that you can use to cast tensor data
types in your graph.

@@string_to_number
@@to_double
@@to_float
@@to_bfloat16
@@to_int32
@@to_int64
@@cast
@@saturate_cast


TensorFlow provides several operations that you can use to determine the shape
of a tensor and change the shape of a tensor.

@@shape
@@size
@@rank
@@reshape
@@squeeze
@@expand_dims


TensorFlow provides several operations to slice or extract parts of a tensor,
or join multiple tensors together.

@@slice
@@split
@@tile
@@pad
@@concat
@@pack
@@unpack
@@reverse_sequence
@@reverse
@@transpose
@@extract_image_patches
@@space_to_batch
@@batch_to_space
@@space_to_depth
@@depth_to_space
@@gather
@@gather_nd
@@dynamic_partition
@@dynamic_stitch
@@boolean_mask
@@one_hot

"""
pass


def atrous_conv2d(value, filters, rate, padding, name=None):
    """
    [TensorFlow Docs]
    Atrous convolution (a.k.a. convolution with holes or dilated convolution).

    Computes a 2-D atrous convolution, also known as convolution with holes or
    dilated convolution, given 4-D `value` and `filters` tensors. If the `rate`
    parameter is equal to one, it performs regular 2-D convolution. If the `rate`
    parameter is greater than one, it performs convolution with holes, sampling
    the input values every `rate` pixels in the `height` and `width` dimensions.
    This is equivalent to convolving the input with a set of upsampled filters,
    produced by inserting `rate - 1` zeros between two consecutive values of the
    filters along the `height` and `width` dimensions, hence the name atrous
    convolution or convolution with holes (the French word trous means holes in
    English).

    More specifically:

            output[b, i, j, k] = sum_{di, dj, q} filters[di, dj, q, k] *
            value[b, i + rate * di, j + rate * dj, q]

    Atrous convolution allows us to explicitly control how densely to compute
    feature responses in fully convolutional networks. Used in conjunction with
    bilinear interpolation, it offers an alternative to `conv2d_transpose` in
    dense prediction tasks such as semantic image segmentation, optical flow
    computation, or depth estimation. It also allows us to effectively enlarge
    the field of view of filters without increasing the number of parameters or
    the amount of computation.

    For a description of atrous convolution and how it can be used for dense
    feature extraction, please see: [Semantic Image Segmentation with Deep
    Convolutional Nets and Fully Connected CRFs](http://arxiv.org/abs/1412.7062).
    The same operation is investigated further in [Multi-Scale Context Aggregation
    by Dilated Convolutions](http://arxiv.org/abs/1511.07122). Previous works
    that effectively use atrous convolution in different ways are, among others,
    [OverFeat: Integrated Recognition, Localization and Detection using
    Convolutional Networks](http://arxiv.org/abs/1312.6229) and [Fast Image
    Scanning with Deep Max-Pooling Convolutional Neural Networks]
    (http://arxiv.org/abs/1302.1700). Atrous convolution is also closely related
    to the so-called noble identities in multi-rate signal processing.

    There are many different ways to implement atrous convolution (see the refs
    above). The implementation here reduces

            atrous_conv2d(value, filters, rate, padding=padding)

    to the following three operations:

            paddings = ...
            net = space_to_batch(value, paddings, block_size=rate)
            net = conv2d(net, filters, strides=[1, 1, 1, 1], padding="VALID")
            crops = ...
            net = batch_to_space(net, crops, block_size=rate)

    Advanced usage. Note the following optimization: A sequence of `atrous_conv2d`
    operations with identical `rate` parameters, 'SAME' `padding`, and filters
    with odd heights/ widths:

            net = atrous_conv2d(net, filters1, rate, padding="SAME")
            net = atrous_conv2d(net, filters2, rate, padding="SAME")
            ...
            net = atrous_conv2d(net, filtersK, rate, padding="SAME")

    can be equivalently performed cheaper in terms of computation and memory as:

            pad = ...  # padding so that the input dims are multiples of rate
            net = space_to_batch(net, paddings=pad, block_size=rate)
            net = conv2d(net, filters1, strides=[1, 1, 1, 1], padding="SAME")
            net = conv2d(net, filters2, strides=[1, 1, 1, 1], padding="SAME")
            ...
            net = conv2d(net, filtersK, strides=[1, 1, 1, 1], padding="SAME")
            net = batch_to_space(net, crops=pad, block_size=rate)

    because a pair of consecutive `space_to_batch` and `batch_to_space` ops with
    the same `block_size` cancel out when their respective `paddings` and `crops`
    inputs are identical.

    Args:
        value: A 4-D `Tensor` of type `float`. It needs to be in the default "NHWC"
            format. Its shape is `[batch, in_height, in_width, in_channels]`.
        filters: A 4-D `Tensor` with the same type as `value` and shape
            `[filter_height, filter_width, in_channels, out_channels]`. `filters`'
            `in_channels` dimension must match that of `value`. Atrous convolution is
            equivalent to standard convolution with upsampled filters with effective
            height `filter_height + (filter_height - 1) * (rate - 1)` and effective
            width `filter_width + (filter_width - 1) * (rate - 1)`, produced by
            inserting `rate - 1` zeros along consecutive elements across the
            `filters`' spatial dimensions.
        rate: A positive int32. The stride with which we sample input values across
            the `height` and `width` dimensions. Equivalently, the rate by which we
            upsample the filter values by inserting zeros across the `height` and
            `width` dimensions. In the literature, the same parameter is sometimes
            called `input stride` or `dilation`.
        padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
        name: Optional name for the returned tensor.

    Returns:
        A `Tensor` with the same type as `value`.

    Raises:
        ValueError: If input/output depth does not match `filters`' shape, or if
            padding is other than `'VALID'` or `'SAME'`.
    """
    pass


def avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
    """
    [TensorFlow Docs]
    Performs the average pooling on the input.

    Each entry in `output` is the mean of the corresponding size `ksize`
    window in `value`.

    Args:
        value: A 4-D `Tensor` of shape `[batch, height, width, channels]` and type
            `float32`, `float64`, `qint8`, `quint8`, or `qint32`.
        ksize: A list of ints that has length >= 4.
            The size of the window for each dimension of the input tensor.
        strides: A list of ints that has length >= 4.
            The stride of the sliding window for each dimension of the
            input tensor.
        padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
            See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
        data_format: A string. 'NHWC' and 'NCHW' are supported.
        name: Optional name for the operation.

    Returns:
        A `Tensor` with the same type as `value`.  The average pooled output tensor.
    """
    pass


def avg_pool3d(input, ksize, strides, padding, name=None):
    """
    [TensorFlow Docs]
    Performs 3D average pooling on the input.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
        ksize: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The size of the window for each dimension of
            the input tensor. Must have `ksize[0] = ksize[1] = 1`.
        strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        The average pooled output tensor.
    """
    pass


def avg_pool3d_grad(orig_input_shape, grad, ksize, strides, padding,
                    name=None):
    """
    [TensorFlow Docs]
    Computes gradients of average pooling function.

    Args:
        orig_input_shape: A `Tensor` of type `int32`.
            The original input dimensions.
        grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Output backprop of shape `[batch, depth, rows, cols, channels]`.
        ksize: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The size of the window for each dimension of
            the input tensor. Must have `ksize[0] = ksize[1] = 1`.
        strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `grad`. The backprop for input.
    """
    pass


def batch_norm_with_global_normalization(t,
                                         m,
                                         v,
                                         beta,
                                         gamma,
                                         variance_epsilon,
                                         scale_after_normalization,
                                         name=None):
    """
    [TensorFlow Docs]
    Batch normalization.

    This op is deprecated. See `tf.nn.batch_normalization`.

    Args:
        t: A 4D input Tensor.
        m: A 1D mean Tensor with size matching the last dimension of t.
            This is the first output from tf.nn.moments,
            or a saved moving average thereof.
        v: A 1D variance Tensor with size matching the last dimension of t.
            This is the second output from tf.nn.moments,
            or a saved moving average thereof.
        beta: A 1D beta Tensor with size matching the last dimension of t.
            An offset to be added to the normalized tensor.
        gamma: A 1D gamma Tensor with size matching the last dimension of t.
            If "scale_after_normalization" is true, this tensor will be multiplied
            with the normalized tensor.
        variance_epsilon: A small float number to avoid dividing by 0.
        scale_after_normalization: A bool indicating whether the resulted tensor
            needs to be multiplied with gamma.
        name: A name for this operation (optional).

   Returns:
     A batch-normalized `t`.
    """
    pass


def batch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None):
    """
    [TensorFlow Docs]
    Batch normalization.

    As described in http://arxiv.org/abs/1502.03167.
    Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
    `scale` \\\\(\gamma\\\\) to it, as well as an `offset` \\\\(\\beta\\\\):

    \\\\(\\frac{\gamma(x-\mu)}{\sigma}+\\beta\\\\)

    `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
    shapes:
        * In all generality, they can have the same number of dimensions as the
            input `x`, with identical sizes as `x` for the dimensions that are not
            normalized over (the 'depth' dimension(s)), and dimension 1 for the
            others which are being normalized over.
            `mean` and `variance` in this case would typically be the outputs of
            `tf.nn.moments(..., keep_dims=True)` during training, or running averages
            thereof during inference.
        * In the common case where the 'depth' dimension is the last dimension in
            the input tensor `x`, they may be one dimensional tensors of the same
            size as the 'depth' dimension.
            This is the case for example for the common `[batch, depth]` layout of
            fully-connected layers, and `[batch, height, width, depth]` for
            convolutions.
            `mean` and `variance` in this case would typically be the outputs of
            `tf.nn.moments(..., keep_dims=False)` during training, or running averages
            thereof during inference.

    Args:
        x: Input `Tensor` of arbitrary dimensionality.
        mean: A mean `Tensor`.
        variance: A variance `Tensor`.
        offset: An offset `Tensor`, often denoted \\\\(\\beta\\\\) in equations, or
            None. If present, will be added to the normalized tensor.
        scale: A scale `Tensor`, often denoted \\\\(\gamma\\\\) in equations, or
            `None`. If present, the scale is applied to the normalized tensor.
        variance_epsilon: A small float number to avoid dividing by 0.
        name: A name for this operation (optional).

    Returns:
        the normalized, scaled, offset tensor.
    """
    pass


def bias_add(value, bias, data_format=None, name=None):
    """
    [TensorFlow Docs]
    Adds `bias` to `value`.

    This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
    Broadcasting is supported, so `value` may have any number of dimensions.
    Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
    case where both types are quantized.

    Args:
        value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
            `int16`, `int8`, `complex64`, or `complex128`.
        bias: A 1-D `Tensor` with size matching the last dimension of `value`.
            Must be the same type as `value` unless `value` is a quantized type,
            in which case a different quantized type may be used.
        data_format: A string. 'NHWC' and 'NCHW' are supported.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with the same type as `value`.
    """
    pass


def bias_add_grad(out_backprop, data_format=None, name=None):
    """
    [TensorFlow Docs]
    The backward operation for "BiasAdd" on the "bias" tensor.

    It accumulates all the values from out_backprop into the feature dimension.
    For NHWC data format, the feature dimension is the last. For NCHW data format,
    the feature dimension is the third-to-last.

    Args:
        out_backprop: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Any number of dimensions.
        data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
            Specify the data format of the input and output data. With the
            default format "NHWC", the bias tensor will be added to the last dimension
            of the value tensor.
            Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
            The tensor will be added to "in_channels", the third-to-the-last
          dimension.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `out_backprop`.
        1-D with size the feature dimension of `out_backprop`.
    """
    pass


def bias_add_v1(value, bias, name=None):
    """
    [TensorFlow Docs]
    Adds `bias` to `value`.

    This is a deprecated version of bias_add and will soon to be removed.

    This is (mostly) a special case of `tf.add` where `bias` is restricted to 1-D.
    Broadcasting is supported, so `value` may have any number of dimensions.
    Unlike `tf.add`, the type of `bias` is allowed to differ from `value` in the
    case where both types are quantized.

    Args:
        value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`,
            `int16`, `int8`, `complex64`, or `complex128`.
        bias: A 1-D `Tensor` with size matching the last dimension of `value`.
            Must be the same type as `value` unless `value` is a quantized type,
            in which case a different quantized type may be used.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with the same type as `value`.
    """
    pass


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
    """
    [TensorFlow Docs]
    Creates a bidirectional recurrent neural network.

    Similar to the unidirectional case above (rnn) but takes input and builds
    independent forward and backward RNNs with the final forward and backward
    outputs depth-concatenated, such that the output will have the format
    [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
    forward and backward cell must match. The initial state for both directions
    is zero by default (but can be set optionally) and no intermediate states are
    ever returned -- the network is fully unrolled for the given (passed in)
    length(s) of the sequence(s) or completely unrolled if length(s) is not given.

    Args:
        cell_fw: An instance of RNNCell, to be used for forward direction.
        cell_bw: An instance of RNNCell, to be used for backward direction.
        inputs: A length T list of inputs, each a tensor of shape
            [batch_size, input_size].
        initial_state_fw: (optional) An initial state for the forward RNN.
            This must be a tensor of appropriate type and shape
            `[batch_size x cell_fw.state_size]`.
            If `cell_fw.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
        initial_state_bw: (optional) Same as for `initial_state_fw`, but using
            the corresponding properties of `cell_bw`.
        dtype: (optional) The data type for the initial state.  Required if
            either of the initial states are not provided.
        sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
            containing the actual lengths for each of the sequences.
        scope: VariableScope for the created subgraph; defaults to "BiRNN"

    Returns:
        A tuple (outputs, output_state_fw, output_state_bw) where:
            outputs is a length `T` list of outputs (one for each input), which
                are depth-concatenated forward and backward outputs.
            output_state_fw is the final state of the forward rnn.
            output_state_bw is the final state of the backward rnn.

    Raises:
        TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
        ValueError: If inputs is None or an empty list.
    """
    pass


def compute_accidental_hits(true_classes, sampled_candidates, num_true,
                            seed=None, name=None):
    """
    [TensorFlow Docs]
    Compute the position ids in `sampled_candidates` matching `true_classes`.

    In Candidate Sampling, this operation facilitates virtually removing
    sampled classes which happen to match target classes.  This is done
    in Sampled Softmax and Sampled Logistic.

    See our [Candidate Sampling Algorithms
    Reference](http://www.tensorflow.org/extras/candidate_sampling.pdf).

    We presuppose that the `sampled_candidates` are unique.

    We call it an 'accidental hit' when one of the target classes
    matches one of the sampled classes.  This operation reports
    accidental hits as triples `(index, id, weight)`, where `index`
    represents the row number in `true_classes`, `id` represents the
    position in `sampled_candidates`, and weight is `-FLOAT_MAX`.

    The result of this op should be passed through a `sparse_to_dense`
    operation, then added to the logits of the sampled classes. This
    removes the contradictory effect of accidentally sampling the true
    target classes as noise classes for the same example.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size,
            num_true]`. The target classes.
        sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
            The sampled_candidates output of CandidateSampler.
        num_true: An `int`.  The number of target classes per training example.
        seed: An `int`. An operation-specific seed. Default is 0.
        name: A name for the operation (optional).

    Returns:
        indices: A `Tensor` of type `int32` and shape `[num_accidental_hits]`.
            Values indicate rows in `true_classes`.
        ids: A `Tensor` of type `int64` and shape `[num_accidental_hits]`.
            Values indicate positions in `sampled_candidates`.
        weights: A `Tensor` of type `float` and shape `[num_accidental_hits]`.
            Each value is `-FLOAT_MAX`.

    """
    pass


def conv1d(value, filters, stride, padding,
           use_cudnn_on_gpu=None, data_format=None,
           name=None):
    """
    [TensorFlow Docs]
    Computes a 1-D convolution given 3-D input and filter tensors.

    Given an input tensor of shape [batch, in_width, in_channels]
    and a filter / kernel tensor of shape
    [filter_width, in_channels, out_channels], this op reshapes
    the arguments to pass them to conv2d to perform the equivalent
    convolution operation.

    Internally, this op reshapes the input tensors and invokes
    `tf.nn.conv2d`.  A tensor of shape [batch, in_width, in_channels]
    is reshaped to [batch, 1, in_width, in_channels], and the filter
    is reshaped to [1, filter_width, in_channels, out_channels].
    The result is then reshaped back to [batch, out_width, out_channels]
    (where out_width is a function of the stride and padding as in
    conv2d) and returned to the caller.

    Args:
        value: A 3D `Tensor`.  Must be of type `float32` or `float64`.
        filters: A 3D `Tensor`.  Must have the same type as `input`.
        stride: An `integer`.  The number of entries by which
            the filter is moved right at each step.
        padding: 'SAME' or 'VALID'
        use_cudnn_on_gpu: An optional `bool`.  Defaults to `True`.
        data_format: An optional `string` from `"NHWC", "NCHW"`.  Defaults
            to `"NHWC"`, the data is stored in the order of
            [batch, in_width, in_channels].  The `"NCHW"` format stores
            data as [batch, in_channels, in_width].
        name: A name for the operation (optional).

    Returns:
        A `Tensor`.  Has the same type as input.
    """
    pass


def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None,
           data_format=None, name=None):
    """
    [TensorFlow Docs]
    Computes a 2-D convolution given 4-D `input` and `filter` tensors.

    Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
    and a filter / kernel tensor of shape
    `[filter_height, filter_width, in_channels, out_channels]`, this op
    performs the following:

    1. Flattens the filter to a 2-D matrix with shape
     `[filter_height * filter_width * in_channels, output_channels]`.
    2. Extracts image patches from the input tensor to form a *virtual*
     tensor of shape `[batch, out_height, out_width,
     filter_height * filter_width * in_channels]`.
    3. For each patch, right-multiplies the filter matrix and the image patch
     vector.

    In detail, with the default NHWC format,

            output[b, i, j, k] =
          sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *
                          filter[di, dj, q, k]

    Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

    Args:
        input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
        filter: A `Tensor`. Must have the same type as `input`.
        strides: A list of `ints`.
            1-D of length 4.  The stride of the sliding window for each dimension
            of `input`. Must be in the same order as the dimension specified with format.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
        data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
            Specify the data format of the input and output data. With the
            default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
            Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def conv2d_backprop_filter(input, filter_sizes, out_backprop, strides,
                           padding, use_cudnn_on_gpu=None, data_format=None,
                           name=None):
    """
    [TensorFlow Docs]
    Computes the gradients of convolution with respect to the filter.

    Args:
        input: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
            4-D with shape `[batch, in_height, in_width, in_channels]`.
        filter_sizes: A `Tensor` of type `int32`.
            An integer vector representing the tensor shape of `filter`,
            where `filter` is a 4-D
            `[filter_height, filter_width, in_channels, out_channels]` tensor.
        out_backprop: A `Tensor`. Must have the same type as `input`.
            4-D with shape `[batch, out_height, out_width, out_channels]`.
            Gradients w.r.t. the output of the convolution.
        strides: A list of `ints`.
            The stride of the sliding window for each dimension of the input
            of the convolution. Must be in the same order as the dimension specified with
            format.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
        data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
            Specify the data format of the input and output data. With the
            default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
            Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. 4-D with shape
        `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
        the `filter` input of the convolution.
    """
    pass


def conv2d_backprop_input(input_sizes, filter, out_backprop, strides, padding,
                          use_cudnn_on_gpu=None, data_format=None, name=None):
    """
    [TensorFlow Docs]
    Computes the gradients of convolution with respect to the input.

    Args:
        input_sizes: A `Tensor` of type `int32`.
            An integer vector representing the shape of `input`,
            where `input` is a 4-D `[batch, height, width, channels]` tensor.
        filter: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
            4-D with shape
            `[filter_height, filter_width, in_channels, out_channels]`.
        out_backprop: A `Tensor`. Must have the same type as `filter`.
            4-D with shape `[batch, out_height, out_width, out_channels]`.
            Gradients w.r.t. the output of the convolution.
        strides: A list of `ints`.
            The stride of the sliding window for each dimension of the input
            of the convolution. Must be in the same order as the dimension specified with
            format.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        use_cudnn_on_gpu: An optional `bool`. Defaults to `True`.
        data_format: An optional `string` from: `"NHWC", "NCHW"`. Defaults to `"NHWC"`.
            Specify the data format of the input and output data. With the
            default format "NHWC", the data is stored in the order of:
          [batch, in_height, in_width, in_channels].
            Alternatively, the format could be "NCHW", the data storage order of:
          [batch, in_channels, in_height, in_width].
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `filter`.
        4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
        w.r.t. the input of the convolution.
    """
    pass


def conv2d_transpose(value,
                     filter,
                     output_shape,
                     strides,
                     padding="SAME",
                     name=None):
    """
    [TensorFlow Docs]
    The transpose of `conv2d`.

    This operation is sometimes called "deconvolution" after [Deconvolutional
    Networks](http://www.matthewzeiler.com/pubs/cvpr2010/cvpr2010.pdf), but is
    actually the transpose (gradient) of `conv2d` rather than an actual
    deconvolution.

    Args:
        value: A 4-D `Tensor` of type `float` and shape
            `[batch, height, width, in_channels]`.
        filter: A 4-D `Tensor` with the same type as `value` and shape
            `[height, width, output_channels, in_channels]`.  `filter`'s
            `in_channels` dimension must match that of `value`.
        output_shape: A 1-D `Tensor` representing the output shape of the
            deconvolution op.
        strides: A list of ints. The stride of the sliding window for each
            dimension of the input tensor.
        padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
            See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
        name: Optional name for the returned tensor.

    Returns:
        A `Tensor` with the same type as `value`.

    Raises:
        ValueError: If input/output depth does not match `filter`'s shape, or if
            padding is other than `'VALID'` or `'SAME'`.
    """
    pass


def conv3d(input, filter, strides, padding, name=None):
    """
    [TensorFlow Docs]
    Computes a 3-D convolution given 5-D `input` and `filter` tensors.

    In signal processing, cross-correlation is a measure of similarity of
    two waveforms as a function of a time-lag applied to one of them. This
    is also known as a sliding dot product or sliding inner-product.

    Our Conv3D implements a form of cross-correlation.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Shape `[batch, in_depth, in_height, in_width, in_channels]`.
        filter: A `Tensor`. Must have the same type as `input`.
            Shape `[filter_depth, filter_height, filter_width, in_channels, out_channels]`.
            `in_channels` must match between `input` and `filter`.
        strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def conv3d_backprop_filter(input, filter, out_backprop, strides, padding,
                           name=None):
    """
    [TensorFlow Docs]
    Computes the gradients of 3D convolution with respect to the filter.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Shape `[batch, depth, rows, cols, in_channels]`.
        filter: A `Tensor`. Must have the same type as `input`.
            Shape `[depth, rows, cols, in_channels, out_channels]`.
            `in_channels` must match between `input` and `filter`.
        out_backprop: A `Tensor`. Must have the same type as `input`.
            Backprop signal of shape `[batch, out_depth, out_rows, out_cols, out_channels]`.
        strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def conv3d_backprop_input(input, filter, out_backprop, strides, padding,
                          name=None):
    """
    [TensorFlow Docs]
    Computes the gradients of 3D convolution with respect to the input.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Shape `[batch, depth, rows, cols, in_channels]`.
        filter: A `Tensor`. Must have the same type as `input`.
            Shape `[depth, rows, cols, in_channels, out_channels]`.
            `in_channels` must match between `input` and `filter`.
        out_backprop: A `Tensor`. Must have the same type as `input`.
            Backprop signal of shape `[batch, out_depth, out_rows, out_cols, out_channels]`.
        strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


"""Data Flow Operations."""
pass


def depthwise_conv2d(input, filter, strides, padding, name=None):
    """
    [TensorFlow Docs]
    Depthwise 2-D convolution.

    Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
    and a filter tensor of shape
    `[filter_height, filter_width, in_channels, channel_multiplier]`
    containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
    applies a different filter to each input channel (expanding from 1 channel
    to `channel_multiplier` channels for each), then concatenates the results
    together.  The output has `in_channels * channel_multiplier` channels.

    In detail,

            output[b, i, j, k * channel_multiplier + q] =
          sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                       filter[di, dj, k, q]

    Must have `strides[0] = strides[3] = 1`.  For the most common case of the
    same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

    Args:
        input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
        filter: 4-D with shape
            `[filter_height, filter_width, in_channels, channel_multiplier]`.
        strides: 1-D of size 4.  The stride of the sliding window for each
            dimension of `input`.
        padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
            See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
        name: A name for this operation (optional).

    Returns:
        A 4-D `Tensor` of shape
        `[batch, out_height, out_width, in_channels * channel_multiplier].`
    """
    pass


def depthwise_conv2d_native(input, filter, strides, padding, name=None):
    """
    [TensorFlow Docs]
    Computes a 2-D depthwise convolution given 4-D `input` and `filter` tensors.

    Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
    and a filter / kernel tensor of shape
    `[filter_height, filter_width, in_channels, channel_multiplier]`, containing
    `in_channels` convolutional filters of depth 1, `depthwise_conv2d` applies
    a different filter to each input channel (expanding from 1 channel to
    `channel_multiplier` channels for each), then concatenates the results
    together. Thus, the output has `in_channels * channel_multiplier` channels.

    for k in 0..in_channels-1
        for q in 0..channel_multiplier-1
            output[b, i, j, k * channel_multiplier + q] =
                sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                          filter[di, dj, k, q]

    Must have `strides[0] = strides[3] = 1`.  For the most common case of the same
    horizontal and vertices strides, `strides = [1, stride, stride, 1]`.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        filter: A `Tensor`. Must have the same type as `input`.
        strides: A list of `ints`.
            1-D of length 4.  The stride of the sliding window for each dimension
            of `input`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def depthwise_conv2d_native_backprop_filter(input, filter_sizes, out_backprop,
                                            strides, padding, name=None):
    """
    [TensorFlow Docs]
    Computes the gradients of depthwise convolution with respect to the filter.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
            4-D with shape `[batch, in_height, in_width, in_channels]`.
        filter_sizes: A `Tensor` of type `int32`.
            An integer vector representing the tensor shape of `filter`,
            where `filter` is a 4-D
            `[filter_height, filter_width, in_channels, depthwise_multiplier]` tensor.
        out_backprop: A `Tensor`. Must have the same type as `input`.
            4-D with shape `[batch, out_height, out_width, out_channels]`.
            Gradients w.r.t. the output of the convolution.
        strides: A list of `ints`.
            The stride of the sliding window for each dimension of the input
            of the convolution.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. 4-D with shape
        `[filter_height, filter_width, in_channels, out_channels]`.  Gradient w.r.t.
        the `filter` input of the convolution.
    """
    pass


def depthwise_conv2d_native_backprop_input(input_sizes, filter, out_backprop,
                                           strides, padding, name=None):
    """
    [TensorFlow Docs]
    Computes the gradients of depthwise convolution with respect to the input.

    Args:
        input_sizes: A `Tensor` of type `int32`.
            An integer vector representing the shape of `input`,
            where `input` is a 4-D `[batch, height, width, channels]` tensor.
        filter: A `Tensor`. Must be one of the following types: `float32`, `float64`.
            4-D with shape
            `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
        out_backprop: A `Tensor`. Must have the same type as `filter`.
            4-D with shape `[batch, out_height, out_width, out_channels]`.
            Gradients w.r.t. the output of the convolution.
        strides: A list of `ints`.
            The stride of the sliding window for each dimension of the input
            of the convolution.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `filter`.
        4-D with shape `[batch, in_height, in_width, in_channels]`.  Gradient
        w.r.t. the input of the convolution.
    """
    pass


def dilation2d(input, filter, strides, rates, padding, name=None):
    """
    [TensorFlow Docs]
    Computes the grayscale dilation of 4-D `input` and 3-D `filter` tensors.

    The `input` tensor has shape `[batch, in_height, in_width, depth]` and the
    `filter` tensor has shape `[filter_height, filter_width, depth]`, i.e., each
    input channel is processed independently of the others with its own structuring
    function. The `output` tensor has shape
    `[batch, out_height, out_width, depth]`. The spatial dimensions of the output
    tensor depend on the `padding` algorithm. We currently only support the default
    "NHWC" `data_format`.

    In detail, the grayscale morphological 2-D dilation is the max-sum correlation
    (for consistency with `conv2d`, we use unmirrored filters):

            output[b, y, x, c] =
         max_{dy, dx} input[b,
                            strides[1] * y + rates[1] * dy,
                            strides[2] * x + rates[2] * dx,
                            c] +
                      filter[dy, dx, c]

    Max-pooling is a special case when the filter has size equal to the pooling
    kernel size and contains all zeros.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        filter: A `Tensor`. Must have the same type as `input`.
        strides: A list of `ints` that has length `>= 4`.
        rates: A list of `ints` that has length `>= 4`.
        padding: A `string` from: `"SAME", "VALID"`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
    """
    pass


def dilation2d_backprop_filter(input, filter, out_backprop, strides, rates,
                               padding, name=None):
    """
    [TensorFlow Docs]
    Computes the gradient of morphological 2-D dilation with respect to the filter.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
            4-D with shape `[batch, in_height, in_width, depth]`.
        filter: A `Tensor`. Must have the same type as `input`.
            3-D with shape `[filter_height, filter_width, depth]`.
        out_backprop: A `Tensor`. Must have the same type as `input`.
            4-D with shape `[batch, out_height, out_width, depth]`.
        strides: A list of `ints` that has length `>= 4`.
            1-D of length 4. The stride of the sliding window for each dimension of
            the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
        rates: A list of `ints` that has length `>= 4`.
            1-D of length 4. The input stride for atrous morphological dilation.
            Must be: `[1, rate_height, rate_width, 1]`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        3-D with shape `[filter_height, filter_width, depth]`.
    """
    pass


def dilation2d_backprop_input(input, filter, out_backprop, strides, rates,
                              padding, name=None):
    """
    [TensorFlow Docs]
    Computes the gradient of morphological 2-D dilation with respect to the input.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
            4-D with shape `[batch, in_height, in_width, depth]`.
        filter: A `Tensor`. Must have the same type as `input`.
            3-D with shape `[filter_height, filter_width, depth]`.
        out_backprop: A `Tensor`. Must have the same type as `input`.
            4-D with shape `[batch, out_height, out_width, depth]`.
        strides: A list of `ints` that has length `>= 4`.
            1-D of length 4. The stride of the sliding window for each dimension of
            the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
        rates: A list of `ints` that has length `>= 4`.
            1-D of length 4. The input stride for atrous morphological dilation.
            Must be: `[1, rate_height, rate_width, 1]`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`.
        4-D with shape `[batch, in_height, in_width, depth]`.
    """
    pass


def dropout(x, keep_prob, noise_shape=None, seed=None, name=None):
    """
    [TensorFlow Docs]
    Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Args:
        x: A tensor.
        keep_prob: A scalar `Tensor` with the same type as x. The probability
            that each element is kept.
        noise_shape: A 1-D `Tensor` of type `int32`, representing the
            shape for randomly generated keep/drop flags.
        seed: A Python integer. Used to create random seeds. See
            [`set_random_seed`](../../api_docs/python/constant_op.md#set_random_seed)
            for behavior.
        name: A name for this operation (optional).

    Returns:
        A Tensor of the same shape of `x`.

    Raises:
        ValueError: If `keep_prob` is not in `(0, 1]`.
    """
    pass


"""Library of dtypes (Tensor element types)."""
pass


def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None):
    """
    [TensorFlow Docs]
    Creates a recurrent neural network specified by RNNCell `cell`.

    This function is functionally identical to the function `rnn` above, but
    performs fully dynamic unrolling of `inputs`.

    Unlike `rnn`, the input `inputs` is not a Python list of `Tensors`.  Instead,
    it is a single `Tensor` where the maximum time is either the first or second
    dimension (see the parameter `time_major`).  The corresponding output is
    a single `Tensor` having the same number of time steps and batch size.

    The parameter `sequence_length` is required and dynamic calculation is
    automatically performed.

    Args:
        cell: An instance of RNNCell.
        inputs: The RNN inputs.
            If time_major == False (default), this must be a tensor of shape:
                `[batch_size, max_time, input_size]`.
            If time_major == True, this must be a tensor of shape:
                `[max_time, batch_size, input_size]`.
        sequence_length: (optional) An int32/int64 vector sized `[batch_size]`.
        initial_state: (optional) An initial state for the RNN.
            If `cell.state_size` is an integer, this must be
            a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
            If `cell.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state.  Required if
            initial_state is not provided.
        parallel_iterations: (Default: 32).  The number of iterations to run in
            parallel.  Those operations which do not have any temporal dependency
            and can be run in parallel, will be.  This parameter trades off
            time for space.  Values >> 1 use more memory but take less time,
            while smaller values use less memory but computations take longer.
        swap_memory: Transparently swap the tensors produced in forward inference
            but needed for back prop from GPU to CPU.  This allows training RNNs
            which would typically not fit on a single GPU, with very minimal (or no)
            performance penalty.
        time_major: The shape format of the `inputs` and `outputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
        A pair (outputs, state) where:
            outputs: The RNN output `Tensor`.
                If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.
                If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.
            state: The final state.  If `cell.state_size` is a `Tensor`, this
                will be shaped `[batch_size, cell.state_size]`.  If it is a tuple,
                this be a tuple with shapes `[batch_size, s] for s in cell.state_size`.

    Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If inputs is None or an empty list.
    """
    pass


def elu(features, name=None):
    """
    [TensorFlow Docs]
    Computes exponential linear: `exp(features) - 1` if < 0, `features` otherwise.

    See [Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)
    ](http://arxiv.org/abs/1511.07289)

    Args:
        features: A `Tensor`. Must be one of the following types: `float32`, `float64`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `features`.
    """
    pass


def embedding_lookup(params, ids, partition_strategy="mod", name=None,
                     validate_indices=True):
    """
    [TensorFlow Docs]
    Looks up `ids` in a list of embedding tensors.

    This function is used to perform parallel lookups on the list of
    tensors in `params`.  It is a generalization of
    [`tf.gather()`](../../api_docs/python/array_ops.md#gather), where `params` is
    interpreted as a partition of a larger embedding tensor.

    If `len(params) > 1`, each element `id` of `ids` is partitioned between
    the elements of `params` according to the `partition_strategy`.
    In all strategies, if the id space does not evenly divide the number of
    partitions, each of the first `(max_id + 1) % len(params)` partitions will
    be assigned one more id.

    If `partition_strategy` is `"mod"`, we assign each id to partition
    `p = id % len(params)`. For instance,
    13 ids are split across 5 partitions as:
    `[[0, 5, 10], [1, 6, 11], [2, 7, 12], [3, 8], [4, 9]]`

    If `partition_strategy` is `"div"`, we assign ids to partitions in a
    contiguous manner. In this case, 13 ids are split across 5 partitions as:
    `[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10], [11, 12]]`

    The results of the lookup are concatenated into a dense
    tensor. The returned tensor has shape `shape(ids) + shape(params)[1:]`.

    Args:
        params: A list of tensors with the same type and which can be concatenated
            along dimension 0. Each `Tensor` must be appropriately sized for the given
            `partition_strategy`.
        ids: A `Tensor` with type `int32` or `int64` containing the ids to be looked
            up in `params`.
        partition_strategy: A string specifying the partitioning strategy, relevant
            if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
            is `"mod"`.
        name: A name for the operation (optional).
        validate_indices: Whether or not to validate gather indices.

    Returns:
        A `Tensor` with the same type as the tensors in `params`.

    Raises:
        ValueError: If `params` is empty.
    """
    pass


def embedding_lookup_sparse(params, sp_ids, sp_weights,
                            partition_strategy="mod",
                            name=None,
                            combiner="mean"):
    """
    [TensorFlow Docs]
    Computes embeddings for the given ids and weights.

    This op assumes that there is at least one id for each row in the dense tensor
    represented by sp_ids (i.e. there are no rows with empty features), and that
    all the indices of sp_ids are in canonical row-major order.

    It also assumes that all id values lie in the range [0, p0), where p0
    is the sum of the size of params along dimension 0.

    Args:
        params: A single tensor representing the complete embedding tensor,
            or a list of P tensors all of same shape except for the first dimension,
            representing sharded embedding tensors.
        sp_ids: N x M SparseTensor of int64 ids (typically from FeatureValueToId),
            where N is typically batch size and M is arbitrary.
        sp_weights: either a SparseTensor of float / double weights, or None to
            indicate all weights should be taken to be 1. If specified, sp_weights
            must have exactly the same shape and indices as sp_ids.
        partition_strategy: A string specifying the partitioning strategy, relevant
            if `len(params) > 1`. Currently `"div"` and `"mod"` are supported. Default
            is `"mod"`. See `tf.nn.embedding_lookup` for more details.
        name: Optional name for the op.
        combiner: A string specifying the reduction op. Currently "mean", "sqrtn"
            and "sum" are supported.
            "sum" computes the weighted sum of the embedding results for each row.
            "mean" is the weighted sum divided by the total weight.
            "sqrtn" is the weighted sum divided by the square root of the sum of the
            squares of the weights.

    Returns:
        A dense tensor representing the combined embeddings for the
        sparse ids. For each row in the dense tensor represented by sp_ids, the op
        looks up the embeddings for all ids in that row, multiplies them by the
        corresponding weight, and combines these embeddings as specified.

        In other words, if
            shape(combined params) = [p0, p1, ..., pm]
        and
            shape(sp_ids) = shape(sp_weights) = [d0, d1, ..., dn]
        then
            shape(output) = [d0, d1, ..., dn-1, p1, ..., pm].

        For instance, if params is a 10x20 matrix, and sp_ids / sp_weights are

            [0, 0]: id 1, weight 2.0
            [0, 1]: id 3, weight 0.5
            [1, 0]: id 0, weight 1.0
            [2, 3]: id 1, weight 3.0

        with combiner="mean", then the output will be a 3x20 matrix where
            output[0, :] = (params[1, :] * 2.0 + params[3, :] * 0.5) / (2.0 + 0.5)
            output[1, :] = params[0, :] * 1.0
            output[2, :] = params[1, :] * 3.0

    Raises:
        TypeError: If sp_ids is not a SparseTensor, or if sp_weights is neither
            None nor SparseTensor.
        ValueError: If combiner is not one of {"mean", "sqrtn", "sum"}.
    """
    pass


"""Operations for embeddings."""
pass


def erosion2d(value, kernel, strides, rates, padding, name=None):
    """
    [TensorFlow Docs]
    Computes the grayscale erosion of 4-D `value` and 3-D `kernel` tensors.

    The `value` tensor has shape `[batch, in_height, in_width, depth]` and the
    `kernel` tensor has shape `[kernel_height, kernel_width, depth]`, i.e.,
    each input channel is processed independently of the others with its own
    structuring function. The `output` tensor has shape
    `[batch, out_height, out_width, depth]`. The spatial dimensions of the
    output tensor depend on the `padding` algorithm. We currently only support the
    default "NHWC" `data_format`.

    In detail, the grayscale morphological 2-D erosion is given by:

            output[b, y, x, c] =
         min_{dy, dx} value[b,
                            strides[1] * y - rates[1] * dy,
                            strides[2] * x - rates[2] * dx,
                            c] -
                      kernel[dy, dx, c]

    Duality: The erosion of `value` by the `kernel` is equal to the negation of
    the dilation of `-value` by the reflected `kernel`.

    Args:
        value: A `Tensor`. 4-D with shape `[batch, in_height, in_width, depth]`.
        kernel: A `Tensor`. Must have the same type as `value`.
            3-D with shape `[kernel_height, kernel_width, depth]`.
        strides: A list of `ints` that has length `>= 4`.
            1-D of length 4. The stride of the sliding window for each dimension of
            the input tensor. Must be: `[1, stride_height, stride_width, 1]`.
        rates: A list of `ints` that has length `>= 4`.
            1-D of length 4. The input stride for atrous morphological dilation.
            Must be: `[1, rate_height, rate_width, 1]`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional). If not specified "erosion2d"
            is used.

    Returns:
        A `Tensor`. Has the same type as `value`.
        4-D with shape `[batch, out_height, out_width, depth]`.

    Raises:
        ValueError: If the `value` depth does not match `kernel`' shape, or if
            padding is other than `'VALID'` or `'SAME'`.
    """
    pass


def fixed_unigram_candidate_sampler(true_classes,
                                    num_true,
                                    num_sampled,
                                    unique,
                                    range_max,
                                    vocab_file='',
                                    distortion=1.0,
                                    num_reserved_ids=0,
                                    num_shards=1,
                                    shard=0,
                                    unigrams=(),
                                    seed=None,
                                    name=None):
    """
    [TensorFlow Docs]
    Samples a set of classes using the provided (fixed) base distribution.

    This operation randomly samples a tensor of sampled classes
    (`sampled_candidates`) from the range of integers `[0, range_max)`.

    The elements of `sampled_candidates` are drawn without replacement
    (if `unique=True`) or with replacement (if `unique=False`) from
    the base distribution.

    The base distribution is read from a file or passed in as an
    in-memory array. There is also an option to skew the distribution by
    applying a distortion power to the weights.

    In addition, this operation returns tensors `true_expected_count`
    and `sampled_expected_count` representing the number of times each
    of the target classes (`true_classes`) and the sampled
    classes (`sampled_candidates`) is expected to occur in an average
    tensor of sampled classes.  These values correspond to `Q(y|x)`
    defined in [this
    document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
    If `unique=True`, then these are post-rejection probabilities and we
    compute them approximately.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size,
            num_true]`. The target classes.
        num_true: An `int`.  The number of target classes per training example.
        num_sampled: An `int`.  The number of classes to randomly sample per batch.
        unique: A `bool`. Determines whether all sampled classes in a batch are
            unique.
        range_max: An `int`. The number of possible classes.
        vocab_file: Each valid line in this file (which should have a CSV-like
            format) corresponds to a valid word ID. IDs are in sequential order,
            starting from num_reserved_ids. The last entry in each line is expected
            to be a value corresponding to the count or relative probability. Exactly
            one of `vocab_file` and `unigrams` needs to be passed to this operation.
        distortion: The distortion is used to skew the unigram probability
            distribution.  Each weight is first raised to the distortion's power
            before adding to the internal unigram distribution. As a result,
            `distortion = 1.0` gives regular unigram sampling (as defined by the vocab
            file), and `distortion = 0.0` gives a uniform distribution.
        num_reserved_ids: Optionally some reserved IDs can be added in the range
            `[0, num_reserved_ids]` by the users. One use case is that a special
            unknown word token is used as ID 0. These IDs will have a sampling
            probability of 0.
        num_shards: A sampler can be used to sample from a subset of the original
            range in order to speed up the whole computation through parallelism. This
            parameter (together with `shard`) indicates the number of partitions that
            are being used in the overall computation.
        shard: A sampler can be used to sample from a subset of the original range
            in order to speed up the whole computation through parallelism. This
            parameter (together with `num_shards`) indicates the particular partition
            number of the operation, when partitioning is being used.
        unigrams: A list of unigram counts or probabilities, one per ID in
            sequential order. Exactly one of `vocab_file` and `unigrams` should be
            passed to this operation.
        seed: An `int`. An operation-specific seed. Default is 0.
        name: A name for the operation (optional).

    Returns:
        sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
            The sampled classes.
        true_expected_count: A tensor of type `float`.  Same shape as
            `true_classes`. The expected counts under the sampling distribution
            of each of `true_classes`.
        sampled_expected_count: A tensor of type `float`. Same shape as
            `sampled_candidates`. The expected counts under the sampling distribution
            of each of `sampled_candidates`.

    """
    pass


def in_top_k(predictions, targets, k, name=None):
    """
    [TensorFlow Docs]
    Says whether the targets are in the top `K` predictions.

    This outputs a `batch_size` bool array, an entry `out[i]` is `true` if the
    prediction for the target class is among the top `k` predictions among
    all predictions for example `i`. Note that the behavior of `InTopK` differs
    from the `TopK` op in its handling of ties; if multiple classes have the
    same prediction value and straddle the top-`k` boundary, all of those
    classes are considered to be in the top `k`.

    More formally, let

        \\(predictions_i\\) be the predictions for all classes for example `i`,
        \\(targets_i\\) be the target class for example `i`,
        \\(out_i\\) be the output for example `i`,

    $$out_i = predictions_{i, targets_i} \in TopKIncludingTies(predictions_i)$$

    Args:
        predictions: A `Tensor` of type `float32`.
            A `batch_size` x `classes` tensor.
        targets: A `Tensor`. Must be one of the following types: `int32`, `int64`.
            A `batch_size` vector of class ids.
        k: An `int`. Number of top elements to look at for computing precision.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `bool`. Computed Precision at `k` as a `bool Tensor`.
    """
    pass


"""Operations often used for initializing tensors."""
pass


def l2_loss(t, name=None):
    """
    [TensorFlow Docs]
    L2 Loss.

    Computes half the L2 norm of a tensor without the `sqrt`:

            output = sum(t ** 2) / 2

    Args:
        t: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Typically 2-D, but may have any dimensions.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `t`. 0-D.
    """
    pass


def l2_normalize(x, dim, epsilon=1e-12, name=None):
    """
    [TensorFlow Docs]
    Normalizes along dimension `dim` using an L2 norm.

    For a 1-D tensor with `dim = 0`, computes

            output = x / sqrt(max(sum(x**2), epsilon))

    For `x` with more dimensions, independently normalizes each 1-D slice along
    dimension `dim`.

    Args:
        x: A `Tensor`.
        dim: Dimension along which to normalize.
        epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
            divisor if `norm < sqrt(epsilon)`.
        name: A name for this operation (optional).

    Returns:
        A `Tensor` with the same shape as `x`.
    """
    pass


def learned_unigram_candidate_sampler(true_classes, num_true, num_sampled,
                                      unique, range_max, seed=None, name=None):
    """
    [TensorFlow Docs]
    Samples a set of classes from a distribution learned during training.

    This operation randomly samples a tensor of sampled classes
    (`sampled_candidates`) from the range of integers `[0, range_max)`.

    The elements of `sampled_candidates` are drawn without replacement
    (if `unique=True`) or with replacement (if `unique=False`) from
    the base distribution.

    The base distribution for this operation is constructed on the fly
    during training.  It is a unigram distribution over the target
    classes seen so far during training.  Every integer in `[0, range_max)`
    begins with a weight of 1, and is incremented by 1 each time it is
    seen as a target class.  The base distribution is not saved to checkpoints,
    so it is reset when the model is reloaded.

    In addition, this operation returns tensors `true_expected_count`
    and `sampled_expected_count` representing the number of times each
    of the target classes (`true_classes`) and the sampled
    classes (`sampled_candidates`) is expected to occur in an average
    tensor of sampled classes.  These values correspond to `Q(y|x)`
    defined in [this
    document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
    If `unique=True`, then these are post-rejection probabilities and we
    compute them approximately.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size,
            num_true]`. The target classes.
        num_true: An `int`.  The number of target classes per training example.
        num_sampled: An `int`.  The number of classes to randomly sample per batch.
        unique: A `bool`. Determines whether all sampled classes in a batch are
            unique.
        range_max: An `int`. The number of possible classes.
        seed: An `int`. An operation-specific seed. Default is 0.
        name: A name for the operation (optional).

    Returns:
        sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
            The sampled classes.
        true_expected_count: A tensor of type `float`.  Same shape as
            `true_classes`. The expected counts under the sampling distribution
            of each of `true_classes`.
        sampled_expected_count: A tensor of type `float`. Same shape as
            `sampled_candidates`. The expected counts under the sampling distribution
            of each of `sampled_candidates`.

    """
    pass


def lrn(input, depth_radius=None, bias=None, alpha=None, beta=None,
        name=None):
    """
    [TensorFlow Docs]
    Local Response Normalization.

    The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
    dimension), and each vector is normalized independently.  Within a given vector,
    each component is divided by the weighted, squared sum of inputs within
    `depth_radius`.  In detail,

            sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
            output = input / (bias + alpha * sqr_sum) ** beta

    For details, see [Krizhevsky et al., ImageNet classification with deep
    convolutional neural networks (NIPS 2012)]
    (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

    Args:
        input: A `Tensor` of type `float32`. 4-D.
        depth_radius: An optional `int`. Defaults to `5`.
            0-D.  Half-width of the 1-D normalization window.
        bias: An optional `float`. Defaults to `1`.
            An offset (usually positive to avoid dividing by 0).
        alpha: An optional `float`. Defaults to `1`.
            A scale factor, usually positive.
        beta: An optional `float`. Defaults to `0.5`. An exponent.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `float32`.
    """
    pass


def log_softmax(logits, name=None):
    """
    [TensorFlow Docs]
    Computes log softmax activations.

    For each batch `i` and class `j` we have

            logsoftmax[i, j] = logits[i, j] - log(sum(exp(logits[i])))

    Args:
        logits: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
            2-D with shape `[batch_size, num_classes]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
    """
    pass


def log_uniform_candidate_sampler(true_classes, num_true, num_sampled, unique,
                                  range_max, seed=None, name=None):
    """
    [TensorFlow Docs]
    Samples a set of classes using a log-uniform (Zipfian) base distribution.

    This operation randomly samples a tensor of sampled classes
    (`sampled_candidates`) from the range of integers `[0, range_max)`.

    The elements of `sampled_candidates` are drawn without replacement
    (if `unique=True`) or with replacement (if `unique=False`) from
    the base distribution.

    The base distribution for this operation is an approximately log-uniform
    or Zipfian distribution:

    `P(class) = (log(class + 2) - log(class + 1)) / log(range_max + 1)`

    This sampler is useful when the target classes approximately follow such
    a distribution - for example, if the classes represent words in a lexicon
    sorted in decreasing order of frequency. If your classes are not ordered by
    decreasing frequency, do not use this op.

    In addition, this operation returns tensors `true_expected_count`
    and `sampled_expected_count` representing the number of times each
    of the target classes (`true_classes`) and the sampled
    classes (`sampled_candidates`) is expected to occur in an average
    tensor of sampled classes.  These values correspond to `Q(y|x)`
    defined in [this
    document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
    If `unique=True`, then these are post-rejection probabilities and we
    compute them approximately.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size,
            num_true]`. The target classes.
        num_true: An `int`.  The number of target classes per training example.
        num_sampled: An `int`.  The number of classes to randomly sample per batch.
        unique: A `bool`. Determines whether all sampled classes in a batch are
            unique.
        range_max: An `int`. The number of possible classes.
        seed: An `int`. An operation-specific seed. Default is 0.
        name: A name for the operation (optional).

    Returns:
        sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
            The sampled classes.
        true_expected_count: A tensor of type `float`.  Same shape as
            `true_classes`. The expected counts under the sampling distribution
            of each of `true_classes`.
        sampled_expected_count: A tensor of type `float`. Same shape as
            `sampled_candidates`. The expected counts under the sampling distribution
            of each of `sampled_candidates`.
    """
    pass


"""Logging and Summary Operations."""
pass


def lrn(input, depth_radius=None, bias=None, alpha=None, beta=None,
        name=None):
    """
    [TensorFlow Docs]
    Local Response Normalization.

    The 4-D `input` tensor is treated as a 3-D array of 1-D vectors (along the last
    dimension), and each vector is normalized independently.  Within a given vector,
    each component is divided by the weighted, squared sum of inputs within
    `depth_radius`.  In detail,

            sqr_sum[a, b, c, d] =
          sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
            output = input / (bias + alpha * sqr_sum) ** beta

    For details, see [Krizhevsky et al., ImageNet classification with deep
    convolutional neural networks (NIPS 2012)]
    (http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks).

    Args:
        input: A `Tensor` of type `float32`. 4-D.
        depth_radius: An optional `int`. Defaults to `5`.
            0-D.  Half-width of the 1-D normalization window.
        bias: An optional `float`. Defaults to `1`.
            An offset (usually positive to avoid dividing by 0).
        alpha: An optional `float`. Defaults to `1`.
            A scale factor, usually positive.
        beta: An optional `float`. Defaults to `0.5`. An exponent.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of type `float32`.
    """
    pass


def make_all(module_name, doc_string_modules=None):
    """
    [TensorFlow Docs]
    Generate `__all__` from the docstring of one or more modules.

    Usage: `make_all(__name__)` or
    `make_all(__name__, [sys.modules(__name__), other_module])`. The doc string
    modules must each a docstring, and `__all__` will contain all symbols with
    `@@` references, where that symbol currently exists in the module named
    `module_name`.

    Args:
        module_name: The name of the module (usually `__name__`).
        doc_string_modules: a list of modules from which to take docstring.
        If None, then a list containing only the module named `module_name` is used.

    Returns:
        A list suitable for use as `__all__`.
    """
    pass


def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
    """
    [TensorFlow Docs]
    Performs the max pooling on the input.

    Args:
        value: A 4-D `Tensor` with shape `[batch, height, width, channels]` and
            type `tf.float32`.
        ksize: A list of ints that has length >= 4.  The size of the window for
            each dimension of the input tensor.
        strides: A list of ints that has length >= 4.  The stride of the sliding
            window for each dimension of the input tensor.
        padding: A string, either `'VALID'` or `'SAME'`. The padding algorithm.
            See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
        data_format: A string. 'NHWC' and 'NCHW' are supported.
        name: Optional name for the operation.

    Returns:
        A `Tensor` with type `tf.float32`.  The max pooled output tensor.
    """
    pass


def max_pool3d(input, ksize, strides, padding, name=None):
    """
    [TensorFlow Docs]
    Performs 3D max pooling on the input.

    Args:
        input: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Shape `[batch, depth, rows, cols, channels]` tensor to pool over.
        ksize: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The size of the window for each dimension of
            the input tensor. Must have `ksize[0] = ksize[1] = 1`.
        strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `input`. The max pooled output tensor.
    """
    pass


def max_pool3d_grad(orig_input, orig_output, grad, ksize, strides, padding,
                    name=None):
    """
    [TensorFlow Docs]
    Computes gradients of max pooling function.

    Args:
        orig_input: A `Tensor` of type `float32`. The original input tensor.
        orig_output: A `Tensor` of type `float32`. The original output tensor.
        grad: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`, `complex128`, `qint8`, `quint8`, `qint32`, `half`.
            Output backprop of shape `[batch, depth, rows, cols, channels]`.
        ksize: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The size of the window for each dimension of
            the input tensor. Must have `ksize[0] = ksize[1] = 1`.
        strides: A list of `ints` that has length `>= 5`.
            1-D tensor of length 5. The stride of the sliding window for each
            dimension of `input`. Must have `strides[0] = strides[4] = 1`.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `grad`.
    """
    pass


def max_pool_with_argmax(input, ksize, strides, padding, Targmax=None,
                         name=None):
    """
    [TensorFlow Docs]
    Performs max pooling on the input and outputs both max values and indices.

    The indices in `argmax` are flattened, so that a maximum value at position
    `[b, y, x, c]` becomes flattened index
    `((b * height + y) * width + x) * channels + c`.

    Args:
        input: A `Tensor` of type `float32`.
            4-D with shape `[batch, height, width, channels]`.  Input to pool over.
        ksize: A list of `ints` that has length `>= 4`.
            The size of the window for each dimension of the input tensor.
        strides: A list of `ints` that has length `>= 4`.
            The stride of the sliding window for each dimension of the
            input tensor.
        padding: A `string` from: `"SAME", "VALID"`.
            The type of padding algorithm to use.
        Targmax: An optional `tf.DType` from: `tf.int32, tf.int64`. Defaults to `tf.int64`.
        name: A name for the operation (optional).

    Returns:
        A tuple of `Tensor` objects (output, argmax).
        output: A `Tensor` of type `float32`. The max pooled output tensor.
        argmax: A `Tensor` of type `Targmax`. 4-D.  The flattened indices of the max values chosen for each output.
    """
    pass


def moments(x, axes, shift=None, name=None, keep_dims=False):
    """
    [TensorFlow Docs]
    Calculate the mean and variance of `x`.

    The mean and variance are calculated by aggregating the contents of `x`
    across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
    and variance of a vector.

    When using these moments for batch normalization (see
    `tf.nn.batch_normalization`):
        * for so-called "global normalization", used with convolutional filters with
            shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
        * for simple batch normalization pass `axes=[0]` (batch only).

    Args:
        x: A `Tensor`.
        axes: array of ints.  Axes along which to compute mean and
            variance.
        shift: A `Tensor` containing the value by which to shift the data for
            numerical stability, or `None` if no shift is to be performed. A shift
            close to the true mean provides the most numerically stable results.
        keep_dims: produce moments with the same dimensionality as the input.
        name: Name used to scope the operations that compute the moments.

    Returns:
        Two `Tensor` objects: `mean` and `variance`.
    """
    pass


def nce_loss(weights, biases, inputs, labels, num_sampled, num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss"):
    """
    [TensorFlow Docs]
    Computes and returns the noise-contrastive estimation training loss.

    See [Noise-contrastive estimation: A new estimation principle for
    unnormalized statistical models]
    (http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
    Also see our [Candidate Sampling Algorithms Reference]
    (../../extras/candidate_sampling.pdf)

    Note: In the case where `num_true` > 1, we assign to each target class
    the target probability 1 / `num_true` so that the target probabilities
    sum to 1 per-example.

    Note: It would be useful to allow a variable number of target classes per
    example.  We hope to provide this functionality in a future release.
    For now, if you have a variable number of target classes, you can pad them
    out to a constant number by either repeating them or by padding
    with an otherwise unused class.

    Args:
        weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
                objects whose concatenation along dimension 0 has shape
                [num_classes, dim].  The (possibly-partitioned) class embeddings.
        biases: A `Tensor` of shape `[num_classes]`.  The class biases.
        inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
                activations of the input network.
        labels: A `Tensor` of type `int64` and shape `[batch_size,
                num_true]`. The target classes.
        num_sampled: An `int`.  The number of classes to randomly sample per batch.
        num_classes: An `int`. The number of possible classes.
        num_true: An `int`.  The number of target classes per training example.
        sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
                `sampled_expected_count`) returned by a `*_candidate_sampler` function.
                (if None, we default to `log_uniform_candidate_sampler`)
        remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
                where a sampled class equals one of the target classes.  If set to
                `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
                learning to generate log-odds instead of log probabilities.  See
                our [Candidate Sampling Algorithms Reference]
                (../../extras/candidate_sampling.pdf).
                Default is False.
        partition_strategy: A string specifying the partitioning strategy, relevant
                if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
                Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
        name: A name for the operation (optional).

    Returns:
        A `batch_size` 1-D tensor of per-example NCE losses.
    """
    pass


def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
    """
    [TensorFlow Docs]
    Calculate the mean and variance of based on the sufficient statistics.

    Args:
        counts: A `Tensor` containing a the total count of the data (one value).
        mean_ss: A `Tensor` containing the mean sufficient statistics: the (possibly
            shifted) sum of the elements to average over.
        variance_ss: A `Tensor` containing the variance sufficient statistics: the
            (possibly shifted) squared sum of the data to compute the variance over.
        shift: A `Tensor` containing the value by which the data is shifted for
            numerical stability, or `None` if no shift was performed.
        name: Name used to scope the operations that compute the moments.

    Returns:
        Two `Tensor` objects: `mean` and `variance`.
    """
    pass


def relu(features, name=None):
    """
    [TensorFlow Docs]
    Computes rectified linear: `max(features, 0)`.

    Args:
        features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `features`.
    """
    pass


def relu6(features, name=None):
    """
    [TensorFlow Docs]
    Computes Rectified Linear 6: `min(max(features, 0), 6)`.

    Args:
        features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`,
            `int16`, or `int8`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` with the same type as `features`.
    """
    pass


def relu_layer(x, weights, biases, name=None):
    """
    [TensorFlow Docs]
    Computes Relu(x * weight + biases).

    Args:
        x: a 2D tensor.  Dimensions typically: batch, in_units
        weights: a 2D tensor.  Dimensions typically: in_units, out_units
        biases: a 1D tensor.  Dimensions: out_units
        name: A name for the operation (optional).  If not specified
            "nn_relu_layer" is used.

    Returns:
        A 2-D Tensor computing relu(matmul(x, weights) + biases).
        Dimensions typically: batch, out_units.
    """
    pass


def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
    """
    [TensorFlow Docs]
    Creates a recurrent neural network specified by RNNCell `cell`.

    The simplest form of RNN network generated is:
        state = cell.zero_state(...)
        outputs = []
        for input_ in inputs:
            output, state = cell(input_, state)
            outputs.append(output)
        return (outputs, state)

    However, a few other options are available:

    An initial state can be provided.
    If the sequence_length vector is provided, dynamic calculation is performed.
    This method of calculation does not compute the RNN steps past the maximum
    sequence length of the minibatch (thus saving computational time),
    and properly propagates the state at an example's sequence length
    to the final state output.

    The dynamic calculation performed is, at time t for batch row b,
        (output, state)(b, t) =
            (t >= sequence_length(b))
                ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
                : cell(input(b, t), state(b, t - 1))

    Args:
        cell: An instance of RNNCell.
        inputs: A length T list of inputs, each a tensor of shape
            [batch_size, input_size].
        initial_state: (optional) An initial state for the RNN.
            If `cell.state_size` is an integer, this must be
            a tensor of appropriate type and shape `[batch_size x cell.state_size]`.
            If `cell.state_size` is a tuple, this should be a tuple of
            tensors having shapes `[batch_size, s] for s in cell.state_size`.
        dtype: (optional) The data type for the initial state.  Required if
            initial_state is not provided.
        sequence_length: Specifies the length of each sequence in inputs.
            An int32 or int64 vector (tensor) size `[batch_size]`, values in `[0, T)`.
        scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
        A pair (outputs, state) where:
            - outputs is a length T list of outputs (one for each input)
            - state is the final state

    Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If `inputs` is `None` or an empty list, or if the input depth
            (column size) cannot be inferred from inputs via shape inference.
    """
    pass


def sampled_softmax_loss(weights, biases, inputs, labels, num_sampled,
                         num_classes, num_true=1,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         name="sampled_softmax_loss"):
    """
    [TensorFlow Docs]
    Computes and returns the sampled softmax training loss.

    This is a faster way to train a softmax classifier over a huge number of
    classes.

    This operation is for training only.  It is generally an underestimate of
    the full softmax loss.

    At inference time, you can compute full softmax probabilities with the
    expression `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.

    See our [Candidate Sampling Algorithms Reference]
    (../../extras/candidate_sampling.pdf)

    Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
    ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

    Args:
        weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
                objects whose concatenation along dimension 0 has shape
                [num_classes, dim].  The (possibly-sharded) class embeddings.
        biases: A `Tensor` of shape `[num_classes]`.  The class biases.
        inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
                activations of the input network.
        labels: A `Tensor` of type `int64` and shape `[batch_size,
                num_true]`. The target classes.  Note that this format differs from
                the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
        num_sampled: An `int`.  The number of classes to randomly sample per batch.
        num_classes: An `int`. The number of possible classes.
        num_true: An `int`.  The number of target classes per training example.
        sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
                `sampled_expected_count`) returned by a `*_candidate_sampler` function.
                (if None, we default to `log_uniform_candidate_sampler`)
        remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
                where a sampled class equals one of the target classes.  Default is
                True.
        partition_strategy: A string specifying the partitioning strategy, relevant
                if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
                Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
        name: A name for the operation (optional).

    Returns:
        A `batch_size` 1-D tensor of per-example sampled softmax losses.

    """
    pass


def separable_conv2d(input, depthwise_filter, pointwise_filter, strides,
                     padding,
                     name=None):
    """
    [TensorFlow Docs]
    2-D convolution with separable filters.

    Performs a depthwise convolution that acts separately on channels followed by
    a pointwise convolution that mixes channels.  Note that this is separability
    between dimensions `[1, 2]` and `3`, not spatial separability between
    dimensions `1` and `2`.

    In detail,

            output[b, i, j, k] = sum_{di, dj, q, r]
          input[b, strides[1] * i + di, strides[2] * j + dj, q] *
          depthwise_filter[di, dj, q, r] *
          pointwise_filter[0, 0, q * channel_multiplier + r, k]

    `strides` controls the strides for the depthwise convolution only, since
    the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
    `strides[0] = strides[3] = 1`.  For the most common case of the same
    horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

    Args:
        input: 4-D `Tensor` with shape `[batch, in_height, in_width, in_channels]`.
        depthwise_filter: 4-D `Tensor` with shape
            `[filter_height, filter_width, in_channels, channel_multiplier]`.
            Contains `in_channels` convolutional filters of depth 1.
        pointwise_filter: 4-D `Tensor` with shape
            `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
            filter to mix channels after `depthwise_filter` has convolved spatially.
        strides: 1-D of size 4.  The strides for the depthwise convolution for
            each dimension of `input`.
        padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
            See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
        name: A name for this operation (optional).

    Returns:
        A 4-D `Tensor` of shape `[batch, out_height, out_width, out_channels]`.

    Raises:
        ValueError: If channel_multiplier * in_channels > out_channels,
            which means that the separable convolution is overparameterized.
    """
    pass


def sigmoid(x, name=None):
    """
    [TensorFlow Docs]
    Computes sigmoid of `x` element-wise.

    Specifically, `y = 1 / (1 + exp(-x))`.

    Args:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
            or `qint32`.
        name: A name for the operation (optional).

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32`
            otherwise the return type is `quint8`.
    """
    pass


def sigmoid_cross_entropy_with_logits(logits, targets, name=None):
    """
    [TensorFlow Docs]
    Computes sigmoid cross entropy given `logits`.

    Measures the probability error in discrete classification tasks in which each
    class is independent and not mutually exclusive.  For instance, one could
    perform multilabel classification where a picture can contain both an elephant
    and a dog at the same time.

    For brevity, let `x = logits`, `z = targets`.  The logistic loss is

                z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
            = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
            = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
            = (1 - z) * x + log(1 + exp(-x))
            = x - x * z + log(1 + exp(-x))

    For x < 0, to avoid overflow in exp(-x), we reformulate the above

                x - x * z + log(1 + exp(-x))
            = log(exp(x)) - x * z + log(1 + exp(-x))
            = - x * z + log(1 + exp(x))

    Hence, to ensure stability and avoid overflow, the implementation uses this
    equivalent formulation

            max(x, 0) - x * z + log(1 + exp(-abs(x)))

    `logits` and `targets` must have the same type and shape.

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of the same shape as `logits` with the componentwise
        logistic losses.

    Raises:
        ValueError: If `logits` and `targets` do not have the same shape.
    """
    pass


def softmax(logits, name=None):
    """
    [TensorFlow Docs]
    Computes softmax activations.

    For each batch `i` and class `j` we have

            softmax[i, j] = exp(logits[i, j]) / sum(exp(logits[i]))

    Args:
        logits: A `Tensor`. Must be one of the following types: `half`, `float32`, `float64`.
            2-D with shape `[batch_size, num_classes]`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `logits`. Same shape as `logits`.
    """
    pass


def softmax_cross_entropy_with_logits(logits, labels, name=None):
    """
    [TensorFlow Docs]
    Computes softmax cross entropy between `logits` and `labels`.

    Measures the probability error in discrete classification tasks in which the
    classes are mutually exclusive (each entry is in exactly one class).  For
    example, each CIFAR-10 image is labeled with one and only one label: an image
    can be a dog or a truck, but not both.

    **NOTE:**  While the classes are mutually exclusive, their probabilities
    need not be.  All that is required is that each row of `labels` is
    a valid probability distribution.  If they are not, the computation of the
    gradient will be incorrect.

    If using exclusive `labels` (wherein one and only
    one class is true at a time), see `sparse_softmax_cross_entropy_with_logits`.

    **WARNING:** This op expects unscaled logits, since it performs a `softmax`
    on `logits` internally for efficiency.  Do not call this op with the
    output of `softmax`, as it will produce incorrect results.

    `logits` and `labels` must have the same shape `[batch_size, num_classes]`
    and the same dtype (either `float32` or `float64`).

    Args:
        logits: Unscaled log probabilities.
        labels: Each row `labels[i]` must be a valid probability distribution.
        name: A name for the operation (optional).

    Returns:
        A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
        softmax cross entropy loss.
    """
    pass


def softplus(features, name=None):
    """
    [TensorFlow Docs]
    Computes softplus: `log(exp(features) + 1)`.

    Args:
        features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `features`.
    """
    pass


def softsign(features, name=None):
    """
    [TensorFlow Docs]
    Computes softsign: `features / (abs(features) + 1)`.

    Args:
        features: A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`, `half`.
        name: A name for the operation (optional).

    Returns:
        A `Tensor`. Has the same type as `features`.
    """
    pass


"""## Sparse Tensor Representation

Tensorflow supports a `SparseTensor` representation for data that is sparse
in multiple dimensions. Contrast this representation with `IndexedSlices`,
which is efficient for representing tensors that are sparse in their first
dimension, and dense along all other dimensions.

@@SparseTensor
@@SparseTensorValue


@@sparse_to_dense
@@sparse_tensor_to_dense
@@sparse_to_indicator
@@sparse_merge


@@sparse_concat
@@sparse_reorder
@@sparse_split
@@sparse_retain
@@sparse_reset_shape
@@sparse_fill_empty_rows

@@sparse_reduce_sum

@@sparse_add
@@sparse_softmax
@@sparse_tensor_dense_matmul
"""
pass


def sparse_softmax_cross_entropy_with_logits(logits, labels, name=None):
    """
    [TensorFlow Docs]
    Computes sparse softmax cross entropy between `logits` and `labels`.

    Measures the probability error in discrete classification tasks in which the
    classes are mutually exclusive (each entry is in exactly one class).  For
    example, each CIFAR-10 image is labeled with one and only one label: an image
    can be a dog or a truck, but not both.

    **NOTE:**  For this operation, the probability of a given label is considered
    exclusive.  That is, soft classes are not allowed, and the `labels` vector
    must provide a single specific index for the true class for each row of
    `logits` (each minibatch entry).  For soft softmax classification with
    a probability distribution for each entry, see
    `softmax_cross_entropy_with_logits`.

    **WARNING:** This op expects unscaled logits, since it performs a softmax
    on `logits` internally for efficiency.  Do not call this op with the
    output of `softmax`, as it will produce incorrect results.

    `logits` must have the shape `[batch_size, num_classes]`
    and dtype `float32` or `float64`.

    `labels` must have the shape `[batch_size]` and dtype `int32` or `int64`.

    Args:
        logits: Unscaled log probabilities.
        labels: Each entry `labels[i]` must be an index in `[0, num_classes)`. Other
            values will result in a loss of 0, but incorrect gradient computations.
        name: A name for the operation (optional).

    Returns:
        A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the
        softmax cross entropy loss.
    """
    pass


def state_saving_rnn(cell, inputs, state_saver, state_name,
                     sequence_length=None, scope=None):
    """
    [TensorFlow Docs]
    RNN that accepts a state saver for time-truncated RNN calculation.

    Args:
        cell: An instance of `RNNCell`.
        inputs: A length T list of inputs, each a tensor of shape
            `[batch_size, input_size]`.
        state_saver: A state saver object with methods `state` and `save_state`.
        state_name: Python string or tuple of strings.  The name to use with the
            state_saver. If the cell returns tuples of states (i.e.,
            `cell.state_size` is a tuple) then `state_name` should be a tuple of
            strings having the same length as `cell.state_size`.  Otherwise it should
            be a single string.
        sequence_length: (optional) An int32/int64 vector size [batch_size].
            See the documentation for rnn() for more details about sequence_length.
        scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
        A pair (outputs, state) where:
            outputs is a length T list of outputs (one for each input)
            states is the final state

    Raises:
        TypeError: If `cell` is not an instance of RNNCell.
        ValueError: If `inputs` is `None` or an empty list, or if the arity and
     type of `state_name` does not match that of `cell.state_size`.
    """
    pass


def sufficient_statistics(x, axes, shift=None, keep_dims=False, name=None):
    """
    [TensorFlow Docs]
    Calculate the sufficient statistics for the mean and variance of `x`.

    These sufficient statistics are computed using the one pass algorithm on
    an input that's optionally shifted. See:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

    Args:
        x: A `Tensor`.
        axes: Array of ints. Axes along which to compute mean and variance.
        shift: A `Tensor` containing the value by which to shift the data for
            numerical stability, or `None` if no shift is to be performed. A shift
            close to the true mean provides the most numerically stable results.
        keep_dims: produce statistics with the same dimensionality as the input.
        name: Name used to scope the operations that compute the sufficient stats.

    Returns:
        Four `Tensor` objects of the same type as `x`:
        * the count (number of elements to average over).
        * the (possibly shifted) sum of the elements in the array.
        * the (possibly shifted) sum of squares of the elements in the array.
        * the shift by which the mean must be corrected or None if `shift` is None.
    """
    pass


def tanh(x, name=None):
    """
    [TensorFlow Docs]
    Computes hyperbolic tangent of `x` element-wise.

    Args:
        x: A Tensor with type `float`, `double`, `int32`, `complex64`, `int64`,
            or `qint32`.
        name: A name for the operation (optional).

    Returns:
        A Tensor with the same type as `x` if `x.dtype != qint32` otherwise
            the return type is `quint8`.
    """
    pass


"""TensorArray operations.


@@TensorArray
"""
pass

"""Helper classes for tensor shape inference."""
pass

"""Utilities to create TensorProtos."""
pass

"""Contains routines for printing protocol messages in text format.

Simple usage example:

    # Create a proto object and serialize it to a text proto string.
    message = my_proto_pb2.MyMessage(foo='bar')
    text_proto = text_format.MessageToString(message)

    # Parse a text proto string.
    message = text_format.Parse(text_proto, my_proto_pb2.MyMessage())
"""
pass


def top_k(input, k=1, sorted=True, name=None):
    """
    [TensorFlow Docs]
    Finds values and indices of the `k` largest entries for the last dimension.

    If the input is a vector (rank-1), finds the `k` largest entries in the vector
    and outputs their values and indices as vectors.  Thus `values[j]` is the
    `j`-th largest entry in `input`, and its index is `indices[j]`.

    For matrices (resp. higher rank input), computes the top `k` entries in each
    row (resp. vector along the last dimension).  Thus,

            values.shape = indices.shape = input.shape[:-1] + [k]

    If two elements are equal, the lower-index element appears first.

    Args:
        input: 1-D or higher `Tensor` with last dimension at least `k`.
        k: 0-D `int32` `Tensor`.  Number of top elements to look for along the last
            dimension (along each row for matrices).
        sorted: If true the resulting `k` elements will be sorted by the values in
            descending order.
        name: Optional name for the operation.

    Returns:
        values: The `k` largest elements along each last dimensional slice.
        indices: The indices of `values` within the last dimension of `input`.
    """
    pass


def uniform_candidate_sampler(true_classes, num_true, num_sampled, unique,
                              range_max, seed=None, name=None):
    """
    [TensorFlow Docs]
    Samples a set of classes using a uniform base distribution.

    This operation randomly samples a tensor of sampled classes
    (`sampled_candidates`) from the range of integers `[0, range_max)`.

    The elements of `sampled_candidates` are drawn without replacement
    (if `unique=True`) or with replacement (if `unique=False`) from
    the base distribution.

    The base distribution for this operation is the uniform distribution
    over the range of integers `[0, range_max)`.

    In addition, this operation returns tensors `true_expected_count`
    and `sampled_expected_count` representing the number of times each
    of the target classes (`true_classes`) and the sampled
    classes (`sampled_candidates`) is expected to occur in an average
    tensor of sampled classes.  These values correspond to `Q(y|x)`
    defined in [this
    document](http://www.tensorflow.org/extras/candidate_sampling.pdf).
    If `unique=True`, then these are post-rejection probabilities and we
    compute them approximately.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size,
            num_true]`. The target classes.
        num_true: An `int`.  The number of target classes per training example.
        num_sampled: An `int`.  The number of classes to randomly sample per batch.
        unique: A `bool`. Determines whether all sampled classes in a batch are
            unique.
        range_max: An `int`. The number of possible classes.
        seed: An `int`. An operation-specific seed. Default is 0.
        name: A name for the operation (optional).

    Returns:
        sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
            The sampled classes.
        true_expected_count: A tensor of type `float`.  Same shape as
            `true_classes`. The expected counts under the sampling distribution
            of each of `true_classes`.
        sampled_expected_count: A tensor of type `float`. Same shape as
            `sampled_candidates`. The expected counts under the sampling distribution
            of each of `sampled_candidates`.
    """
    pass


"""A class to store named variables and a scope operator to manage sharing."""
pass


def weighted_cross_entropy_with_logits(logits, targets, pos_weight,
                                       name=None):
    """
    [TensorFlow Docs]
    Computes a weighted cross entropy.

    This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
    allows one to trade off recall and precision by up- or down-weighting the
    cost of a positive error relative to a negative error.

    The usual cross-entropy cost is defined as:

        targets * -log(sigmoid(logits)) + (1 - targets) * -log(1 - sigmoid(logits))

    The argument `pos_weight` is used as a multiplier for the positive targets:

        targets * -log(sigmoid(logits)) * pos_weight +
                (1 - targets) * -log(1 - sigmoid(logits))

    For brevity, let `x = logits`, `z = targets`, `q = pos_weight`.
    The loss is:

                qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
            = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
            = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
            = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
            = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
            = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

    Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
    the implementation uses

            (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

    `logits` and `targets` must have the same type and shape.

    Args:
        logits: A `Tensor` of type `float32` or `float64`.
        targets: A `Tensor` of the same type and shape as `logits`.
        pos_weight: A coefficient to use on the positive examples.
        name: A name for the operation (optional).

    Returns:
        A `Tensor` of the same shape as `logits` with the componentwise
        weightedlogistic losses.

    Raises:
        ValueError: If `logits` and `targets` do not have the same shape.
    """
    pass


def xw_plus_b(x, weights, biases, name=None):  # pylint: disable=invalid-name
    """Computes matmul(x, weights) + biases.

    Args:
        x: a 2D tensor.  Dimensions typically: batch, in_units
        weights: a 2D tensor.  Dimensions typically: in_units, out_units
        biases: a 1D tensor.  Dimensions: out_units
        name: A name for the operation (optional).  If not specified
            "xw_plus_b" is used.

    Returns:
        A 2-D Tensor computing matmul(x, weights) + biases.
        Dimensions typically: batch, out_units.
    """
    pass


def xw_plus_b_v1(x, weights, biases, name=None):  # pylint: disable=invalid-name
    """Computes matmul(x, weights) + biases.

    This is a deprecated version of that will soon be removed.

    Args:
        x: a 2D tensor.  Dimensions typically: batch, in_units
        weights: a 2D tensor.  Dimensions typically: in_units, out_units
        biases: a 1D tensor.  Dimensions: out_units
        name: A name for the operation (optional).  If not specified
            "xw_plus_b_v1" is used.

    Returns:
        A 2-D Tensor computing matmul(x, weights) + biases.
        Dimensions typically: batch, out_units.
    """
    pass


def zero_fraction(value, name=None):
    """
    [TensorFlow Docs]
    Returns the fraction of zeros in `value`.

    If `value` is empty, the result is `nan`.

    This is useful in summaries to measure and report sparsity.  For example,

            z = tf.Relu(...)
            summ = tf.scalar_summary('sparsity', tf.nn.zero_fraction(z))

    Args:
        value: A tensor of numeric type.
        name: A name for the operation (optional).

    Returns:
        The fraction of zeros in `value`, with type `float32`.
    """
    pass
