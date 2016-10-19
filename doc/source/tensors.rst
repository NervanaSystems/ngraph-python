.. ---------------------------------------------------------------------------
.. Copyright 2016 Nervana Systems Inc.
.. Licensed under the Apache License, Version 2.0 (the "License");
.. you may not use this file except in compliance with the License.
.. You may obtain a copy of the License at
..
..      http://www.apache.org/licenses/LICENSE-2.0
..
.. Unless required by applicable law or agreed to in writing, software
.. distributed under the License is distributed on an "AS IS" BASIS,
.. WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
.. See the License for the specific language governing permissions and
.. limitations under the License.
.. ---------------------------------------------------------------------------

Tensors
=======

Abstractly, an n-tensor is a map from an n-rectangle of non-negative integers to values of homogenous type. In programming languages, there are two kinds of values, l-values and r-values. L-values can appear on the left side of an assignment and r-values can appear on the right side of an assignment. For example, ``x`` can be an l-value or an r-value, while ``x + y`` is an r-value. Likewise, if a tensor's values are l-values, the tensor is an l-tensor, and if the tensor's values are r-values, the tensors is an r-tensor. The tensor ``x`` is an l-tensor since values can be assigned to its elements, as in ``x[...] = y``, while the tensor ``x + y`` is an r-tensor because values cannot be assigned to it. An r-tensor only needs to be able to provide values; it does not need to store them. The tensor ``x + y`` could produce the value for an index ``i`` by providing ``x[i] + y[i]`` every time it is needed, and a constant tensor could ignore the index and always produce the value.

Tensor ``y`` is a *simple view* of tensor ``x`` if there is some index translation function ``f`` such that ``y[i] == x[f(i)]`` for all valid indices of ``y`` and all values of ``x``. Reshaping and slicing are two tensor operations that create views. A  *complex view* involves multiple tensors and index translation functions. If ``x[]`` is a list of tensors, and ``f[]`` a list of index translation functions, and there is a selection function ``s`` such that after setting ``y[i] == x[s(i)][f[s(i)](i)]`` for all valid indices ``i`` of ``y`` and all values of ``x[...]``. In this case, ``s`` selects a tensor and index translation function for that tensor. Padding is an example of a complex view, in which a region of the values come from some other tensor, and the remaining values come from zero tensors. Transformers introduce views as they rewrite more abstract operations as simpler operations available on backends.

We can convert an r-tensor to an l-tensor by allocating an l-tensor for it initializing it with the r-values. If the values at particular indices are going to be used multiple times, this can reduce computation. Not all r-tensors should be converted to l-tensors. If  ``x`` and ``y`` are compatible 1-tensors, ``x - y`` is a r-tensor. If we only want to compute the L2 norm of ``x - y`` we could use MumPy to compute

.. code-block:: python

    def L2(x, y):
        t = x - y
        return np.dot(t.T, t)


NumPy will allocate a tensor for ``t``. Every element in ``x``, ``y`` and ``t`` will be touched once, and pages in ``t`` will be modified. As each page is accessed, it moves to the end of the line for pages to be evicted. Furthermore, the accessing all the elements of ``x``, ``y``, and ``t`` will displace many elements from memory caches. Next, a view of ``t`` for ``t.T`` is allocated by NumPy. Views are tiny compared to tensors. Computing the dot product will access every element of ``t`` again. If ``t`` is larger than the memory cache, the recently cached elements near the end of ``t`` will be evicted so the ones near the beginning of ``t`` can be accessed. When the function returns, the garbage collector would see that the view ``t.T`` was no longer referenced and reclaim it. Since ``t.T`` is the only reference to ``t``, the garbage collector can now reclaim ``t``. All the cache locations displaced by ``t`` are now unused. Furthermore, even though ``t`` is unallocated memory according the the heap, paging still sees it as modified pages. The page will need to be written back to paging before the physical memory can be given to other virtual memory. Likewise, the memory caches see the memory as modified and will need to invalidate caches for other cores.

Compare this with the following function,

.. code-block:: python

    def L2(x, y):
        s = 0
        for i in len(x):
            s = s + (x[i] - y[i])^2
        return s

As in the previous function, ``x`` and ``y`` will need to enter the cache, but there us no ``t`` to be cached, to ``t.T`` and ``t`` to be freed by the garbage collector, and no dirty pages to evict.

Dense L-Tensor Implementation
*****************************

We factor the map from index to value into two components, a map from the index to a non-negative integer, the linear index, and a map from the linear indices to values. Essentially, every n-d tensor is a view of a 1-d linear tensor. The linear tensor is just memory; an l-value is the base address plus the index, adjusted for element size, and the r-value is the contents of the l-value. If the linear tensor has a size, bounds checking may be performed. The n-d map is characterized by an n-tuple of integers, called the stride, an an offset. The offset is added to the dot product of the strides and index to get the linear index. If the linear tensor also has an n-tuple of integers called the shape, bounds checking may be performed on the index. Sometimes it is important to align elements on particular memory boundaries. In this case, in addition to a shape there an n-tuple called the size, which is greater than or equal to the shape to add padding for alignment.

There are many ways to map an index to a linear index. Two of the most common are Row-major and column-major order. In row-major order, the strides are partial products of the allocated sizes for each dimension, multiplied from the right, while for column-major order, from the left. For example, if the sizes of the dimensions of a 3d-tensor are ``(5, 3, 2)`` then the row-major strides would be ``(6, 2, 1)``, and ``(1, 5, 15)`` for column major-order. Note that if two elements of the stride, shape, and size are permuted, then the same linear index is given by permuting the index in the same way. For example, a transpose view just requires these permutations.

Views allow for simpler implementation of tensor operations. For example consider implementing a subtraction operation for arbitrary n-tensors of the same shape. Implemented directory, an n-tuple index iterator would need to be maintained. However, if the n-tuple iterator would iterate over the linearized indices in the same order for both tensors, we can *flatten* the tensors into two 1-d tensor views and use a simple integer iterator. This may result in the elements being accessed in a different order than they would have been with multi-dimensional indices. It is important that the tensors have the same layout. If the strides on one tensor were ``(1, 2)`` and ``(2, 1)`` on the other, the elements would be paired as

.. code-block:: python

    x[0, 0] - y[0, 0]
    x[1, 0] - y[0, 1]
    x[0, 1] - y[1, 0]
    x[1, 1] - y[1, 1]

Basic Tensor Descriptions
*************************

Basic tensor descriptions only have shape and element type information. Although the shape is an ordered list of lengths, the order does not imply a particular layout for the elements. The basic tensor descriptions, with restrictions on dimensions and striding, are appropriate for the basic operations that all transformers must implemenet. The may also be useful for front ends that describe tensors by shape.

If we know the layout of a tensor, we can compute what the layout of slices and reshapings. But in the graph, we only know the layout for those tensors where the layout has been explicitly provided. But we still need information about which tensors are views of each other, dimension lengths, alignment constraints, slicing, etc. We use ``BasicTensorDescription`` to represent all the information the graph needs to know about tensors. During the transformation process, this may vary. When a tensor is first added to the graph, little may be known about it, but by the time execution occurs, layout needs to be known.

``BasicTensorDescription``:
    Describes a tensor by its shape and element type.

    Attributes:
        dtype: The dtype of the elements.

        rank: The number of dimensions.

        read_only: True for an r-tensor, False for an l-tensor.

        shape: An n-tuple of non-negative integers. The length of the tuple is the rank.

        layout: strides and offset, if known.

``SimpleTensorViewDescription(BasicTensorDescription)``:
    Common information for all simple views.

    Attributes:
        tensor: The viewed tensor.

``BroadcastTensorDescription(SimpleTensorViewDescription)``:
    Add broadcast dimensions to the viewed tensor.

    Parameters:
        broadcast_shape: The shape of the view, with 1s denoting broadcast dimensions. The shape of tensor with 1s removed must be the same as this shape with 1s removed.

``FlattenTensorDescription(SimpleTensorViewDescription)``:
    Flatten two or more axes.

    Attributes:
        shape: The shape of the view, where sub-tuples indicated flattened dimensions. For example, ``((32, 32), 128)`` flattens the first two dimensions of ``(32, 32, 128)``. The shape with sub-tuple lengths promoted to the tuple must match the shape of the viewed tensor.

``PermuteTensorDescription(SimpleTensorViewDescription)``:
    Permute two or more axes.

    Attributes:
        permutation: A tuple of the viewed tensor's dimensions in the view. For example, a permutation of ``(1, 2, 0)`` of a tensor with shape ``(2, 3, 5)`` would have shape ``(3, 5, 2)``.

``SliceTensorDescription(SimpleTensorViewDescription)``:
    Slice one or more dimensions.

    Parameters:
        slices: A tuple of slices of the viewed tensor. Must be the same number of dimensions as tensor and contain slices or dimension lengths.

``PadTensorDescription(SimpleTensorViewDescription)``:
    Add padding to one or more dimensions.

    Attributes:
        pre_padding: n-tuple of zero padding added before each dimension.

        post_padding: n-tuple of zero padding added after each dimension

``ComplexTensorViewDescription(BasicTensorDescription)``:
    Describes a complex view of a tensor, i.e. one composed of multiple tensors.  TBD.

Every basic tensor-valued ``Op`` corresponds to an r-tensor (if an allocation, an l-tensor) and has a ``BasicTensorDescription`` describes the tensor, and is computed from the tensor descriptions of the parameters and arguments to the ``Op``.
During the transformation process, the tensor description may be augmented with additional information, such as a storage layout and storage assignment. The value of an ``Op`` might be a different view of a tensor, in which case the sharing must be indicated in its ``tensor_description``. An ``AllocationOp`` is a special case of a tensor-valued ``Op`` in that its tensors is an l-tensor. At the end of the transformation process, all tensor descriptions for l-tensors must contain enough information for them to be allocated.

Axes
****

Axes provide a way to add semantic information about a tensor's dimensions. For example, rather than a tensor having a shape of ``(32, 32)`` we can say it has axes of ``(W, H)``. If one tensor has axes of ``(W, H)`` and another has axes of ``(H, W)`` and we add them, the semantic information tells us that we need to swap the axis order, as written (the chosen layouts may be such that no swapping is actually needed). Axes also simplify broadcasting; if adding a ``(W, H)`` tensor to a ``(W, H, N)`` tensor, we can infer that the first axis should broadcast on the ``N`` axis. This results in a broadcast axis with the same axis class as ``N``.

In a network, the semantics such as "Height," or "Channels" or "Hidden" may apply to dimensions of different lengths in different dimensions. We call these designations the *axis class*. When an axis is created, an axis class may be supplied; if a class is not supplied, a uniqe class is created for the axis.

Elementwise operations match axes by identity. If there is not an identity match, but two axes are of the same class and length, they will match. Otherwise, broadcasting will be used to make the axes the same.

Convolution is more complicated. The filter moves over some axes to form dot products on other axes. The axis classes of the filter and input should match according to the dot product. The filter has output axes, whose classes should match the classes of the output. If the output axes default, they will be generated and use the appropriate classes.

For dot, we associate a partial order with axes; every normal axis has an offset of 0, but we can obtain a related offset axis that is offset by any integer. In the dot product, axes in the first element will match axes in the second axis with an offset one higher. The transpose operation on a tensor makes a view where the axis offsets are all subtracted from -1. This makes ``dot(x, y) = dot(y.T, x.T).T`` hold, and ``dot(x.T, x)`` is the L2 norm for any tensor ``x``.

``TensorDescription(BasicTensorDescription)``:
    Extends a tensor description to have axes. The shape comes from the length of the axes.

    Attributes:
        axes: The axes of the tensor.




