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

Tensor Descriptions
===================

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

Every tensor-valued ``Op`` corresponds to an r-tensor and has a ``tensor_description`` attribute that describes the tensor, and is computed from the tensor descriptions of the parameters and arguments to the ``Op``. During the transformation process, the tensor description may be augmented with additional information, such as a storage layout and storage assignment. The value of an ``Op`` might be a different view of a tensor, in which case the sharing must be indicated in its ``tensor_description``. An ``AllocationOp`` is a special case of a tensor-valued ``Op`` in that its tensors is an l-tensor. At the end of the transformation process, all tensor descriptions for l-tensors must contain enough information for them to be allocated.

Dense L-Tensor Implementation
*****************************

Recall that the map for a tensor has two components, a map from an index to an integer, and a map from an integer to a value. The second map is provided by storage. For the first map, we associate an n-tuple of integers, called the *strides*, with the tensor. We compute the dot product of the strides and the index and add an offset to get the index into storage. If we are being careful, we also associate an n-tuple called the *shape* with the tensor and check that the index is non-negative and less than the shape.

If we permute both arguments to a dot product in the same way, the product does not change. Thus, if we want to use a different index order for the tensor, such as a transpose, we can make a tensor with the same offset and the strides permuted. When we are analyzing computations, we need to know that these two tensors are really views of the same values.

Mapping an n-index to an element of an n-tensor is pretty easy. Now consider subtracting two n-tensors that have the same shape. If you just need values at particular indices, you just access the values at the two tensors, subtract their values, and you have the result; this is the r-tensor of the difference. But if you want to compute all the values, storing them in an l-tensor, you need to enumerate all the indices. For one dimension, this is easy, for two it is not hard, but doing it for an arbitrary number of dimensions makes the index enumeration overhead eclipse the actual computation. But, as each index is computed, we perform the dot product with the strides for the two tensors. In many cases, the strides are such that we can make two new tensor views with fewer dimensions and iterate over simpler indices. We call this *flattening*. The result has fewer dimensions, so we need to *unflatten* the result by creating a view with the original dimensions and an appropriate striding. Our type system can record that our lower-dimensional tensor is a flattened tensor, so that the unflattening parameters can be computed automatically.

For these cases, we are assuming that the storage layout for the tensor is known, as would be the case for primitive ops, or when an explicit layout has been specified. Now lets consider the case where we do not have a storage layout, and two tensors are supposed to be aliases of each other. We can pick an arbitrary layout and proceed in the same way as before. When we assign an actual layout, we can restride the views. It may turn out that some views, such as flattens, cannot be restrided because the enumeration order of the flattened dimensions would change. If we know all the ops the flattened tensor is used with, we might be able to compatibly assign strides to all the tensors. If we know the affected tensor is an r-tensor, we can replace it with a shuffled copy. If an l-value, we need to replace assignments to it with appropriately shuffled assignments.

We can characterize these tensors by shape, stride (implicit from shape and stride), offset, storage, and whether the striding is fixed. This is the only information needed by basic operations.

Axes
****

Rather than specifying the shape of a tensor, we can define axes for the tensors. If tensor :math:`a` has axes :math:`(C, H, W, N)` and we want to subtract tensor :math:`b` with axes :math:`(W,H)`, we know that the result will have rank 4 and that we need to switch :math:`(W,H)` to :math:`(H,W)`. With shapes, the user would need to explicitly tell us that the axes had to be swapped, and would need to explicitly reshape for proper broadcasting.





