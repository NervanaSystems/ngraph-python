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

Tensor Types
============

Every op-graph operation has associated type information for its value, roughly mirrored in the Python types of the implementing classes. Most ops are tensor-valued, but there are several varieties of tensors. Informally, an n-tensor associates each element of an n-rectangle of integers with a value. It is fairly common for the same value to be referenced by multiple tensors.

In most programming languages, there are r-values and l-values. r-values can be used on the right side of an assignment, and l-values can be used on the left side of an assignment. A variable can be an l-value or an r-value, while constants and most arithmetic expressions can only be r-values. R-values are transient; once they have been used, they are no longer needed. L-values can be transient, but more work is needed to determine their lifetime.

Like variables, there are l-tensors and r-rensors. The values of an l-tensor can be assigned, while the values of an r-tensor are only used. For example, if :math:`a` and :math:`b` are compatible tensors, :math:`a-b` is a r-tensor. If we only want to compute :math:`||a-b||^2` we can compute it without storing the tensor :math:`a-b` by computing :math:`s s s = s + (a_i-b_i)^2` for each element.

Tensor libraries often execute one operation at a time, so :math:`a-b` would be computed and stored in a new tensor. Then its transpose would be taken, which would be an aliased view of the same elements, and then the dot product of the two tensors would be computed. At this point, nothing references the transpose tensor, so the garbage collector reclaims its storage. The transpose was the only reference to the difference tensor, so once the transpose is reclaimed, the difference can be reclaimed. In the process, memory caches were filled with the elements of :math:`a-b` that no longer exist, while values that might still be useful were decached to make room the :math:`a-b`.

The tensor type information helps us automatically determine when intermediate results need to be stored, and what the best storage representation for them is based on how they are used and the processing environment. Tensor type information also simplifies the process of converting more abstract representations of computations into versions that can be executed.

Basic Tensors
*************

Recall that an n-tensor is a map from index elements of an n-rectangle to values. When implementing tensor operations in terms of scalar operations, we associated an n-tuple of integers called the *strides* with the tensor. We compute the dot product of the strides and the index, add an offset, and get the address of the element. If we are being careful, we also associate an n-tuple called the *shape* with the tensor and check that the index is non-negative and less than the shape.

If we permute both arguments to a dot product in the same way, the product does not change. Thus, if we want to use a different index order for the tensor, such as a transpose, we can make a tensor with the same offset and the strides permuted. When we are analyzing computations, we need to know that these two tensors are really views of the same values.

Mapping an n-index to an element of an n-tensor is pretty easy. Now consider subtracting two n-tensors that have the same shape. If you just need values at particular indices, you just access the values at the two tensors, subtract their values, and you have the result; this is the r-tensor of the difference. But if you want to compute all the values, storing them in an l-tensor, you need to enumerate all the indices. For one dimension, this is easy, for two it is not hard, but doing it for an arbitrary number of dimensions makes the index enumeration overhead eclipse the actual computation. But, as each index is computed, we perform the dot product with the strides for the two tensors. In many cases, the strides are such that we can make two new tensor views with fewer dimensions and iterate over simpler indices. We call this *flattening*. The result has fewer dimensions, so we need to *unflatten* the result by creating a view with the original dimensions and an appropriate striding. Our type system can record that our lower-dimensional tensor is a flattened tensor, so that the unflattening parameters can be computed automatically.

For these cases, we are assuming that the storage layout for the tensor is known, as would be the case for primitive ops, or when an explicit layout has been specified. Now lets consider the case where we do not have a storage layout, and two tensors are supposed to be aliases of each other. We can pick an arbitrary layout and proceed in the same way as before. When we assign an actual layout, we can restride the views. It may turn out that some views, such as flattens, cannot be restrided because the enumeration order of the flattened dimensions would change. If we know all the ops the flattened tensor is used with, we might be able to compatibly assign strides to all the tensors. If we know the affected tensor is an r-tensor, we can replace it with a shuffled copy. If an l-value, we need to replace assignments to it with appropriately shuffled assignments.

We can characterize these tensors by shape, stride (implicit from shape and stride), offset, storage, and whether the striding is fixed. This is the only information needed by basic operations.

Axes
****

Rather than specifying the shape of a tensor, we can define axes for the tensors. If tensor :math:`a` has axes :math:`(C, H, W, N)` and we want to subtract tensor :math:`b` with axes :math:`(W,H)`, we know that the result will have rank 4 and that we need to switch :math:`(W,H)` to :math:`(H,W)`. With shapes, the user would need to explicitly tell us that the axes had to be swapped, and would need to explicitly reshape for proper broadcasting.





