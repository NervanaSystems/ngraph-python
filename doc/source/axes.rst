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

Axes
****

Note: This describes the next revision of tensor descriptions and axes.





A |geon| graph describes a computation. In the graph, *tensor descriptions* describe the tensors that take part in the computation. All Ops that supply tensor values have a tensor description that describes the kind of tensor computed. Generally, the tensor description of an Op is determined from a combination of the arguments and other parameters supplied to the Op. In many cases, |geon| can use the tensor descriptions to automatically insert Ops into the graph to adapt the tensor that would be produced by an Op into the format required by another Op.

Tensor Semantics
================

The specific semantics of a tensor varies from frontend to frontend, while the actual implementation of a tensor is dependent on the backend. |Geon| provides a generalized view of tensor semantics and implementation that frontends and backends translate as appopriate.

|Geon| treats a tensor semantics as a view of storage. In normal use, the view and the storage are created simultaneously. Additional views of the storage can be created with slicing and reshaping operation. In order to model and manipulate the computation, |geon| tensor descriptions need to model the relation between the view and the storage. A view has a dtype, shape, strides, offset, and storage. Indices, which must be between 0 and the shape, are mapped to storage offsets by adding the offset to the dot product of the index and the strides.

The |geon| model of tensor implementation is similar, except that the views refer to a reference to storage. The reference to storage can be changed by the runtime, to support double buffering device I/O. Transformers can implement setting the reference by updating the buffer pointers when the reference changes.

Low-level Tensor Descriptions
=============================

Low-level tensor descriptions are close to the tensor implementation. The ``LowBufferDescription`` has a type and size, while the ``LowTensorDescription`` has a type, dimensions, shape, strides, buffer, and offset. The strides of a ``LowTensorDescription`` are in terms of elements since the alignment constraints for types may depend on the backend, and in a heterogeneous environment their may be mixed constraints.

Since semantically distinct tensors may share the same actual storage when the need for their values do not overlap, the representation of device tensors is slightly different. A ``DeviceBufferStorage`` has an element type and a size.  A ``DeviceBufferReference`` has an element type and a minimum size. It can be associated with any ``DeviceBufferStorage`` of the same type thay meets the minimum size constraint. A ``DeviceBufferReference`` is associated with one or more ``LowBufferDescription``s. For each required dim/shape/stride/offset of the ``LowTensorDescription``s associated with the ``LowBufferDescription`` there is a ``DeviceTensor``. The ``DeviceTensor``, ``DeviceBufferReference`` and ``DeviceBufferStorage`` are subclassed by transformers and serve as runtime handles to actual device storage.

Low-level operations only need low-level tensor descriptions, and may have additional constraints on dimensions and strides.

Tensor Descriptions with Positional Axes
========================================

Positional axes are an extension to the low-level tensor descriptions to simplify the implementation of Ops that work with tensors of arbitrary dimension and layout.

Elementwise positional axes operations match the axes of their arguments and broadcast on non-matching axes, so the result has axes of the union of the axes, optionally extended with the axes parameter. In order to make the result axes predictable, the order of the result axes is the order of the appended axes list from each argument, with duplicates removed from left to right, although the order can be overridden with the axes parameter.

The dot operation is a little more complicated. In mathematics, a tensor is an array of scalar functions of an array. If the functions are all multilinear, the tensor is fully characterized by an array of scalars of dimension the sum of the input and output array dimensions. The space of functions from :math:`H\rightarrow R` is denoted :math:`H^*`. For a mathematical tensor :math:`T:X\rightarrow Y` represented as an array :math:`T`, and a value :math:`x\in X`, :math:`dot(T, x) = y\in Y` applies the tensor to :math:`x`.  The array :math:`T` will have axes associated with :math:`X` and :math:`Y`.  The axes associated with :math:`X` in :math:`T` produces a scalar value for :math:`x\in X`, so these are in the space :math:`X^*`.  In general, :math:`M\times N^*\ne M^* \times N^*`, but the relation does hold if we restrict ourselves to multilinear functions.

We define ``dot`` so that if an axis :math:`M^*` is in the first argument and :math:`M` in the second argument, these are paired for multiplication and reduction, and the result has the remaining axes from the two arguments.  There must be no duplicates among the remaining axes.  Sometimes we have the situation where we have a :math:`M` axis in the first argument that we want to reduce from the right. We do this with an "anti-star" axis, namely :math:`M^{-*}`.

This brings us to the transpose operation. It changes the axis :math:`M^{k*}` to :math:`M^{(1-k)*}` but leaves values the same. Then :math:`dot(x,y)=dot(y^t,x^t)^t`.