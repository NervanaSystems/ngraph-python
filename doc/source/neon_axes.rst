
.. ---------------------------------------------------------------------------
.. Copyright 2016-2018 Intel Corporation
..
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

.. Note::
   The API for axis type specification is still heavily under development. As such, this document represents both the current state of affairs and where the API is heading.

Neon extends the implementation of ``Axes`` in Intel Nervana Graph by introducing the concept of axis types. Axis types allow the frontend to make assumptions about tensor dimensions that are not as strict as requiring a specific axis ordering in order to remove some of the specification burden from the user. Different layers require different sets of axis types from their inputs. For instance, a ``Convolution`` layer can only operate on inputs that contain 1 ``channel axis``, 1 to 3 ``spatial axes``, and 1 ``batch axis``. Similarly, an unrolled RNN can only operate on inputs that contain 1 ``recurrent axis``. Each axis type has a default name or set of names that can be used that are listed below. These default values can be overridden during a layer's ``__call__`` method, making it easy to use axis names that best fit the type of data being processed by the network.

Axis Types
----------

- ``recurrent_axes``: The default name for all recurrent axes is "REC". Recurrent layers can currently only operate over a single recurrent axis, though this restriction may be lifted in the future.
- ``spatial_axes``: Spatial axes currently support up to three dimensions.
    - ``height``: The default name for the height axis is "H".
    - ``width``: The default name for the width axis is "W".
    - ``depth``: The default name for the depth axis is "D".
- ``channel_axes``: The default name for the channel axis is "C". Convolutional layers can currently only operate over a single channel axis, though this restriction may be lifted in the future.
- ``batch_axes``: The default name for the batch axis is "N". Currently the batch axis cannot be overridden.
