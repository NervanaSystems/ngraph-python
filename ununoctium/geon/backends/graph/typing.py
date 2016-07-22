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
from builtins import object
import numpy as np
from future.utils import with_metaclass


class TypeDescriptorMetaType(type):

    def __new__(cls, name, parents, attrs):
        return super(
            TypeDescriptorMetaType,
            cls).__new__(
            cls,
            name,
            parents,
            attrs)

    def __call__(cls, *args, **kargs):
        return super(TypeDescriptorMetaType, cls).__call__(*args, **kargs)

    def __getitem__(cls, args):
        if not isinstance(args, tuple):
            args = tuple((args,))
        return super(TypeDescriptorMetaType, cls).__call__(*args)


class TypeDescriptor(with_metaclass(TypeDescriptorMetaType, object)):

    def __init__(self, **kargs):
        super(TypeDescriptor, self).__init__(**kargs)


class Void(TypeDescriptor):

    def __init__(self, **kargs):
        super(Void, self).__init__(**kargs)

    def __repr__(self):
        return 'Void[]'


class DType(TypeDescriptor):

    def __init__(self, dt=np.float32, **kargs):
        super(DType, self).__init__(**kargs)
        self.__dtype = np.dtype(dt)

    @property
    def type(self):
        return self.__dtype.type

    @property
    def dtype(self):
        return self.__dtype

    def __repr__(self):
        return 'DType[{dtype}]'.format(dtype=self.dtype)


class Axes(TypeDescriptor):

    def __init__(self, *axes, **kargs):
        super(Axes, self).__init__(**kargs)
        self.axes = axes

    def __repr__(self):
        return 'AxesType{axes}'.format(axes=self.axes)


class Array(TypeDescriptor):

    def __init__(self, axes, dtype=np.float32, **kargs):
        super(Array, self).__init__(**kargs)
        if dtype is None:
            pass
        self.dtype = dtype
        self.axes = axes

    def __repr__(self):
        return 'ArrayType[{dtype},{axes}]'.format(
            dtype=self.type, axes=self.axes)


class Tuple(TypeDescriptor):

    def __init__(self, dtype, elt_types, **kargs):
        super(Tuple, self).__init__(**kargs)
        self.elt_type = elt_types

    def __repr__(self):
        return 'Tuple{elt_types}'.format(elt_types=self.elt_types)
