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
from functools import wraps
import ngraph as ng


def shape_to_axes(shape):
    return [ng.make_axis(s) for s in shape] if shape else ng.make_axis()


def args_shape_to_axes(pos):
    """
    Decorator to convert shape to axes.

    Arguments:
        pos: Ordinal position of shape in args tuple

    """
    def outer(func):
        @wraps(func)
        def inner(*args, **kw):
            if pos is None:
                return func(*args, **kw)
            shape = args[pos]
            axes = shape_to_axes(shape)
            temp = list(args)
            temp[pos] = axes
            args = tuple(temp)
            return func(*args, **kw)
        return inner
    return outer


@args_shape_to_axes(1)
def make_const_op(const, axes=None, name=None):
    return ng.constant(const, axes).named(name)
