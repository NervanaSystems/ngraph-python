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
import numpy as np
from functools import partial


class GaussianInit(object):
    def __init__(self, mean=0.0, var=0.01):
        self.functor = partial(np.random.normal, mean, var)

    def __call__(self, out_axes):
        return self.functor(out_axes.lengths)


class UniformInit(object):
    def __init__(self, low=-0.01, high=0.01):
        self.functor = partial(np.random.uniform, low, high)

    def __call__(self, out_axes):
        return self.functor(out_axes.lengths)


class ConstantInit(object):
    def __init__(self, val=0.0):
        self.val = val

    def __call__(self, out_axes):
        return self.val
