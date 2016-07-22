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
import geon.backends.graph.funs as be
from neon.util.argparser import NeonArgparser
from neon.data import ImageLoader
from neon.initializers import Uniform
import geon.backends.graph.axis as ax

from geon.backends.graph.layer import *
from geon.backends.graph.optimizer import *
from geon.backends.graph.cost import CrossEntropyBinary, CrossEntropyMulti, SumSquared, \
    Misclassification
from geon.backends.graph.activation import Rectlin, Identity, Explin, Normalizer, Softmax, Tanh, \
    Logistic
from geon.backends.graph.model import Model
from geon.backends.graph.optimizer import GradientDescentMomentum
from geon.backends.graph.arrayaxes import Axes

from geon.backends.graph.callbacks import *
from neon.optimizers.optimizer import Schedule, StepSchedule, PowerSchedule, ExpSchedule, \
    PolySchedule
