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

# commonly used modules.  Should these still be imported in neon frontend?
from ngraph import make_axes
from ngraph.frontends.neon.axis import ax
from ngraph.frontends.neon.activation import Rectlin, Rectlinclip, Identity, Explin, Normalizer, Softmax, Tanh, \
    Logistic
from ngraph.frontends.neon.argparser import NgraphArgparser
from ngraph.frontends.neon.arrayiterator import *
from ngraph.frontends.neon.callbacks import *
# from ngraph.frontends.neon.callbacks2 import *
from ngraph.frontends.neon.layer import *
from ngraph.frontends.neon.model import *
from ngraph.frontends.neon.optimizer import *
from ngraph.frontends.neon.initializer import *
