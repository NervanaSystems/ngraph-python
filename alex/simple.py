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

import neon.alex.nodes as g

def linear_regression():
    x = g.data_tensor((10), name='x')
    y = g.data_tensor((1), name='y')

    w = g.variable_tensor((1,10))
    #w.initial_value(0)
    b = g.variable_tensor((10))
    #b.initial_value(0)

    y0 = w*x+b
    e = g.norm2(y0-y)

    vars = [v for v in e.variables()]
    d = y0.diff(vars)

    print(d)



linear_regression()
