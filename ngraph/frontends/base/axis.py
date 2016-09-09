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
from ngraph import Axis

# Define the standard Neon axes
#: The batch axis
N = Axis(name='N', batch=True)
C = Axis(name='C')
D = Axis(name='D')
H = Axis(name='H')
W = Axis(name='W')
T = Axis(name='T', recurrent=True)
R = Axis(name='R')
S = Axis(name='S')
K = Axis(name='K')
M = Axis(name='M')
P = Axis(name='P')
Q = Axis(name='Q')

# Target
Y = Axis(name='Y')
