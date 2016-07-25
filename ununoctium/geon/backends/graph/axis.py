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
from geon.backends.graph.arrayaxes import AxisVar

# Define the standard Neon axes
N = AxisVar(name='N')
C = AxisVar(name='C')
D = AxisVar(name='D')
H = AxisVar(name='H')
W = AxisVar(name='W')
T = AxisVar(name='T')
R = AxisVar(name='R')
S = AxisVar(name='S')
K = AxisVar(name='K')
M = AxisVar(name='M')
P = AxisVar(name='P')
Q = AxisVar(name='Q')

# Target
Y = AxisVar(name='Y')

# Dataloader phase
Phi = AxisVar(name='Phi')
