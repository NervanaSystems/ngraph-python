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
from ngraph.op_graph.axes import Axis
from ngraph.util.names import NameScope


ax = NameScope(name="ax")

# Define the standard Neon axes

# N = number of images (minibatch size)
ax.N = Axis(batch=True)

# C = number of input channels
ax.C = Axis()

# D = depth
ax.D = Axis()

# H = input image height
ax.H = Axis()

# W = input image width
ax.W = Axis()

# TODO This isn't one of the traditional axes
# Recurrent axis
ax.REC = Axis(recurrent=True)

# R = filter height
ax.R = Axis()

# S = filter width
ax.S = Axis()

# T = filter depth
ax.T = Axis()

# K = number of output channels
ax.K = Axis()

# M = output image depth
ax.M = Axis()

# P = output image height
ax.P = Axis()

# Q = output image width
ax.Q = Axis()

# Target
ax.Y = Axis()
