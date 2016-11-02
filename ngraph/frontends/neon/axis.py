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
import ngraph as ng

# Define axis roles
ar = ng.makeNameScope(name="ar")

ar.Height = ng.makeAxisRole()
ar.Width = ng.makeAxisRole()
ar.Depth = ng.makeAxisRole()
ar.Channels = ng.makeAxisRole()

# Define the standard Neon axes
ax = ng.makeNameScope(name="ax")

ax.N = ng.makeAxis(batch=True, docstring="minibatch size")

ax.C = ng.makeAxis(roles=[ar.Channels], docstring="number of input channels")
ax.D = ng.makeAxis(roles=[ar.Depth], docstring="input image depth")
ax.H = ng.makeAxis(roles=[ar.Height], docstring="input image height")
ax.W = ng.makeAxis(roles=[ar.Width], docstring="input image width")

ax.REC = ng.makeAxis(recurrent=True, docstring="recurrent axis")

ax.R = ng.makeAxis(roles=[ar.Height], docstring="filter height")
ax.S = ng.makeAxis(roles=[ar.Width], docstring="filter width")
ax.T = ng.makeAxis(roles=[ar.Depth], docstring="filter depth")

ax.K = ng.makeAxis(roles=[ar.Channels], docstring="number of output channels")
ax.M = ng.makeAxis(roles=[ar.Depth], docstring="output image depth")
ax.P = ng.makeAxis(roles=[ar.Height], docstring="output image height")
ax.Q = ng.makeAxis(roles=[ar.Width], docstring="output image width")

ax.Y = ng.makeAxis(docstring="target")
