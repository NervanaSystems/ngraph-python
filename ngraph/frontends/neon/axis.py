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
ar = ng.make_name_scope(name="ar")

ar.Height = ng.make_axis_role()
ar.Width = ng.make_axis_role()
ar.Depth = ng.make_axis_role()
ar.Channel = ng.make_axis_role()
ar.Channelout = ng.make_axis_role()
ar.Time = ng.make_axis_role()

ar.FeatureMap_inum = ng.make_axis_role()
ar.FeatureMap_dim0 = ng.make_axis_role()
ar.FeatureMap_dim1 = ng.make_axis_role()
ar.FeatureMap_dim2 = ng.make_axis_role()
ar.FeatureMap_onum = ng.make_axis_role()

# Define the standard Neon axes
ax = ng.make_name_scope(name="ax")

ax.N = ng.make_axis(batch=True, docstring="minibatch size")

ax.C = ng.make_axis(roles=[ar.FeatureMap_inum], docstring="number of input feature maps")
ax.D = ng.make_axis(roles=[ar.FeatureMap_dim0], docstring="input image depth")
ax.H = ng.make_axis(roles=[ar.FeatureMap_dim1], docstring="input image height")
ax.W = ng.make_axis(roles=[ar.FeatureMap_dim2], docstring="input image width")

ax.REC = ng.make_axis(roles=[ar.Time], recurrent=True, docstring="recurrent axis")

ax.T = ng.make_axis(roles=[ar.FeatureMap_dim0], docstring="filter depth")
ax.R = ng.make_axis(roles=[ar.FeatureMap_dim1], docstring="filter height")
ax.S = ng.make_axis(roles=[ar.FeatureMap_dim2], docstring="filter width")
ax.J = ng.make_axis(roles=[ar.FeatureMap_inum], docstring="filter channel size (for crossmap pooling)")
ax.K = ng.make_axis(roles=[ar.FeatureMap_onum], docstring="number of output feature maps")

# ax.M = ng.make_axis(roles=[ar.Depth], docstring="output image depth")
# ax.P = ng.make_axis(roles=[ar.Height], docstring="output image height")
# ax.Q = ng.make_axis(roles=[ar.Width], docstring="output image width")

ax.Y = ng.make_axis(docstring="target")
