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


class Error(Exception):
    """
    Base class for graph errors.
    """
    pass


class MissingGraphError(Error):
    """
    Graph cannot be determined.
    """


class UnititializedVariableError(Error):
    """
    Attempt to use the value of an unitialized variable.
    """


class IncompatibleShapesError(Error):
    """
    Incompatible shapes.
    """


class IncompatibleTypesError(Error):
    """
    Incompatible graph types.
    """


class RedefiningConstantError(Error):
    """
    Redefining a constant value.
    """


class NameException(Exception):
    """
    Trying to set a value in a name generator.
    """
    pass
