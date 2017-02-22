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
from weakref import WeakValueDictionary
from builtins import object


class NameableValue(object):
    """
    An object that can be named.

    Arguments:
        graph_label_type: A label that should be used when drawing the graph.  Defaults to
            the class name.
        name (str): The name of the object.
        **kwargs: Parameters for related classes.

    Attributes:
        graph_label_type: A label that should be used when drawing the graph.
        id: Unique id for this object.
    """
    __counter = 0
    __all_names = WeakValueDictionary()

    def __init__(self, name=None, graph_label_type=None, docstring=None, **kwargs):
        super(NameableValue, self).__init__(**kwargs)

        if name is None:
            name = type(self).__name__
        self.name = name

        if graph_label_type is None:
            graph_label_type = self.name
        self.graph_label_type = graph_label_type

        self.__doc__ = docstring

    @property
    def graph_label(self):
        """The label used for drawings of the graph."""
        return "{}[{}]".format(self.graph_label_type, self.name)

    @property
    def name(self):
        """The name."""
        return self.__name

    @name.setter
    def name(self, name):
        """
        Sets the object name to a unique name based on name.

        Arguments:
            name: Prefix for the name
        """
        if name == type(self).__name__ or name in NameableValue.__all_names:
            while True:
                c_name = "{}_{}".format(name, type(self).__counter)
                if c_name not in NameableValue.__all_names:
                    name = c_name
                    break
                type(self).__counter += 1
        NameableValue.__all_names[name] = self
        self.__name = name

    @property
    def short_name(self):
        sn = self.name.split('_')[0]
        if sn.find('.') != -1:
            sn = sn.split('.')[1]
        return sn

    def named(self, name):
        self.name = name
        return self
