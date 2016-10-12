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

import collections


class OrderedSet(set, collections.MutableSequence):
    """
    A set that iterates over its elements in the order they were added.

    Arguments:
        values: A sequence of initial values.
    """
    def __init__(self, values=None):
        self.elt_list = []
        if values is not None:
            if isinstance(values, collections.Sequence):
                self.update(values)
            else:
                raise ValueError()

    def __iter__(self):
        return self.elt_list.__iter__()

    def __reversed__(self):
        return self.elt_list.__reversed__()

    def __len__(self):
        return self.elt_list.__len__()

    def __getitem__(self, index):
        return self.elt_list.__getitem__(index)

    def add(self, value):
        """
        Add a value to the set.

        If not already in the set, the value will be appended to the list of elements.

        Args:
            value: The value to be added.

        """
        if value not in self:
            self.elt_list.append(value)
            super(OrderedSet, self).add(value)

    def update(self, values):
        """
        Add a sequence of values to the set.

        Args:
            values (sequence): Add each value in the sequence to the set.

        Returns:

        """
        if not isinstance(values, collections.Sequence):
            raise ValueError()
        for value in values:
            self.add(value)

    def clear(self):
        """
        Remove all elements from the set.
        """
        super(OrderedSet, self).clear()
        self.elt_list.clear()

    def union(self, values):
        """
        Return an OrderedSet which is the union of this set and a sequence of values.

        Args:
            values: A sequence of values.

        Returns:
            An OrderedSet containing the union of this set and the elements of values,
            with the elements of this set appearing before values in iteration.

        """
        result = OrderedSet(self)
        result.update(values)
        return result

    def pop(self):
        """
        Removes the last element added from the set.

        Returns:
            The last element added.

        """
        value = self.elt_list.pop()
        super(OrderedSet, self).remove(value)
        return value

    def remove(self, value):
        """
        Remove a value from the set.

        Args:
            value: The value to be removed.
        """
        self.elt_list.remove(value)
        super(OrderedSet, self).remove(value)
