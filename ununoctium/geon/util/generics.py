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

from functools import wraps


class TypeMethods(object):
    def __init__(self, base_method, **kvargs):
        super(TypeMethods, self).__init__(**kvargs)
        self.base_method = base_method
        self.methods = {}

    def on_type_wrapper(self, generic_function, dispatch_type):
        """
        Makes a wrapper to add the dispatch_type method to method_dispatch

        Arguments:
            generic_function: The generic function.
            dispatch_type: The dispatch type.

        Returns: An wrapper that adds a method for dispatch_type to generic_function

        """
        def add_method(method):
            """
            Adds f as the generic_function method for dispatch_type.

            Arguments:
                method: The method.

            Returns:

            """
            self.methods[dispatch_type] = method
            return generic_function

        return add_method

    def get_method(self, dispatch_arg):
        for t in type(dispatch_arg).__mro__:
            method = self.methods.get(t, None)
            if method is not None:
                return method
        return self.base_method


def generic_function(base_function):
    """
    Makes a function generic on its first argument's type.

    The default base function should be marked with @generic_function.
    Specialized handlers should be marked with @{base}.on_type(type)

    Example:

    .. code-block:: python

        @generic_function
        def visit(arg)
            print(arg)

        @visit.on_type(X):
        def visit(arg):
            print("Visiting an X")

        @visit.on_type(Y):
        def visit(arg):
            print("Visiting a Y")

    Arguments:
        base_function: Default implementation of the function.

    Returns:
        The generic function
    """
    type_methods = TypeMethods(base_function)

    @wraps(base_function)
    def generic(dispatch_arg, *args, **kwargs):
        """
        The generic function's implementation.

        Arguments:
            dispatch_arg: The argument that distpaches.
            *args: Remaining args.
            **kwargs: Keyword args.

        Returns: The result of the selected method.

        """
        return type_methods.get_method(dispatch_arg)(dispatch_arg, *args, **kwargs)

    def on_type(dispatch_type):
        """
        Marks the handler sub-method for when the first argument has type dispatch_type.

        Arguments:
            dispatch_type: The dispatch type.

        Returns: The wrapper for the method.

        """
        return type_methods.on_type_wrapper(generic, dispatch_type)

    generic.on_type = on_type

    return generic


def generic_method(base_method):
    """
    Makes a method generic on its first argument.

    A generic method is like a generic function, except that dispatch is on the type of the first
    non-self argument.  The first method, the default, should be marked with @generic_method.
    Specialized methods should be marked with @method.on_type(type)

    Example:

    .. code-block:: python

        class Visitor(object):
            def __init__(self, values):
                self.xs = []
                self.ys = []
                self.others = []

                for value in values:
                    self.visit(value)

            @generic_method
            def visit(self, arg)
                self.others.append(arg)

            @visit.on_type(X):
            def visit(self, arg):
                self.xs.append(arg)

            @visit.on_type(Y):
            def visit(self, arg):
                self.ys.append(arg)

    Arguments:
        base_method: Default implementation of the method.

    Returns:
        The generic method
    """
    type_methods = TypeMethods(base_method)

    @wraps(base_method)
    def generic(s, dispatch_arg, *args, **kwargs):
        """
        The generic method's implementation.

        Arguments:
            s: self.
            dispatch_arg: Argument that controls generic method selection.
            *args: Remaining positional args.
            **kwargs: Keyword args.

        Returns: The result of the selected method.

        """
        return type_methods.get_method(dispatch_arg)(s, dispatch_arg, *args, **kwargs)

    def on_type(dispatch_type):
        """
        Marks the handler sub-method for when the first argument has type dispatch_type.

        Arguments:
            dispatch_type: The dispatch type.

        Returns: The wrapper for the method.

        """
        return type_methods.on_type_wrapper(generic, dispatch_type)

    generic.on_type = on_type

    return generic
