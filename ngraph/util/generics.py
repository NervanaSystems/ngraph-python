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


_use_default = object()

# TODO @selector.on_type_next_method(X) with arg pos in extension.
# TODO List of dispatch args


class TypeMethods(object):
    """
    Maintains the mapping of types to method handlers.

    Also, keeps the `dispatch_base_type` which is the superclass that all types must subclass to
    dispatch on. (For safety).

    Arguments:
        dispatch_type: Function which returns the type to dispatch on.
        next_method_arg: If specified, an arg in this position will be passed which calls the
            next most specific method.
        dispatch_base_type: All dispatch types must be subtypes of this type. Defaults to object.
        extends: Extends this generic function.

    Attributes:
        dispatch_base_type: All dispatch types must be subtypes of this type.
        methods: Map from dispatch type to (method, next_method_arg) defined in this TypeMethods.
        type_cache: Cache from seen dispatch types to methods.
        super_type_methods: TypeMethods that inherit from this TypeMethod.
        extends: Generic function being extended. Must have type_method attribute.
        dispatch_type: Function which returns the type to dispatch on.
        next_method_arg: If not None, next_method will be passed in this position.
    """
    def __init__(self,
                 dispatch_type_fun,
                 next_method_arg=None,
                 dispatch_base_type=object,
                 extends=None,
                 **kvargs):
        super(TypeMethods, self).__init__(**kvargs)
        self.methods = {}
        self.type_cache = {}
        self.extends = extends
        self.super_type_methods = set()
        if extends is not None:
            self.extends.type_methods.super_type_methods.add(self)
        self.dispatch_base_type = dispatch_base_type
        self.dispatch_type_fun = dispatch_type_fun
        self.next_method_arg = next_method_arg

    def on_type_wrapper(self, generic_function, dispatch_type, next_method_arg):
        """
        Makes a wrapper to add the dispatch_type method to method_dispatch

        Arguments:
            generic_function: The generic function.
            dispatch_type: The dispatch type.
            next_method_arg: If _use_default, use the generic function default, otherwise override.

        Returns: A wrapper that adds a method for dispatch_type to generic_function

        """
        if next_method_arg is _use_default:
            next_method_arg = self.next_method_arg

        def add_method(method):
            """
            Adds f as the generic_function method for dispatch_type.

            Arguments:
                method: The method.

            Returns:

            """
            if not issubclass(dispatch_type, self.dispatch_base_type):
                raise ValueError("Dispatch type {} must be a subclass of `{}`"
                                 .format(dispatch_type, self.dispatch_base_type))

            self.methods[dispatch_type] = (method, next_method_arg)
            self.clear_cache()
            return generic_function

        return add_method

    def clear_cache(self):
        """
        Clears the dispatch cache.

        Called when a method is added.

        """
        self.type_cache = {}
        for sup in self.super_type_methods:
            sup.clear_cache()
        self.__all_methods = None

    def get_handler(self, dispatch_type):
        """
        Returns the method to handle dispatch_value.

        Args:
            dispatch_value: The argument that controls method selection.

        Returns:
            A callable.

        """
        try:
            return self.type_cache[dispatch_type]
        except KeyError:
            methods = []
            for t in dispatch_type.__mro__:
                f = self
                while True:
                    method, next_method_arg = f.methods.get(t, (None, None))
                    if next_method_arg is None:
                        next_method_arg = f.next_method_arg
                    if method:
                        methods.append((next_method_arg, method))
                    extends = f.extends
                    if extends is not None:
                        f = extends.type_methods
                    else:
                        break

            def make_wrapped_method(next_method_arg, next_method, method):
                """
                Inserts the next most specific method in passed args.

                Args:
                    next_method_arg: Position for next_method
                    next_method: Handler for next_method
                    method: The method to call.

                Returns:
                    Value returned from method.

                """
                @wraps(method)
                def wrapped_method(*args, **kwargs):
                    new_args = list(args[:next_method_arg])
                    new_args.append(next_method)
                    new_args.extend(args[next_method_arg:])
                    return method(*new_args, **kwargs)
                return wrapped_method

            def next_method(*args, **kwargs):
                """
                Raises ValueError if we don't find a method.

                Returns:

                """
                raise ValueError()

            for next_method_arg, method in reversed(methods):
                if next_method_arg is not None:
                    next_method = make_wrapped_method(next_method_arg, next_method, method)
                else:
                    next_method = method
            self.type_cache[dispatch_type] = next_method
            return next_method

    def __call__(self, base_function):
        self.methods[self.dispatch_base_type] = (base_function, None)

        @wraps(base_function)
        def generic(*args, **kwargs):
            """
            The generic function's implementation.

            Arguments:
                dispatch_arg: The argument that distpaches.
                *args: Remaining args.
                **kwargs: Keyword args.

            Returns: The result of the selected method.

            """
            return self.get_handler(self.dispatch_type_fun(*args, **kwargs))(*args, **kwargs)

        def on_type(dispatch_type, next_method_arg=_use_default):
            """
            Marks the handler sub-method for when the first argument has type dispatch_type.

            Arguments:
                dispatch_type: The dispatch type.

            Returns: The wrapper for the method.
            """
            return self.on_type_wrapper(generic, dispatch_type, next_method_arg=next_method_arg)

        def extension(next_method_arg=None, dispatch_type=None, dispatch_base_type=None):
            if next_method_arg is None:
                next_method_arg = self.next_method_arg
            if dispatch_base_type is None:
                dispatch_base_type = self.dispatch_base_type
            if dispatch_type is None:
                dispatch_type = self.dispatch_type_fun
            return TypeMethods(dispatch_type_fun=dispatch_type,
                               next_method_arg=next_method_arg,
                               dispatch_base_type=dispatch_base_type,
                               extends=generic)

        def next_method():
            raise TypeMethods.NextMethod()

        generic.on_type = on_type
        generic.type_methods = self
        generic.next_method = next_method
        generic.extension = extension
        return generic


def generic_function(dispatch_base_type=object, extends=None, next_method_arg=None):
    """
    Makes a function generic on its first argument's type.

    The default base function should be marked with @generic_function.
    Specialized handlers should be marked with @{base}.on_type(type)

    dispatch_base_type: The base class which all dispatch args must subclass from.
    extends: A generic function to extend.
    next_method_arg: If supplied, all methods will receive a callable in this position
        that calls the next most specific method.

    Example:

    .. code-block:: python

        @generic_function()
        def visit(arg)
            print(arg)

        @visit.on_type(X):
        def visit(arg):
            print("Visiting an X")

        @visit.on_type(Y):
        def visit(arg):
            print("Visiting a Y")

    Returns:
        The generic function
    """
    return TypeMethods(extends=extends,
                       dispatch_type_fun=lambda *args, **kwargs: type(args[0]),
                       dispatch_base_type=dispatch_base_type,
                       next_method_arg=next_method_arg)


def generic_method(dispatch_base_type=object, extends=None, next_method_arg=None):
    """
    Makes a method generic on its first argument.

    A generic method is like a generic function, except that dispatch is on the type of the first
    non-self argument.  The first method, the default, should be marked with @generic_method.
    Specialized methods should be marked with @method.on_type(type)

    dispatch_base_type: The base class which all dispatch args must subclass from.
    extends: A generic function extended by this generic function.
    next_method_arg: If supplied, all methods will receive a callable in this position
        that calls the next most specific method.

    Example:

    .. code-block:: python

        class Visitor(object):
            def __init__(self, values):
                self.xs = []
                self.ys = []
                self.others = []

                for value in values:
                    self.visit(value)

            @generic_method()
            def visit(self, arg)
                self.others.append(arg)

            @visit.on_type(X):
            def visit(self, arg):
                self.xs.append(arg)

            @visit.on_type(Y):
            def visit(self, arg):
                self.ys.append(arg)

    Returns:
        The generic method
    """
    return TypeMethods(extends=extends,
                       dispatch_type_fun=lambda self, *args, **kwargs: type(args[0]),
                       dispatch_base_type=dispatch_base_type,
                       next_method_arg=next_method_arg)
