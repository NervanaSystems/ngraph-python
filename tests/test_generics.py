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

from ngraph.util.generics import generic_function, generic_method


class A(object):
    pass


class B(A):
    pass


class C(A):
    pass


@generic_function()
def selector(x, y):
    return 'base'


@selector.on_type(float)
def selector(x, y):
    return 'float'


@selector.on_type(int)
def selector(x, y):
    return 'int'


@selector.on_type(tuple)
def selector(x, y):
    return 'tuple'


@selector.on_type(A)
def selector(x, y):
    return 'A'


@selector.on_type(C)
def selector(x, y):
    return 'C'


# Chain to selector
@selector.extension(next_method_arg=0)
def derived_selector(next_method, x, y):
    return 'rebase'


@derived_selector.on_type(A)
def derived_selector(next_method, x, y):
    return ('Arebase', next_method(x, y))


def test_generic_function():
    assert selector(object(), 3) is 'base'
    assert selector(3.4, 'foo') is 'float'
    assert selector(3, 4) is 'int'
    assert selector((2, 3), 11) is 'tuple'
    assert selector(A(), 1) is 'A'
    assert selector(B(), 2) is 'A'
    assert selector(C(), 3) is 'C'
    assert derived_selector(object(), 3) is 'rebase'
    assert derived_selector(3.4, 'foo') is 'float'
    assert derived_selector(3, 4) is 'int'
    assert derived_selector((2, 3), 11) is 'tuple'
    assert derived_selector(A(), 0) == ('Arebase', 'A')
    assert derived_selector(B(), 2) == ('Arebase', 'A')
    assert derived_selector(C(), 3) == 'C'


class Visitor(object):
    @generic_method()
    def selector(self, x, y):
        return 'base'

    @selector.on_type(float)
    def selector(self, x, y):
        return 'float'

    @selector.on_type(int)
    def selector(self, x, y):
        return 'int'

    @selector.on_type(tuple)
    def selector(self, x, y):
        return 'tuple'

    @selector.on_type(A)
    def selector(self, x, y):
        return 'A'

    @selector.on_type(C)
    def selector(self, x, y):
        return 'C'


class SubVisitor(Visitor):
    # Chain to Visitor.selector
    @Visitor.selector.extension()
    def selector(self, x, y):
        pass

    @selector.on_type(A, next_method_arg=1)
    def selector(self, next_method, x, y):
        return ('Sub', next_method(self, x, y))

    @selector.on_type(B)
    def selector(self, x, y):
        return 'BSub'

    # Chain to the function derived_selector
    @generic_method(extends=derived_selector)
    def f(self, next_method, x, y):
        pass

    @f.on_type(C, next_method_arg=1)
    def f(self, next_method, x, y):
        return ('f', next_method(x, y))


def test_generic_method():
    visitor = Visitor()
    assert visitor.selector(object(), 3) is 'base'
    assert visitor.selector(3.4, 'foo') is 'float'
    assert visitor.selector(3, 4) is 'int'
    assert visitor.selector((2, 3), 11) is 'tuple'
    assert visitor.selector(A(), 1) is 'A'
    assert visitor.selector(B(), 2) is 'A'
    assert visitor.selector(C(), 3) is 'C'
    subvisitor = SubVisitor()
    assert subvisitor.selector(A(), 1) == ('Sub', 'A')
    assert subvisitor.selector(B(), 1) == 'BSub'
    assert subvisitor.f(C(), 1) == ('f', 'C')
