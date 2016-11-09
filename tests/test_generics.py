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


def generic_checker(check):
    check(object(), 3, 'base')
    check(3.4, 'foo', 'float')
    check(3, 4, 'int')
    check((2, 3), 11, 'tuple')
    check(A(), 1, 'A')
    check(B(), 2, 'A')
    check(C(), 3, 'C')


def test_generic_function():
    @generic_function()
    def selector(x, y):
        return ('base', x, y)

    @selector.on_type(float)
    def selector(x, y):
        return ('float', x, y)

    @selector.on_type(int)
    def selector(x, y):
        return ('int', x, y)

    @selector.on_type(tuple)
    def selector(x, y):
        return ('tuple', x, y)

    @selector.on_type(A)
    def selector(x, y):
        return ('A', x, y)

    @selector.on_type(C)
    def selector(x, y):
        return ('C', x, y)

    def check(x, y, tag):
        assert (tag, x, y) == selector(x, y)

    generic_checker(check)


class Visitor(object):
    @generic_method()
    def selector(self, x, y):
        return ('base', x, y)

    @selector.on_type(float)
    def selector(self, x, y):
        return ('float', x, y)

    @selector.on_type(int)
    def selector(self, x, y):
        return ('int', x, y)

    @selector.on_type(tuple)
    def selector(self, x, y):
        return ('tuple', x, y)

    @selector.on_type(A)
    def selector(self, x, y):
        return ('A', x, y)

    @selector.on_type(C)
    def selector(self, x, y):
        return ('C', x, y)


def test_generic_method():
    visitor = Visitor()

    def check(x, y, tag):
        assert (tag, x, y) == visitor.selector(x, y)

    generic_checker(check)
