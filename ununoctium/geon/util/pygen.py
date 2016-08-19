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

import tempfile
import re
import atexit
import os
from contextlib import contextmanager


@contextmanager
def indenting(code_writer):
    try:
        code_writer.indent(1)
        yield (code_writer)
    finally:
        code_writer.indent(-1)


class PyGen(object):
    def __init__(self, indentation=0, **kwargs):
        super(PyGen, self).__init__(**kwargs)
        self.indentation = indentation
        self.__code = ""

    def indent(self, indentation):
        self.indentation += indentation

    def get_arg_name(self, x):
        return x

    def append(self, code, *args, **kwargs):
        """
        Add code formatted with args and kwargs to generated code.
        :param code: String with {} formatting
        :param args: Format args for code.
        :param kwargs: Format kwargs for code.
        """
        nameargs = (self.name(arg) for arg in args)
        namekwargs = {k: self.name(v) for k, v in kwargs.items()}

        def indent(code):
            """
            Indent first line of code by 4n spaces.

            :param code:
            :param n:
            :return: Indented code with trailing space removed.
            """

            def remove_indentation(code):
                """
                Shift first line left to remove whitespace, shift other lines same amount.

                Code string should appear in file as '''
                some python
                    some more python
                    some more python
                '''
                Trailing space is also removed.

                :param code:
                :return:
                """
                p = re.search(r"(\n\W*)", code)
                if p:
                    code = re.sub(p.group(1), "\n", code)
                return code.strip()

            return re.sub(r"(^|\n)", r"\1" + " " * (4 * self.indentation),
                          remove_indentation(code))

        self.append_raw(indent(code.format(*nameargs, **namekwargs)))

    def append_raw(self, code, lines=1):
        self.endl(lines)
        self.__code += code

    def endl(self, n=1):
        self.__code += "\n" * n

    @property
    def code(self):
        return self.__code

    def compile(self, prefix):
        fd, filename = tempfile.mkstemp(".py", prefix, text=True)
        os.write(fd, self.code)
        os.close(fd)
        atexit.register(lambda: os.unlink(filename))

        code = compile(self.code, filename, "exec")
        r = {}
        exec (code, globals(), r)
        return r
