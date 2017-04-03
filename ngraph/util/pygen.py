# ----------------------------------------------------------------------------
# Copyright 2017 Nervana Systems Inc.
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

from six import exec_
import tempfile
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
    """
    Helper for generating code to be run locally.

    Arguments:
        indentation: Initial indentation of code.
        prefix: Prefix for file.

    Attributes:
        indentation: Current indentation.
        prefix: File generation prefix.
        globals: The current environment.
        filenames: List of files to be deleted on exit.
        filename: Name of last generated file.

    """
    def __init__(self, indentation=0, prefix="", **kwargs):
        super(PyGen, self).__init__(**kwargs)
        self.indentation = indentation
        self.prefix = prefix
        self.globals = dict()
        self.filenames = []
        self.__code = list()
        self.filename = None
        self.indent_strings = ['', '    ', '        ', '            ']
        atexit.register(self.__exit_handler)

    def indent(self, indentation):
        """
        Increase indentation.

        Args:
            indentation: Amount to increase indentation.

        """
        self.indentation += indentation

    def get_arg_name(self, x):
        return x

    def name(self, x):
        raise NotImplementedError("Must be implemented by a subclass")

    def append(self, code, *args, **kwargs):
        """
        Add code formatted with args and kwargs to generated code.

        Arguments:
            code: String with {} formatting
            args: Format args for code.
            kwargs: Format kwargs for code.
        """
        nameargs = (self.name(arg) for arg in args)
        namekwargs = {k: self.name(v) for k, v in kwargs.items()}

        fcode = code.format(*nameargs, **namekwargs)
        indent_string = self.indent_strings[self.indentation]
        for line in iter(fcode.splitlines()):
            self.__code.append(indent_string)
            self.__code.append(line)
            self.__code.append('\n')

    def endl(self, n=1):
        """
        Add end of lines.

        Arguments:
            n: Number of end of lines. Defaults to 1.

        """
        self.__code.extend(["\n"] * n)

    def __exit_handler(self):
        for filename in self.filenames:
            os.unlink(filename)
        self.filenames = []

    @property
    def code(self):
        """

        Returns: The generated code.

        """
        return ''.join(self.__code)

    @property
    def code_length(self):
        """

        Returns: The number of code segments.

        """
        return len(self.__code)

    def execute(self, code=None):
        """
        Execute code directly into the environment. This is useful for initializing the
        environment with imports.

        Arguments:
            code: Something to execute.  Defaults to self.code.

        """
        if code is None:
            code = self.code
            self.__code = []
        exec_(code, self.globals)

    def evaluate(self, code=None):
        """
        Evaluate an expression in the environment.

        Arguments:
            code: An expression to evaluate. Defaults to self.code.

        Returns:
            The result of the evaluation.

        """
        if code is None:
            code = self.code
            self.__code = []
        return eval(code, self.globals)

    def write_to_file(self, file):
        """
        Writes self.code to a file.

        Args:
            file: The code.

        """
        for s in self.__code:
            file.write(s)
        self.__code = []

    def compile(self):
        """
        Compiles self.code and loads it into the environment.

        Returns: The updated environment.

        """
        file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix=self.prefix,
                                           delete=False)
        self.filename = file.name
        self.filenames.append(self.filename)
        self.write_to_file(file)
        file.close()

        with open(self.filename, 'r') as file:
            source = file.read()
            code = compile(source, self.filename, "exec")
            exec_(code, self.globals, self.globals)

        return self.globals
