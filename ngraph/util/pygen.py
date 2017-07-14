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


class PyModule(dict):
    """
    A Python module that can be executed, evaluates, and compiled into.

    Arguments:
        source: A dictionary to be added to the initial environment.
        prefix: Prefix for files compiled into the environment.

    Attributes:
        filename: The filename of the last file compiled into the environment.
        filenames: A list of all files compiled into the environment. These will
            be deleted at exit.
        prefix: The prefix for generated files.
    """
    def __init__(self, source=None, prefix=""):
        if source is not None:
            self.update(source)
        self.prefix = prefix
        self.filename = None
        self.filenames = []
        atexit.register(self.__exit_handler)

    def __exit_handler(self):
        for filename in self.filenames:
            os.unlink(filename)
        self.filenames = []

    def execute(self, code):
        """
        Execute code directly into the environment. This is useful for initializing the
        environment with imports.

        Arguments:
            code: Something to execute.

        """
        exec_(code, self, self)

    def evaluate(self, code):
        """
        Evaluate an expression in the environment.

        Arguments:
            code: An expression to evaluate.

        Returns:
            The result of the evaluation.

        """
        return eval(code, self)

    codegen_count = 0

    def compile(self, source):
        """
        Compiles self.code and loads it into the environment.

        Returns: The updated environment.

        """
        if False:
            # Set to True to get a file with all the generated code
            f = open('codegen.py', 'w' if PyModule.codegen_count == 0 else 'a')
            f.write('\n\n')
            f.write('#========================================================================\n')
            f.write('# code fragment {}\n'.format(PyModule.codegen_count))
            f.write('#========================================================================\n')
            f.write('\n\n')
            f.write(source)
            f.close()
            PyModule.codegen_count += 1

        file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', prefix=self.prefix,
                                           delete=False)
        self.filename = file.name
        self.filenames.append(self.filename)
        file.write(source)
        file.close()

        code = compile(source, self.filename, "exec")
        exec_(code, self, self)


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
        self.__code = list()
        self.indent_strings = ['', '    ', '        ', '            ']

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

    @property
    def code(self):
        """

        Returns: The generated code.

        """
        return ''.join(self.__code)

    def take_code(self):
        result = self.code
        self.__code = []
        return result

    @property
    def code_length(self):
        """

        Returns: The number of code segments.

        """
        return len(self.__code)
