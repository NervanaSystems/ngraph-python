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
# ----------------------------------------------------------------------------

from __future__ import division
from builtins import object, round
from collections import defaultdict
from orderedset import OrderedSet

from ngraph.op_graph.op_graph import as_op, ReturnOp, LiteralScalarOp

import numpy as np


class ElementType(object):
    """
    Element types for tensors.

    Provides element types independent of dtypes.

    Arguments:
        dtype: The dtype corresponding to this element type.
        ctype: The ctype corresponding to this element type.
        ptype: The python type corresponding to this element type.

    """

    dtype_map = {}
    ctype_map = {}
    ptype_map = {}

    def __init__(self,
                 ctype,
                 ptype,
                 **kwargs):
        super(ElementType, self).__init__(**kwargs)
        self.ctype = ctype
        self.ptype = ptype
        self.dtype = np.dtype(self.ptype)

        self.dtype_map[self.dtype] = self
        self.ctype_map[self.ctype] = self
        self.ptype_map[self.ptype] = self
        self.size = self.dtype.itemsize


# Define the supported element types
float32_t = ElementType('float32_t', np.float32)
float16_t = ElementType('float16_t', np.float16)
int32_t = ElementType('int32_t', np.int32)
int64_t = ElementType('int64_t', np.int64)
int8_t = ElementType('int8_t', np.int8)
uint32_t = ElementType('uint32_t', np.uint32)
uint64_t = ElementType('uint64_t', np.uint64)
uint8_t = ElementType('uint8_t', np.uint8)


def etype(x):
    """
    Universal translator to supported element types.

    Args:
        x: An element type.

    Returns:
        An ElementType. Raises ValueError for unsupported types.

    """
    if isinstance(x, ElementType):
        return x
    elif isinstance(x, np.dtype):
        return ElementType.dtype_map[x]
    elif isinstance(x, type):
        return ElementType.ptype_map[x]
    elif isinstance(x, str):
        return ElementType.ctype_map[x]
    else:
        raise ValueError("Invalid element type")


class ExecutionGraphElt(object):
    """
    An element of an exection graph.

    Arguments:
        execution_graph: The execution graph that indexes this exop.

    Attributes:
        execution_graph: The execution graph that indexes this exop.

    """

    def __init__(self, execution_graph, **kwargs):
        super(ExecutionGraphElt, self).__init__(**kwargs)
        self.execution_graph = execution_graph


class InputDecl(object):
    """
    Describes an input for an exop.

    Arguments:
        exop: The exop.
        pos: The position of the value, defaults to 0.
        tensor_description: Tensor description of the value. Describes the view.
        source_output_decl: The output_decl that supplies the value for this input.

    Attributes:
        exop: The exop.
        pos: The position of the value.
        tensor_view_decl: The tensor view where the value is read from.
        source_output_decl: The output_decl that supplies the value for this input.

    """

    def __init__(self,
                 exop=None,
                 pos=None,
                 tensor_description=None,
                 source_output_decl=None,
                 **kwargs):
        super(InputDecl, self).__init__(**kwargs)
        self.exop = exop
        self.pos = pos
        if tensor_description is None:
            tensor_description = source_output_decl.tensor_description
        self.__tensor_description = tensor_description
        self.__tensor_view_decl = None
        self.__source_output_decl = None
        self.source_output_decl = source_output_decl

    @property
    def tensor_description(self):
        """

        Returns:
            The TensorDescription associated with this InputDecl. Being phased out.

        """
        return self.__tensor_description

    @tensor_description.setter
    def tensor_description(self, tensor_description):
        """
        Being phased out.
        Args:
            tensor_description:

        Returns:

        """
        # assert self.__tensor_description.axes_key == tensor_description.axes_key
        self.__tensor_description = tensor_description

    @property
    def tensor_view_decl(self):
        return self.__tensor_view_decl

    @tensor_view_decl.setter
    def tensor_view_decl(self, tensor_view_decl):
        self.__tensor_view_decl = tensor_view_decl

    @property
    def tensor_decl(self):
        """

        Returns:
            The TensorDecl associated with the OutputDecl that supplies the value for this input.

        """
        return self.tensor_view_decl.tensor_decl

    @property
    def source_output_decl(self):
        """

        Returns:
            The OutputDecl that supplies a value for this InputDecl.

        """
        return self.__source_output_decl

    @source_output_decl.setter
    def source_output_decl(self, output_decl):
        """
        Changes the value assigned to this argument, updating value users.

        Args:
            output_decl: The new value for this argument.

        """
        if self.__source_output_decl is not None:
            self.__source_output_decl.user_input_decls.remove(self)
            self.__tensor_view_decl.readers.remove(self)
        if self.__source_output_decl is not None and output_decl is not None:
            self.__tensor_description = output_decl.tensor_description
        self.__source_output_decl = output_decl
        if output_decl is not None:
            output_decl.user_input_decls.add(self)
            self.__tensor_view_decl = \
                output_decl.tensor_view_decl.get_tensor_view(self.__tensor_description,
                                                             reader=self)

    def __repr__(self):
        return "Arg({exop}:{pos})".format(exop=self.exop.name, pos=self.pos)


class OutputDecl(object):
    """
    One value computed by an exop.

    Arguments:
        exop: The exop.
        pos: The position of the value, defaults to 0.
        tensor_decl: The TensorDecl where the output will be written.
        tensor_description: Tensor description describing the view of the tensor_decl.
        write_view: The tensor view where the value is written.

    Attributes:
        exop: The exop.
        pos: The position of the value.
        tensor_view_decl: The tensor view decl for where this output is written.
        user_input_decls: InputDecls using this output.
    """

    def __init__(self, exop=None, pos=None, tensor_decl=None, tensor_description=None, **kwargs):
        super(OutputDecl, self).__init__(**kwargs)
        self.exop = exop
        self.pos = pos
        self.__tensor_description = tensor_description
        self.__tensor = None
        self.__tensor_view_decl = None
        self.user_input_decls = set()
        self.tensor_decl = tensor_decl

    @property
    def tensor_description(self):
        """

        Returns:
            The TensorDescription associated with this OutputDecl. Being phased out as we
            replace tensor descriptions with TensorViewDecl/TensorViewLayoutDecl.

        """
        return self.__tensor_description

    @tensor_description.setter
    def tensor_description(self, tensor_description):
        """
        Being phased out.

        Args:
            tensor_description:

        """
        # assert self.__tensor_description.axes_key == tensor_description.axes_key
        self.__tensor_description = tensor_description

    @property
    def tensor_decl(self):
        """

        Returns:
            The TensorDecl associated with this output.

        """
        return self.__tensor

    @tensor_decl.setter
    def tensor_decl(self, tensor_decl):
        """
        Change the TensorDecl, updating tensor_view_decl in the process.

        Args:
            tensor_decl: The new TensorRecl.

        """
        if self.__tensor is tensor_decl:
            return
        if self.__tensor is not None:
            tensor_decl.merge_flags(self.__tensor)
        self.__tensor = tensor_decl
        self.tensor_view_decl = tensor_decl.get_tensor_view(self.__tensor_description, writer=self)

    @property
    def tensor_view_decl(self):
        """

        Returns:
            The TensorViewDecl for this output.

        """
        return self.__tensor_view_decl

    @tensor_view_decl.setter
    def tensor_view_decl(self, tensor_view_decl):
        """
        Change the TensorViewDecl associated with this output.

        Args:
            tensor_view_decl:

        Returns:

        """
        if tensor_view_decl is None and len(self.user_input_decls) > 0:
            raise ValueError("Cannot deallocate a view that is in use")
        self.__tensor_view_decl = tensor_view_decl
        tensor_view_decl.value = self
        if tensor_view_decl is not None:
            for input_decl in self.user_input_decls:
                input_decl.tensor_description = self.__tensor_description
                input_decl.tensor_view_decl = \
                    tensor_view_decl.get_tensor_view(input_decl.tensor_description,
                                                     reader=input_decl)

    def __repr__(self):
        return "Val({exop}:{pos})".format(exop=self.exop.name, pos=self.pos)


class ExOp(ExecutionGraphElt):
    """
    An exop that indicates an op to be executed.

    The op might be different from what was originally found in the computation graph.
    The args are exops that reflect the current version of the graph, and may differ
    from the exops of the op's args.
    The views_in are the current tensor views for the args.
    The views_out are the current tensor views for any results.

    Arguments:
        create_value: Create an output.
        op: The computation graph Op.
        computation_graph: The ComputationDecl owning this exop.
        prev_exop: The exop that precedes this op.
        next_exop: The exop that will follow this op.

    Attributes:
        op: The computation graph op to execute.
        input_decls: InputDecls for this exop.
        views_in: Views for the inputs.
        views_out: Views for the results.
        tensor_decl: Tensor of the primary output.
        tensor_view: View of the primary output.
        ref_ops: All computation graph ops covered by this op
        op_map: A map from ops to ref ops, sha

    """

    def __init__(self,
                 create_value=True,
                 op=None,
                 computation_graph=None,
                 prev_exop=None,
                 next_exop=None,
                 **kwargs):
        super(ExOp, self).__init__(execution_graph=computation_graph.execution_graph,
                                   **kwargs)
        self.__input_decls = []
        # Kludge until we have values with writers/readers
        self.write_args = []
        self.__output_decls = []
        self.computation_graph = computation_graph
        self.__op = None
        self.ref_ops = set()
        self.op = op
        self.prev_exop = prev_exop
        self.next_exop = next_exop
        self.liveness_live_list = []
        self.liveness_free_list = []
        self.liveness_new_list = []
        if self.op is not None:
            self.computation_graph.ops[self.op] = self
            self.add_ref_op(self.op)

        for arg in self.op.args:
            arg = arg.effective_tensor_op
            exop = self.computation_graph.get_exop(arg)
            output_decls = exop.output_decls[0]
            self.add_input_decl(source_output_decl=output_decls)

        if create_value and self.op.is_tensor_op:
            tensor_description = self.op.tensor_description()
            tensor_decl = self.computation_graph.get_tensor_decl(op=self.op)
            self.add_output_decl(tensor_decl, tensor_description)

    @property
    def input_decls(self):
        return self.__input_decls

    @property
    def output_decls(self):
        return self.__output_decls

    def add_input_decl(self, source_output_decl):
        input_decl = InputDecl(exop=self,
                               pos=len(self.__input_decls),
                               source_output_decl=source_output_decl)
        self.__input_decls.append(input_decl)
        return input_decl

    def add_write_arg(self, source_output_decl, tensor_description=None):
        """
        Temporary. Makes a pseudo-input; associated with WriteOp.

        Args:
            source_output_decl:
            tensor_description:

        Returns:

        """
        arg = InputDecl(exop=self,
                        pos=len(self.__input_decls),
                        source_output_decl=source_output_decl,
                        tensor_description=tensor_description)
        self.write_args.append(arg)
        return arg

    def add_output_decl(self, tensor_decl, tensor_description=None):
        """
        Adds an OutputDecl with a given TensorDecl and view description.

        Args:
            tensor_decl: Describes the tensor for the output.
            tensor_description: Describes the view to create.

        Returns:
            The new OutputDecl.

        """
        if tensor_description is None:
            tensor_description = tensor_decl.tensor_description_base
        output_decl = OutputDecl(exop=self,
                                 pos=len(self.output_decls),
                                 tensor_decl=tensor_decl,
                                 tensor_description=tensor_description)
        self.output_decls.append(output_decl)
        return output_decl

    def take_output_decl(self, output_decl):
        output_decl.exop = self
        output_decl.pos = len(self.output_decls)

    @property
    def op(self):
        """

        Returns:
            The op-graph Op associated with this exop.

        """
        return self.__op

    @op.setter
    def op(self, op):
        """
        Changes the op-graph Op assciated with this exop.

        Args:
            op: The new op-graph O.

        """
        if op is None:
            if self.__op is not None:
                raise ValueError("Cannot set op to None.")
            return
        if op.is_tensor_op:
            tensor_op = op.tensor
            if op is not tensor_op and not tensor_op.is_state_op:
                self.add_ref_op(op)
                op = tensor_op

        self.__op = op
        if op is not None:
            self.add_ref_op(op)

    def add_ref_op(self, op):
        """
        Add another op-graph Op that references this exop.

        Args:
            op: The computation graph op freferencing this exop.

        """
        self.ref_ops.add(op)
        self.computation_graph.ops[op] = self

    def memory_usage(self):
        """
        Get the memory usage of this op which is the sum of the sizes of all
        off the live tensors.

        Arguments:
          None

        Returns:
          Memory usage in bytes
        """
        size = 0
        for node in self.liveness_live_list:
            size += node.size
        return size

    def memory_footprint(self):
        """
        Get the total memory footprint of this op. The footprint hightest memory
        address used by the tensors in this op

        Arguments:
          None

        Returns:
          Memory footprint in bytes
        """
        max_mem = 0
        for node in self.liveness_live_list:
            if node.buffer_pool_offset is not None:
                offset = node.size + node.buffer_pool_offset
                max_mem = max([offset, max_mem])
        return max_mem

    def memory_efficiency(self):
        mem = 100
        if self.memory_footprint() > 0:
            mem = round(float(self.memory_usage()) / float(self.memory_footprint()) * 100)
            mem = int(mem)
        return mem

    @property
    def is_exop_end_of_list(self):
        """

        Returns:
            True if this represents the guard past the exop list. See ExOpBlock.

        """
        return False

    @property
    def name(self):
        return str(id(self)) if self.op is None else '{}'.format(self.op.name)

    def __repr__(self):
        return '{nn}:{id}\n\targs: {in_args}\n\tvalues: {out_args}\n\t\
live: {live}\n\tnew: {new}\n\tfree: {free}'.format(
            nn=self.op.name,
            id=id(self),
            in_args=", ".join([str(x.source_output_decl) for x in self.__input_decls]),
            out_args=", ".join([str(x) for x in self.output_decls]),
            live=self.liveness_live_list,
            new=self.liveness_new_list,
            free=self.liveness_free_list
        )


def literal_scalar_exop(scalar, computation_graph):
    """
    Creates an Exop for a scalar value.

    Args:
        scalar: The scalar value.
        computation_graph: The computation graph associated with the exop.

    Returns:
        An Exop.

    """
    exop = ExOp(computation_graph=computation_graph, op=LiteralScalarOp(scalar=scalar))
    exop.output_decls[0].tensor_decl.is_compile_only = True
    return exop


class ExOpBlock(ExecutionGraphElt):
    """
    A list of exops to be executed sequentially.

    Attributes:
        computation_graph: The associated computation graph.
        prev_exop: The latst exop.
        next_exop: The first exop.
        root_set: Set of exops whose values are needed.

    """

    def __init__(self, computation_graph=None, **kwargs):
        if computation_graph is None:
            raise ValueError("computation_graph must be specified.")
        super(ExOpBlock, self).__init__(execution_graph=computation_graph.execution_graph,
                                        **kwargs)
        self.computation_graph = computation_graph
        # Doubly linked loop, with self as termination
        self.prev_exop = self
        self.next_exop = self

        self.root_set = OrderedSet()

    @property
    def is_exop_end_of_list(self):
        """

        Returns:
            True if this represents the guard past the exop list. See ExecuteOp.

        """
        return True

    class ExOpForwardIterator(object):

        def __init__(self, exop_term):
            self.exop_term = exop_term
            self.exop = self.exop_term.next_exop

        def next(self):
            if self.exop.is_exop_end_of_list:
                raise StopIteration
            result = self.exop
            self.exop = result.next_exop
            return result

        __next__ = next  # Python 3.X compatibility

    class ExOpReversedIterator(object):

        def __init__(self, exop_term):
            self.exop_term = exop_term
            self.exop = self.exop_term.prev_exop

        def __iter__(self):
            return self

        def next(self):
            if self.exop.is_exop_end_of_list:
                raise StopIteration
            result = self.exop
            self.exop = result.prev_exop
            return result

        __next__ = next  # Python 3.X compatibility

    def __iter__(self):
        return ExOpBlock.ExOpForwardIterator(self)

    def __reversed__(self):
        return ExOpBlock.ExOpReversedIterator(self)

    def add_ops(self, roots, after_exop=None):
        """
        Add exops needed to compute ops in roots.

        Args:
            roots: A collection of ops whose values are needed.
            after_exop: Where in the list to add the ops. Defaults to the end.

        """
        if after_exop is None:
            after_exop = self.prev_exop
        # Get computation graph ops that have already been computed
        computed_ops = set()
        exop = after_exop
        while not exop.is_exop_end_of_list:
            computed_ops.add(exop.op)
            computed_ops.update(exop.ref_ops)
            computed_ops.update(input_decl.exop.op for input_decl in exop.input_decls)
            for input_decl in exop.input_decls:
                computed_ops.update(input_decl.source_output_decl.exop.ref_ops)
            exop = exop.prev_exop

        available = OrderedSet()
        counts = dict()
        parents = defaultdict(OrderedSet)
        ready = OrderedSet()

        available.update(roots)
        while available:
            op = available.pop()

            if op in counts or op in computed_ops:
                continue

            children = OrderedSet((child for child in op.all_deps if child not in computed_ops))
            if children:
                counts[op] = len(children)
                for child in children:
                    parents[child].add(op)
                available.update(children)
            else:
                ready.add(op)

        while ready:
            op = ready.pop()
            after_exop = self.add_op(op, after_exop=after_exop)
            for p in parents.get(op, []):
                count = counts[p] - 1
                if count == 0:
                    ready.add(p)
                    del counts[p]
                else:
                    counts[p] = count
        if len(counts) > 0:
            raise ValueError("Graph not a DAG")

    def add_op(self, op, after_exop):
        """
        Add an exop for op to be executed after after_exop.

        Args:
            op: The op.
            after_exop: The exop to precede op.

        Returns:
            The new last op. If the op is executable, it will be the added exop,
            othwerwise the previous after_exop.

        """
        if after_exop is None:
            after_exop = self
        if op.is_sequencing_op:
            return after_exop

        exec_op = ExOp(computation_graph=self.computation_graph, op=op)
        return self.add_exop(exec_op, after_exop)

    def add_exop(self, exop, after_exop=None):
        """
        Add exop to the list of exops, after after_exop.

        Args:
            exop:
                The exop to add.

            after_exop:
                If specified, the exop that should be added after after_exop. Defaults to the
                last exop added.

        Returns:
            The exop.

        """
        if after_exop is None:
            after_exop = self.prev_exop

        # Insert between after_exop and the op after after_exop
        before_exop = after_exop.next_exop

        # Add after after_exop
        after_exop.next_exop = exop
        exop.prev_exop = after_exop

        # Add before before_exop
        before_exop.prev_exop = exop
        exop.next_exop = before_exop

        return exop

    def move_exop_to_after_exop(self, exop, after_exop):
        exop.prev_exop.next_exop = exop.next_exop
        exop.next_exop.prev_exop = exop.prev_exop
        exop.prev_exop = after_exop
        exop.next_exop = after_exop.next_exop
        after_exop.next_exop = exop
        exop.next_exop.prev_exop = exop

    def remove_exop(self, exop):
        exop.prev_exop.next_exop = exop.next_exop
        exop.next_exop.prev_exop = exop.prev_exop
        for input_decl in exop.input_decls:
            input_decl.source_output_decl.user_input_decls.remove(input_decl)

    def replace_op(self, old_op, new_op):
        # TODO Replacing an op can remove ops. For example, (x + 2) * 1 -> x + 2
        # replaces the * with +, so * and 1 drop out
        # 1 dropping out means one less constant tensor, if it's not used
        # anywhere else
        # * dropping out means a change to sequencing.
        new_op = as_op(new_op)
        old_exop = self.computation_graph.get_exop(old_op)
        if old_op is new_op:
            # Hetr bashes some ops. See MutateInsteadOfCopyWithNewArgsMixin, issue #1410
            after_exop = old_exop.prev_exop
            self.remove_exop(old_exop)
            self.add_ops([new_op], after_exop=after_exop)
            return
        new_exop = self.computation_graph.get_exop(new_op, None)
        if new_exop is None:
            self.add_ops([new_op], after_exop=old_exop.prev_exop)
            new_exop = self.computation_graph.get_exop(new_op, None)
        self.replace_users(old_exop, new_exop)
        self.remove_exop(old_exop)
        if old_exop in self.root_set:
            self.root_set.remove(old_exop)
            self.root_set.add(new_exop)

    def replace_users(self, old_exop, new_exop):
        """
        Replace all users of old_exop with new_exop.

        Args:
            old_exop: The original exop.
            new_exop: The replaceent exop.

        """
        for old_output_decl, new_output_decl in zip(old_exop.output_decls, new_exop.output_decls):
            self.replace_output_decl(old_output_decl, new_output_decl)
        for op in old_exop.ref_ops:
            new_exop.add_ref_op(op)
        self.computation_graph.ops[old_exop.op] = new_exop

    def replace_output_decl(self, old_output_decl, new_output_decl):
        for input_decl in set(old_output_decl.user_input_decls):
            input_decl.source_output_decl = new_output_decl
        new_output_decl.tensor_decl.merge_flags(old_output_decl.tensor_decl)
        old_output_decl.exop.output_decls[old_output_decl.pos] = new_output_decl

    def replace_exop(self, old_exop, new_exop):
        self.add_exop(new_exop, old_exop.prev_exop)
        self.replace_users(old_exop, new_exop)
        self.remove_exop(old_exop)

    def merge_exop(self, old_exop, new_exop):
        """
        new_exop, which should already exist, takes over for old_exop.

        Args:
            old_exop:
            new_exop:

        """
        self.replace_users(old_exop, new_exop)
        self.remove_exop(old_exop)

    def memory_footprint(self):
        max_mem = 0
        for node in self:
            max_mem = max([node.memory_footprint(), max_mem])
        return max_mem

    def worst_case_footprint(self):
        mem = 0
        for var in self.get_temp_vars():
            mem += var.tensor_view_decl.tensor_decl.size
        return mem

    def memory_efficiency(self):
        footprint = self.memory_footprint()
        usage = 0
        for node in self.ops:
            usage = max(usage, node.memory_usage())
        result = 100
        if footprint > 0:
            result = int(round((float(usage) / float(footprint)) * 100))
        return result

    def persistent_size(self):
        mem = 0
        for var in self.get_persistent_vars():
            mem += var.tensor_view_decl.tensor_decl.size
        return mem

    def get_vars(self):
        vars = set()
        for exop in self:
            vars |= set(input_decl.source_output_decl for input_decl in exop.input_decls)
            vars |= set(exop.output_decls)
        return vars

    def get_temp_vars(self):
        result = list()
        for var in self.get_vars():
            if not var.tensor_view_decl.tensor_decl.is_persistent:
                result.append(var)
        return result

    def get_persistent_vars(self):
        result = list()
        for var in self.get_vars():
            if var.tensor_view_decl.tensor_decl.is_persistent:
                result.append(var)
        return result


class TensorDecl(ExecutionGraphElt):
    """
    Allocate for a tensor.

    Arguments:
        op: The AllocateTensorOp
        element_type: The type of the elements.
        size: The number of elements.
        is_persistent: True if the tensor is persistent.
        is_input: True if the tensor can be used as an argument.
        tensor_description_base: The base tensor description for the tensor.
        source_tensor: For a clone, the tensor that started the chain of clones
            this tensor is cloned from.

    Parameters:
        op: The AllocateTensorOp
        element_type: The type of the elements.
        size: The number of elements.
        is_persistent: True if the tensor is persistent.
        is_input: True if the tensor can be used as an argument.
        is_output: True if the tensor needs to be available for output. Defaults to is_persistent.
        tensor_descriptions: The set of tensor descriptions for the tensor.
        tensor_description_base: The tensor description base for this tensor.
        is_compile_only: If True, this tensor is only needed during compilation, and should not be
            allocated.
    """

    def __init__(self,
                 op,
                 element_type,
                 size,
                 is_persistent,
                 is_input,
                 tensor_description_base,
                 is_output=None,
                 is_constant=False,
                 tensor_description=None,
                 is_compile_only=False,
                 exop=None,
                 **kwargs):
        super(TensorDecl, self).__init__(**kwargs)
        self.op = op
        self.element_type = etype(element_type)
        self.size = size
        self.is_persistent = is_persistent
        self.is_input = is_input
        if is_output is None:
            is_output = is_persistent
        self.is_output = is_output
        self.buffer_pool_offset = None
        self.tensor_view_decls = {}
        self.tensor_description_base = tensor_description_base
        self.lifespan = None
        self.is_constant = is_constant
        self.is_compile_only = is_compile_only
        self.initial_value = None
        self.exop = exop
        if tensor_description is None:
            if op is None:
                tensor_description = self.tensor_description_base
            else:
                if op.tensor.is_state_op:
                    self.initial_value = op.tensor.initial_value
                tensor_description = op.tensor_description()
        self.root_tensor_view_decl = self.get_tensor_view(tensor_description)
        # TODO Needed for initialization. Use exop value instead.
        # self.add_value(self, tensor_description)
        self.source_tensor = self

    def get_tensor_view(self, tensor_description=None, reader=None, writer=None):
        """
        Get a view of this tensor.

        Args:
            tensor_description: Describes the view. Defaults to base tensor view.
            reader: If not None, reader is added to the view's readers.
            writer: If not None, writer is added to the view's writers.

        Returns:
            A tensor view.

        """
        if tensor_description is None:
            tensor_description = self.tensor_description_base
        tensor_view = self.tensor_view_decls.get(tensor_description.axes_key, None)
        if tensor_view is None:
            tensor_view = TensorViewDecl(self, tensor_description,
                                         execution_graph=self.execution_graph)
            self.tensor_view_decls[tensor_description.axes_key] = tensor_view
        if reader is not None:
            tensor_view.readers.add(reader)
        if writer is not None:
            tensor_view.writers.add(writer)
        return tensor_view

    def merge_flags(self, tensor):
        self.is_input |= tensor.is_input
        self.is_persistent |= tensor.is_persistent
        self.is_output |= tensor.is_output

    @property
    def buffer_key(self):
        """

        Returns: The key that makes this tensor unique in a buffer.

        """
        return self.tensor_description_base

    @property
    def prefix(self):
        if self.is_persistent:
            return "a_"
        return "a_{}".format(self.execution_graph.computation_decl.computation_op.name)

    @property
    def variable_name(self):
        return "{}_{}".format(self.prefix, self.tensor_name)

    @property
    def tensor_name(self):
        """

        Returns: Name used for the tensor.

        """
        return self.tensor_description_base.name

    @property
    def buffer_name(self):
        """

        Returns: Name used for the buffer.

        """
        return self.tensor_description_base.name

    @property
    def name(self):
        return str(id(self)) if self.op is None else '{}'.format(self.op.name)

    def __repr__(self):
        return self.tensor_description_base.name


class TensorViewDecl(ExecutionGraphElt):
    """
    Declare a view of a tensor.

    Arguments:
        tensor: The tensor.
        tensor_description: The description of the view.

    """

    def __init__(self,
                 tensor_decl,
                 tensor_description,
                 **kwargs):
        super(TensorViewDecl, self).__init__(**kwargs)
        self.tensor_decl = tensor_decl
        self.tensor_description = tensor_description
        self.initializers = OrderedSet()
        self.readers = OrderedSet()
        self.writers = OrderedSet()
        self.value = None

    @property
    def name(self):
        shape_str = "x".join((str(_) for _ in self.tensor_description.shape))
        return "{}_v_{}_{}".format(self.tensor_decl.variable_name,
                                   self.tensor_description.name,
                                   shape_str)

    @property
    def op(self):
        return self.tensor_decl.op

    def get_tensor_view(self, tensor_description, reader=None, writer=None):
        return self.tensor_decl.get_tensor_view(tensor_description, reader=reader, writer=writer)

    @property
    def json_dict(self):
        return {'id': self.tensor_description.name,
                'size': self.tensor_decl.size,
                'buffer_pool_offset': self.tensor_decl.buffer_pool_offset}

    @property
    def key(self):
        """
        Returns: A tuple unique to this view of the tensor.

        """
        return self.tensor_description.parameter_key


_default_default = []


class ComputationDecl(ExecutionGraphElt):
    """
    One computation to be run.

    Every computation has its own execution graph. Persistent tensors are shared
    between computations, other tensors are not.

    Attributes:
        computation: The computation op.
        ops: A map from ops to the exop that handles the op in this computation.
        exop: The SSA block of exops for this computation.
        values: The ops whose values are returned from the computation.
        tensors: Map from base tensor descriptions to tensors.

    """

    def __init__(self, computation_op, **kwargs):
        super(ComputationDecl, self).__init__(**kwargs)
        self.computation_op = computation_op
        self.ops = {}
        self.tensors = {}
        self.op_returns = {}

        # exops = []
        self.exop_block = ExOpBlock(computation_graph=self)
        self.exop_block.add_ops([self.computation_op])

        self.returns = ExOp(computation_graph=self, op=ReturnOp())
        self.exop_block.add_exop(self.returns, None)

        # Get the exops we need values for, so that if they are computed at compile-time we still
        # have a view to their value.
        self.exop_block.root_set = OrderedSet(
            self.get_exop(op) for op in computation_op.values if op.is_tensor_op)
        for exop in self.exop_block.root_set:
            for output_decl in exop.output_decls:
                input_decl = self.returns.add_input_decl(output_decl)
                self.op_returns[exop.op] = input_decl
                self.op_returns[exop.op.tensor] = input_decl
                output_decl.tensor_view_decl.tensor_decl.is_output = True

        self.values = set(self.get_exop(op) for op in computation_op.values
                          if op.tensor.is_tensor_op)

    def get_tensor_decl(self, op):
        return self.execution_graph.get_tensor_decl(op=op)

    def get_exop(self, op, default=_default_default):
        original_op = op
        op = op.effective_tensor_op
        if op.is_state_op:
            raise ValueError("Use get_tensor for state {}".format(original_op))
        exop = self.ops.get(op, None)
        if exop is not None:
            return exop
        if default is not _default_default:
            return default
        raise KeyError("Unhandled op: {}".format(original_op))


class ExecutionState(object):
    """
    Proxy for the state of a device.

    Arguments:
        transformer: The associated transformer.

    """

    def __init__(self, transformer=None, **kwargs):
        super(ExecutionState, self).__init__(**kwargs)
        self.__transformer = transformer
        self.__tensors_decls = {}

    @property
    def transformer(self):
        return self.__transformer

    def make_execution_graph(self, computation):
        return ExecutionGraph(self, computation)

    def get_op_tensor(self, op):
        tensor_description = op.tensor_description()
        tensor_description_base = tensor_description.base
        return self.__tensors_decls.get(tensor_description_base)

    def ensure_tensor_decl(self, execution_graph, tensor_description=None, op=None):
        tensor_description_base = tensor_description.base
        if tensor_description_base.op is None:
            raise ValueError(
                "Tensor description base {} has no Op".format(tensor_description_base))

        tensor_decl = self.__tensors_decls.get(tensor_description_base.op, None)
        if tensor_decl is None:
            tensor_decl = TensorDecl(op,
                                     element_type=etype(tensor_description_base.dtype),
                                     size=tensor_description_base.tensor_size,
                                     is_persistent=tensor_description_base.is_persistent,
                                     is_input=tensor_description_base.is_input,
                                     tensor_description_base=tensor_description_base,
                                     execution_graph=execution_graph)
            self.__tensors_decls[tensor_description_base.op] = tensor_decl
        return tensor_decl


class ExecutionGraph(object):
    """
    Information for compiling a computation_op.

    Arguments:
        execution_state: The execution state the graph will be applied to. The definitons in
            the execution state can be used in the execution graph.
        computation_op: A computation to be processed
    """

    def __init__(self, execution_state, computation_op, **kwargs):
        super(ExecutionGraph, self).__init__(**kwargs)
        self.execution_state = execution_state
        self.tensor_decls = {}
        self.computation_decl = ComputationDecl(computation_op=computation_op,
                                                execution_graph=self)

    def get_tensor_decl(self, tensor_description=None, op=None):
        if tensor_description is None:
            tensor_description = op.tensor_description()
        tensor_description_base = tensor_description.base
        if tensor_description_base.is_persistent:
            return self.execution_state.ensure_tensor_decl(self, tensor_description, op)
        if tensor_description_base.op is None:
            raise ValueError(
                "Tensor description base {} has no Op".format(tensor_description_base))
        tensor_decl = self.tensor_decls.get(tensor_description_base.op, None)
        if tensor_decl is None:
            tensor_decl = TensorDecl(op,
                                     element_type=etype(tensor_description_base.dtype),
                                     size=tensor_description_base.tensor_size,
                                     is_persistent=tensor_description_base.is_persistent,
                                     is_input=tensor_description_base.is_input,
                                     tensor_description_base=tensor_description_base,
                                     execution_graph=self)
            self.tensor_decls[tensor_description_base.op] = tensor_decl
        return tensor_decl
